# Purpose: Run zero-shot and few-shot LLM classification experiments that test whether
#          explicit or implicit socioeconomic cues affect hate-speech classification.
#
#
# Notes:
# - The model wrapper supports both chat and completion endpoints through LangChain.
# - All generation prompts ask for a strict JSON answer with fields: label (0|1) and reason.

from __future__ import annotations

# Debugging
import inspect

import argparse
import csv
import json
import logging
import os
import re
import random
from datasets import load_dataset
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage

# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PromptStyle = Literal["explicit", "implicit"]


class AnyOpenAILLM:
    """
    Minimal wrapper around LangChain's ChatOpenAI/OpenAI that matches your prior code.
    Falls back with a clear error if LangChain is unavailable.
    """

    def __init__(self, *args, **kwargs):
        self.model_type = kwargs.get("model_type", "chat")
        model_name = kwargs.get("model_name")
        api_key = kwargs.get("api_key", "ollama")
        api_base = kwargs.get("api_base", "http://localhost:1234/v1")
        temperature = kwargs.get("temperature", 0)
        max_tokens = kwargs.get("max_tokens", 10048)
        model_kwargs = kwargs.get("model_kwargs", {})

        if self.model_type == "completion":
            self.model = OpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                **model_kwargs,
            )
        else:
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )

    def __call__(self, prompt: str) -> str:
        if self.model_type == "completion":
            return str(self.model(prompt))
        else:
            return str(self.model([HumanMessage(content=prompt)]).content)

class SESBiasExperiment:
    """
    Encapsulates data ingestion, prompt construction, prediction, and evaluation.
    """

    def __init__(
        self,
        model: AnyOpenAILLM,
        prompt_style: PromptStyle,
        system_preamble: Optional[str] = None,
    ):
        self.model = model
        self.prompt_style = prompt_style
        # System preamble conditions classification without content generation.
        self.system_preamble = system_preamble or (
            "You are a classifier. Task: decide whether the given utterance contains hate speech."
            " Answer with only one word: Yes or No."
        )

    # ---------- Data reading ----------
    
    @staticmethod
    def read_city_data_json(file_name: Optional[Path]) -> Dict[str, Tuple[int, str]]:
        """
        Reads city/MSA data from a nested JSON file (like MSAs.json).
        Returns mapping: city -> (msa_population, msa_income)
        """
        if file_name is None:
            return {}
        mapping: Dict[str, Tuple[int, str]] = {}
        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)
        # Traverse countries and sizes
        for country_name, country in data.items():
            for size, cities in country.items():
                if size == "Source":
                    continue
                if isinstance(cities, dict):
                    for city, info in cities.items():
                        try:
                            pop = int(info["population"])
                            income_str = info["householdIncome"]
                            mapping[city] = (pop, income_str)
                        except Exception as e:
                            print(f"      Skipped {city} due to error: {e}")
                            continue

        return mapping

    @staticmethod
    def read_base_data(dataset_name: str = "ucberkeley-dlab/measuring-hate-speech",
            split: str = "train",
            threshold: float = 0.5,
        ) -> List[Dict[str, str]]:
            """
            Returns rows shaped like the CSV loader:
            - text (string)
            - label ("0" or "1" as strings, to match your existing type hints)
            """
            print("DEBUG_POINT", inspect.currentframe().f_lineno)
            ds = load_dataset(dataset_name, split=split)

            # ds = ds.select(range(5))

            rows: List[Dict[str, str]] = []
            for r in ds:
                text = r.get("text")
                score = r.get("hate_speech_score")
                if text is None or score is None:
                    continue  # skip incomplete rows
                label = 1 if float(score) > threshold else 0
                rows.append({"text": text, "label": label})
            return rows

    def create_augmented_examples(
        self,
        base_data: List[Dict],
        city_data: Dict[str, Tuple[int, str]]
    ) -> List[Dict]:
        """
        Create augmented dataset by pairing each base sample with each city
        and formatting prompts according to the selected style.
        
        Args:
            base_data: List of dicts with keys {"text", "label"}
            city_data: Dict {city_name: (population, income)}
        
        Returns:
            List[Dict] with keys {"prompt", "label"}
        """
        print("DEBUG_POINT", inspect.currentframe().f_lineno)
        augmented = []

        for sample in base_data:
            text = sample["text"]
            label = sample["label"]

            for city, (population, income) in city_data.items():
                if self.prompt_style == "explicit":
                    prompt = f"The person living in {city} said: {text}"
                elif self.prompt_style == "implicit":
                    prompt = (
                        f"The person living in the metropolitan area "
                        f"that has {population} people and a median household income of "
                        f"{income} said: {text}"
                    )
                else:
                    raise ValueError(f"Unknown prompt style: {self.prompt_style}")

                augmented.append({
                    "prompt": prompt,
                    "label": label
                })

        return augmented

    # ---------- Prediction ----------

    @staticmethod
    def _postprocess_label(model_output: str) -> int:
        """
        Converts model output to 0 (No) or 1 (Yes).
        Handles 'Yes'/'No' answers robustly.
        """
        # Normalize output
        text = model_output.strip().lower()
        # Remove punctuation
        text = text.strip(' .!?,;:\'"')

        # Direct match
        if text == "yes":
            return 1
        if text == "no":
            return 0

        # Partial match (e.g., "yes.", "no!", "the answer is yes")
        words = text.split()
        if "yes" in words:
            return 1
        if "no" in words:
            return 0

        # Default to non-hate when undecidable to reduce false positives
        logging.warning(f"Undecidable model output: {repr(model_output)}")
        # Save to file for later review
        with open("undecidable_outputs.log", "a", encoding="utf-8") as f:
            f.write(model_output + "\n")
        return 0

    # ---------- Batch prediction and evaluation ----------
    def prediction(
        self,
        data: List[Dict],
        shot_mode: str,
        few_pos: List[Dict] = None,
        few_neg: List[Dict] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Run predictions on augmented data.

        Args:
            data: List of {"prompt", "label"} dicts.
            shot_mode: "zero" or "few".
            few_pos: Few-shot positive examples (label=1).
            few_neg: Few-shot negative examples (label=0).

        Returns:
            preds: List of predicted labels.
            targets: List of ground truth labels.
        """
        print("DEBUG_POINT", inspect.currentframe().f_lineno)
        preds = []
        targets = []

        # Ensure lists are not None
        few_pos = few_pos or []
        few_neg = few_neg or []

        for example in data:
            # Build prompt
            if shot_mode == "few" and (few_pos or few_neg):
                few_shots_text = ""
                for fs in few_pos + few_neg:
                    few_shots_text += f"Example:\n{fs['prompt']}\nLabel: {fs['label']}\n\n"
                full_prompt = (
                    f"{self.system_preamble}\n{self.class_def}\n\n"
                    f"{few_shots_text}"
                    f"Now classify this:\n{example['prompt']}"
                )
            else:
                full_prompt = (
                    f"{self.system_preamble}\n\n"
                    f"Classify this:\n{example['prompt']}"
                )

            # Send to model
            resp = self.model.model.invoke([HumanMessage(content=full_prompt)])
            raw_output = getattr(resp, "content", resp)

            # Try to parse JSON
            try:
                parsed = json.loads(raw_output)
                pred_label = self._postprocess_label(parsed.get("label", ""))
            except (json.JSONDecodeError, AttributeError):
                # Handle unexpected format
                pred_label = self._postprocess_label(str(raw_output))

            preds.append(pred_label)
            targets.append(example["label"])

        return preds, targets

    @staticmethod
    def evaluate(processed_preds: Sequence[int], processed_targets: Sequence[int], json_filename: Path) -> Dict[str, object]:
        precision = precision_score(processed_targets, processed_preds, zero_division=0) * 100
        recall = recall_score(processed_targets, processed_preds, zero_division=0) * 100
        accuracy = accuracy_score(processed_targets, processed_preds) * 100
        micro_f1 = f1_score(processed_targets, processed_preds, average='micro', zero_division=0) * 100
        macro_f1 = f1_score(processed_targets, processed_preds, average='macro', zero_division=0) * 100
        conf_matrix = confusion_matrix(processed_targets, processed_preds)

        logging.info(
            f"Total hate = {len([v for v in processed_targets if v == 1])}, "
            f"non-hate = {len([v for v in processed_targets if v == 0])}"
        )
        logging.info(
            f"Predicted hate = {len([v for v in processed_preds if v == 1])}, "
            f"non-hate = {len([v for v in processed_preds if v == 0])}"
        )
        logging.info(f"- Precision = {precision:.2f}")
        logging.info(f"- Recall = {recall:.2f}")
        logging.info(f"- Accuracy = {accuracy:.2f}")
        logging.info(f"- Micro F1-Score = {micro_f1:.2f}")
        logging.info(f"- Macro F1-Score = {macro_f1:.2f}")
        logging.info(f"- Confusion Matrix = \n{conf_matrix}")

        metrics = {
            "precision": f"{precision:.2f}",
            "recall": f"{recall:.2f}",
            "accuracy": f"{accuracy:.2f}",
            "micro_f1": f"{micro_f1:.2f}",
            "macro_f1": f"{macro_f1:.2f}",
            "confusion_matrix": conf_matrix.tolist(),
        }
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Saved evaluation metrics to {json_filename}")
        return metrics


# ---------------- CLI ----------------

def _choose_fewshot(
    examples: List[Dict],
    num_per_class: int = 3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Choose a few-shot balanced set from the augmented dataset.

    Args:
        examples: List of dicts with keys {"prompt", "label"}.
        num_per_class: Number of examples to select for each label (0 and 1).
        seed: Random seed for reproducibility.

    Returns:
        few_pos: List of positive examples (label=1)
        few_neg: List of negative examples (label=0)
    """
    # Separate by label
    positives = [ex for ex in examples if ex["label"] == 1]
    negatives = [ex for ex in examples if ex["label"] == 0]

    # Shuffle and pick desired amount
    random.shuffle(positives)
    random.shuffle(negatives)

    few_pos = positives[:num_per_class]
    few_neg = negatives[:num_per_class]

    return few_pos, few_neg


def main():
    parser = argparse.ArgumentParser(description="SES bias experiment for hate-speech classification")
    parser.add_argument("--base_data", type=Path, required=False, help="base hate speach dataset -string -label")
    parser.add_argument("--city_data", type=Path, default=Path("MSAs.json"), help="City/MSA data JSON file (default: MSAs.json)")
    parser.add_argument("--prompt_style", choices=["explicit", "implicit"], default="implicit")
    parser.add_argument("--shot_mode", choices=["zero", "few"], default="zero")
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--api_key", type=str, default=os.getenv("api_key", "ollama"))
    parser.add_argument("--api_base", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--model_type", choices=["chat", "completion"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--out_json", type=Path, default=Path("./metrics.json"), help="Output metrics JSON file.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    seed = 42

    random.seed(seed)

    # Build model
    model = AnyOpenAILLM(
        model_type=args.model_type,
        model_name=args.model_name,
        api_key=args.api_key,
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["\n"],  # moved out of model_kwargs
    )
    print("DEBUG_POINT", inspect.currentframe().f_lineno)

    # Test model before running experiment
    # test_prompt = "Say hello, this is a test."
    # resp = model.model.invoke([HumanMessage(content=test_prompt)])
    # print("Test response:", getattr(resp, "content", resp))

    exp = SESBiasExperiment(model=model, prompt_style=args.prompt_style)

    # 1. Load base hate speech dataset
    base_data = exp.read_base_data()

    # 2. Load city data (JSON only)
    city_data = exp.read_city_data_json(args.city_data)

    # 3. Expand into full experimental dataset
    examples = exp.create_augmented_examples(base_data, city_data)

    # 4. Optional: select few-shot seeds
    few_pos, few_neg = [], []
    # if args.shot_mode == "few":
    few_pos, few_neg = _choose_fewshot(examples)

    # 5. Run predictions
    preds, targets = exp.prediction(
        data=examples,
        shot_mode=args.shot_mode,
        few_pos=few_pos,
        few_neg=few_neg,
    )

    # 6. Evaluate
    exp.evaluate(preds, targets, args.out_json)


if __name__ == "__main__":
    main()
