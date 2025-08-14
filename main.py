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
from dataclasses import dataclass
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
            " Output JSON with keys 'label' and 'reason'."
            " The 'label' must be 1 for Yes or 0 for No."
            " Do not add extra keys. Do not paraphrase the input."
            " If uncertain, decide using the definitions below."
        )

        # Short, explicit instructions to stabilize outputs.
        self.class_def = (
            "Definition of hate speech: language that attacks a target based on protected characteristics."
            " Non-hate includes neutral mentions and context quoting without endorsement."
        )

    # ---------- Prompting ----------

    def zeroshot_prompting(self, dialogue: str) -> str:
        """
        Build a zero-shot classification prompt.
        """
        return self._format_prompt(header="Zero-shot classification", dialogue=dialogue)

    # def fewshot_prompting(
    #     self,
    #     manip_examples: Sequence[Example],
    #     nonmanip_examples: Sequence[Example],
    #     dialogue: str,
    # ) -> str:
    #     """
    #     Build a few-shot prompt with balanced positive and negative examples.
    #     """
    #     shots = []
    #     for ex in manip_examples:
    #         shots.append(self._format_single_example(ex, label=1))
    #     for ex in nonmanip_examples:
    #         shots.append(self._format_single_example(ex, label=0))

    #     exemplars = "\n\n".join(shots)
    #     body = (
    #         f"{self.system_preamble}\n\n"
    #         f"{self.class_def}\n\n"
    #         f"Here are labeled examples:\n{exemplars}\n\n"
    #         f"Now classify the final sample.\n"
    #         f"{dialogue}\n\n"
    #         "Return only JSON like: {\"label\": 0 or 1, \"reason\": \"...\"}"
    #     )
    #     return body

    def _format_prompt(self, header: str, dialogue: str) -> str:
        return (
            f"{self.system_preamble}\n\n"
            f"{self.class_def}\n\n"
            f"{header}\n"
            f"{dialogue}\n\n"
            "Return only JSON like: {\"label\": 0 or 1, \"reason\": \"...\"}"
        )

    # def _format_single_example(self, ex: Example, label: int) -> str:
    #     if self.prompt_style == "explicit_city":
    #         assert ex.city, "city is required for explicit_city examples"
    #         prefix = f"The person living in {ex.city} said: {ex.text}"
    #     else:
    #         assert ex.msa_population is not None and ex.msa_income is not None, \
    #             "msa_population and msa_income are required for implicit_msa examples"
    #         prefix = (
    #             "The person living in the metropolitan statistical area that has "
    #             f"{ex.msa_population} people and an average household income of "
    #             f"{ex.msa_income} dollars said: {ex.text}"
    #         )
    #     return f"Example:\nInput: {prefix}\nOutput: {{\"label\": {label}, \"reason\": \"given\"}}"

    # ---------- Data reading ----------
    
    @staticmethod
    def read_city_data_json(file_name: Optional[Path]) -> Dict[str, Tuple[int, int]]:
        """
        Reads city/MSA data from a nested JSON file (like MSAs.json).
        Returns mapping: city -> (msa_population, msa_income)
        """
        if file_name is None:
            return {}
        mapping: Dict[str, Tuple[int, int]] = {}
        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)
        print("Loaded JSON keys:", list(data.keys()))
        # Traverse countries and sizes
        for country_name, country in data.items():
            print(f"Country: {country_name}")
            for size, cities in country.items():
                print(f"  Size: {size}")
                if size == "Source":
                    continue
                if isinstance(cities, dict):
                    for city, info in cities.items():
                        print(f"    City: {city}, Info: {info}")
                        try:
                            pop = int(info["population"])
                            income_str = info["householdIncome"]
                            income_num = int(''.join(filter(str.isdigit, income_str)))
                            mapping[city] = (pop, income_num)
                        except Exception as e:
                            print(f"      Skipped {city} due to error: {e}")
                            continue
        print("Final mapping:", mapping)
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

            rows: List[Dict[str, str]] = []
            for r in ds:
                text = r.get("text")
                score = r.get("hate_speech_score")
                if text is None or score is None:
                    continue  # skip incomplete rows
                label = "1" if float(score) > threshold else "0"
                rows.append({"text": text, "label": label})
            return rows

    # @staticmethod
    # def read_city_data(file_name: Optional[Path]) -> Dict[str, Tuple[int, int]]:
    #     """
    #     Reads city metadata. Returns mapping:
    #         city -> (msa_population, msa_income)
    #     Expected columns:
    #         city, msa_population, msa_income
    #     Values are coerced to int. Rows with missing values are skipped.
    #     """
    #     print("DEBUG_POINT", inspect.currentframe().f_lineno)
    #     if file_name is None:
    #         return {}
    #     mapping: Dict[str, Tuple[int, int]] = {}
    #     with open(file_name, newline="", encoding="utf-8") as f:
    #         reader = csv.DictReader(f)
    #         required = {"city", "msa_population", "msa_income"}
    #         missing = required - set(reader.fieldnames or [])
    #         if missing:
    #             raise ValueError(f"Missing required columns in city data: {missing}")
    #         for r in reader:
    #             try:
    #                 city = r["city"]
    #                 pop = int(r["msa_population"])
    #                 inc = int(r["msa_income"])
    #                 mapping[city] = (pop, inc)
    #             except Exception:
    #                 continue
    #     return mapping

    def create_augmented_examples(
        self,
        base_data: List[Dict],
        city_data: Dict[str, Tuple[int, int]]
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
        augmented = []

        for sample in base_data:
            text = sample["text"]
            label = sample["label"]

            for city, (population, income) in city_data.items():
                if self.prompt_style == "explicit":
                    prompt = f"The person living in {city} said: {text}"
                elif self.prompt_style == "implicit":
                    prompt = (
                        f"The person living in the metropolitan statistical area "
                        f"that has {population} people and an average household income of "
                        f"{income} dollars said: {text}"
                    )
                else:
                    raise ValueError(f"Unknown prompt style: {self.prompt_style}")

                augmented.append({
                    "prompt": prompt,
                    "label": label
                })

        return augmented

    # ---------- Prediction ----------

    def build_dialogue(self,) -> str:
        """
        Constructs the input string according to the selected prompt style.
        """

    def predict_one(self, ):
        """
        Gets a single prediction. Returns 0 or 1.
        """
        return # Return something here

    @staticmethod
    def _postprocess_label(model_output: str) -> int:
        """
        Extracts a strict 0/1 from a JSON-like response.
        Tolerates minor formatting noise.
        """
        # Try JSON first
        try:
            obj = json.loads(model_output)
            val = int(obj.get("label"))
            return 1 if val == 1 else 0
        except Exception:
            pass

        # Fallback: regex search for "label": <0|1>
        m = re.search(r'"label"\s*:\s*(0|1)', model_output)
        if m:
            return int(m.group(1))

        # Fallback: keyword search
        text = model_output.lower()
        if "label" in text and "1" in text and "0" not in text:
            return 1
        if "label" in text and "0" in text and "1" not in text:
            return 0

        # Default to non-hate when undecidable to reduce false positives
        return 0

    # ---------- Batch prediction and evaluation ----------

    def prediction():
            return #preds, targets

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

# def _choose_fewshot(examples: List[Example], k_pos: int = 3, k_neg: int = 3) -> Tuple[List[Example], List[Example]]:
#     pos = [e for e in examples if e.label == 1][:k_pos]
#     neg = [e for e in examples if e.label == 0][:k_neg]
#     return pos, neg


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

    # # Test model before running experiment
    # test_prompt = "Say hello, this is a test."
    # resp = model.model.invoke([HumanMessage(content=test_prompt)])
    # print("Test response:", getattr(resp, "content", resp))
    # exit(0)

    exp = SESBiasExperiment(model=model, prompt_style=args.prompt_style)

    # 1. Load base hate speech dataset
    base_data = exp.read_base_data()

    # 2. Load city data (JSON only)
    city_data = exp.read_city_data_json(args.city_data)

    # 3. Expand into full experimental dataset
    examples = exp.create_augmented_examples(base_data, city_data)

    # 4. Optional: select few-shot seeds
    # few_pos, few_neg = [], []
    # if args.shot_mode == "few":
    #     few_pos, few_neg = _choose_fewshot(examples)

    # 5. Run predictions
    preds, targets = exp.prediction(
        data=examples,
        shot_mode=args.shot_mode,
        # few_pos=few_pos,
        # few_neg=few_neg,
    )

    # 6. Evaluate
    exp.evaluate(preds, targets, args.out_json)


if __name__ == "__main__":
    main()
