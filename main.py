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
import hashlib
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

# Import custom LLM wrappers
from openAI_llm import AnyOpenAILLM
from hf_local_llm import HFLocalLLM

# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)



PromptStyle = Literal["explicit", "implicit"]

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
        city_data: Dict[str, Tuple[int, str]],
        styles: List[str],
        shot_modes: List[str]
    ) -> List[Dict]:
        """
        Create augmented dataset by pairing each base sample with each city,
        formatting prompts for each selected style, and tagging with metadata.

        Args:
            base_data: List of dicts with keys {"text", "label"}
            city_data: Dict {city_name: (population, income)}
            styles: List of prompt styles to include (["explicit", "implicit"])
            shot_modes: List of shot modes to include (["few", "zero"])

        Returns:
            List[Dict] with keys {"base_idx", "style", "shot_mode", "prompt", "label"}
        """
        augmented = []

        for idx, sample in enumerate(base_data):
            text = sample["text"]
            label = sample["label"]

            for style in styles:
                for city, (population, income) in city_data.items():
                    if style == "explicit":
                        prompt = f"The person living in {city} said: {text}"
                    elif style == "implicit":
                        prompt = (
                            f"The person living in the metropolitan area "
                            f"that has {population} people and a median household income of "
                            f"{income} said: {text}"
                        )
                    else:
                        raise ValueError(f"Unknown prompt style: {style}")

                    # Create one augmented example for each shot mode
                    for mode in shot_modes:
                        augmented.append({
                            "base_idx": idx,
                            "style": style,
                            "shot_mode": mode,
                            "prompt": prompt,
                            "label": label
                        })

        return augmented
    
    @staticmethod
    def choose_fewshot(
        examples: List[Dict],
        num_per_class: int = 3,
        style: Optional[str] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Choose a few-shot balanced set from the augmented dataset.

        Args:
            examples: List of dicts with keys including {"prompt", "label", "style", "shot_mode"}.
            num_per_class: Number of examples to select for each label (0 and 1).
            style: Optional filter for style (e.g., "explicit" or "implicit").
            shot_mode: Optional filter for shot mode (e.g., "few" or "zero").
            seed: Random seed for reproducibility.

        Returns:
            few_pos: List of positive examples (label=1)
            few_neg: List of negative examples (label=0)
        """

        # Filtering by style
        filtered = examples
        if style is not None:
            filtered = [ex for ex in filtered if ex.get("style") == style]

        # Separate by label
        positives = [ex for ex in filtered if ex["label"] == 1]
        negatives = [ex for ex in filtered if ex["label"] == 0]

        random.shuffle(positives)
        random.shuffle(negatives)

        few_pos = positives[:num_per_class]
        few_neg = negatives[:num_per_class]

        return few_pos, few_neg

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
        # logging.warning(f"Undecidable model output: {repr(model_output)}")
        # Save to file for later review
        with open("undecidable_outputs.log", "a", encoding="utf-8") as f:
            f.write(model_output + "\n")
        return 0

    # ---------- Batch prediction and evaluation ----------
    def _build_prompt(self, prompt_text: str, shot_mode: str, few_pos: List[Dict], few_neg: List[Dict]) -> str:
        """Centralized prompt construction."""
        def label_to_text(label: int) -> str:
            return "Yes" if label == 1 else "No"

        if shot_mode == "few" and (few_pos or few_neg):
            few_shots_text = "".join(
                f"Example:\n{fs['prompt']}\nLabel: {label_to_text(fs['label'])}\n\n"
                for fs in (few_pos + few_neg)
            )
            return (
                f"{self.system_preamble}\n\n"
                f"{few_shots_text}"
                f"Now classify this:\n{prompt_text}"
            )

        return f"{self.system_preamble}\n\nClassify this:\n{prompt_text}"
    
    @staticmethod
    def args_hash(args):
        # Select relevant args to hash
        relevant = {
            "model_name": args.model_name,
            "prompt_style": args.prompt_style,
            "shot_mode": args.shot_mode,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            # Add more if needed
        }
        # Serialize and hash
        arg_str = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(arg_str.encode("utf-8")).hexdigest()[:8]  # Short hash

    def prediction(
        self,
        aug_data: List[Dict],
        base_data: List[Dict],
        styles: List[str],
        shot_modes: List[str],
        few_pos: List[Dict] = None,
        few_neg: List[Dict] = None
    ) -> List[Dict]:
        """
        Run predictions for base data and augmented data, grouped by base_idx/style/shot_mode.

        Args:
            aug_data: List of dicts with keys {"base_idx", "style", "shot_mode", "prompt", "label"}.
            base_data: List of dicts with keys {"text", "label"}.
            styles: Which styles to process.
            shot_modes: Which shot modes to process.
            few_pos: Few-shot positive examples.
            few_neg: Few-shot negative examples.

        Returns:
            List of dicts in the form:
            [
                {
                    "base": int,
                    "label": int,
                    "explicit": {
                        "few": {"preds": [...]},
                        "zero": {"preds": [...]}
                    },
                    "implicit": { ... }
                },
                ...
            ]
        """
        few_pos = few_pos or []
        few_neg = few_neg or []
        results = []
        history = []

        for base_idx, base_ex in enumerate(base_data):
            # Base prediction (always zero-shot)
            base_prompt = self._build_prompt(base_ex["text"], "zero", few_pos, few_neg)
            base_raw_output = self.model(base_prompt)
            base_pred = self._postprocess_label(str(base_raw_output))

            # Track base prompt/output
            history.append({
                "base_idx": base_idx,
                "type": "base",
                "prompt": base_prompt,
                "output": str(base_raw_output),
                "pred_label": base_pred,
                "label": base_ex["label"]
            })

            # Start entry
            entry = {
                "base": base_pred,
                "label": base_ex["label"]
            }
            # Prepare style/shot_mode containers
            for style in styles:
                entry[style] = {}
                for mode in shot_modes:
                    entry[style][mode] = {"preds": []}

            # Get matching augmented examples for this base
            matching_augs = [a for a in aug_data if a["base_idx"] == base_idx]

            # Process predictions
            for style in styles:
                for mode in shot_modes:
                    subset = [
                        a for a in matching_augs
                        if a["style"] == style and a["shot_mode"] == mode
                    ]
                    for ex in subset:
                        prompt = self._build_prompt(ex["prompt"], mode, few_pos, few_neg)
                        raw_output = self.model(prompt)
                        pred_label = self._postprocess_label(str(raw_output))
                        entry[style][mode]["preds"].append(pred_label)
                        # Track augmented prompt/output
                        history.append({
                            "base_idx": base_idx,
                            "type": "augmented",
                            "style": style,
                            "shot_mode": mode,
                            "prompt": prompt,
                            "output": str(raw_output),
                            "pred_label": pred_label,
                            "label": ex["label"]
                        })

            results.append(entry)

        return results, history

    @staticmethod
    def evaluate(
        results: List[Dict],
        json_filename: Path
    ) -> Dict[str, object]:
        """
        Evaluate predictions against ground truth and track flips relative to base predictions.

        Args:
            results: List of dicts from `prediction` function.
            json_filename: Path to save metrics JSON.

        Returns:
            metrics: Dict with evaluation metrics and flip stats.
        """
        # --- Flatten all base predictions and labels for standard metrics ---
        processed_preds = [entry["base"] for entry in results]
        processed_targets = [entry["label"] for entry in results]

        precision = precision_score(processed_targets, processed_preds, zero_division=0) * 100
        recall = recall_score(processed_targets, processed_preds, zero_division=0) * 100
        accuracy = accuracy_score(processed_targets, processed_preds) * 100
        micro_f1 = f1_score(processed_targets, processed_preds, average='micro', zero_division=0) * 100
        macro_f1 = f1_score(processed_targets, processed_preds, average='macro', zero_division=0) * 100
        conf_matrix = confusion_matrix(processed_targets, processed_preds)

        # --- Flip tracking ---
        flip_counts = defaultdict(int)
        total_counts = defaultdict(int)

        for entry in results:
            base_pred = entry["base"]
            for style, style_data in entry.items():
                if style in ("base", "label"):  # skip non-style keys
                    continue
                for mode, mode_data in style_data.items():
                    preds = mode_data.get("preds", [])
                    for p in preds:
                        total_counts[(style, mode)] += 1
                        if p != base_pred:
                            flip_counts[(style, mode)] += 1

        flip_stats = {
            f"{style}_{mode}": {
                "flips": flip_counts[(style, mode)],
                "total": total_counts[(style, mode)],
                "flip_rate": (
                    flip_counts[(style, mode)] / total_counts[(style, mode)] * 100
                    if total_counts[(style, mode)] > 0 else 0
                )
            }
            for style, mode in total_counts.keys()
        }

        # --- Logging ---
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
        logging.info(f"- Flip stats: {json.dumps(flip_stats, indent=2)}")

        # --- Save metrics ---
        metrics = {
            "precision": f"{precision:.2f}",
            "recall": f"{recall:.2f}",
            "accuracy": f"{accuracy:.2f}",
            "micro_f1": f"{micro_f1:.2f}",
            "macro_f1": f"{macro_f1:.2f}",
            "confusion_matrix": conf_matrix.tolist(),
            "flip_stats": flip_stats
        }
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Saved evaluation metrics to {json_filename}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="SES bias experiment for hate-speech classification")
    parser.add_argument("--backend", choices=["openai", "hf_local"], default="hf_local")
    parser.add_argument("--base_data", type=Path, required=False, help="base hate speach dataset -string -label")
    parser.add_argument("--city_data", type=Path, default=Path("MSAs.json"), help="City/MSA data JSON file (default: MSAs.json)")
    parser.add_argument("--prompt_style", choices=["explicit", "implicit"], default="implicit")
    parser.add_argument("--shot_mode", choices=["zero", "few"], default="few")
    parser.add_argument("--api_key", type=str, default=os.getenv("api_key", "ollama"))
    parser.add_argument("--api_base", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--model_type", choices=["chat", "completion"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--out_json", type=Path, default=Path("./metrics.json"), help="Output metrics JSON file.")

    # parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    # parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--model_name", type=str, default="tiiuae/Falcon-H1-1.5B-Instruct")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--hf_dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    parser.add_argument("--hf_device_map", type=str, default="auto")
    parser.add_argument("--hf_max_new_tokens", type=int, default=64)
    parser.add_argument("--hf_temperature", type=float, default=0.0)
    parser.add_argument("--hf_top_p", type=float, default=1.0)
    parser.add_argument("--hf_do_sample", action="store_true")
    parser.add_argument("--hf_load_in_4bit", action="store_true")
    parser.add_argument("--hf_trust_remote_code", action="store_true", default=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    seed = 42

    random.seed(seed)

    # Build model
    if args.backend == "openai":
        model = AnyOpenAILLM(
            model_type=args.model_type,
            model_name=args.model_name,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["\n"],  # moved out of model_kwargs
        )
    else:
        if not args.model_name:
            raise ValueError("When --backend hf_local, you must set --hf_model_name.")
        # Base kwargs for HFLocalLLM
        hf_kwargs = dict(
            model_name=args.model_name,
            hf_token=args.hf_token,  # or env HF_TOKEN
            dtype=args.hf_dtype,
            device_map=args.hf_device_map,
            max_new_tokens=args.hf_max_new_tokens,
            top_p=args.hf_top_p,
            do_sample=args.hf_do_sample,
            load_in_4bit=args.hf_load_in_4bit,
            trust_remote_code=args.hf_trust_remote_code,
        )

        # Only include temperature if sampling is enabled
        if args.hf_do_sample:
            hf_kwargs["temperature"] = args.hf_temperature

        model = HFLocalLLM(**hf_kwargs)

    exp = SESBiasExperiment(model=model, prompt_style=args.prompt_style)

    print("[INFO] Loading base hate speech dataset")
    # 1. Load base hate speech dataset
    base_data = exp.read_base_data()

    print("[INFO] Loading city data from JSON file")
    # 2. Load city data (JSON only)
    city_data = exp.read_city_data_json(args.city_data)

    print("[INFO] Expanding into full experimental dataset (creating augmented examples)")
    # 3. Expand into full experimental dataset
    # Get styles and shot modes from args
    styles = [args.prompt_style]
    shot_modes = [args.shot_mode]

    print("[INFO] Creating augmented examples")
    # Create augmented examples
    examples = exp.create_augmented_examples(base_data, city_data, styles, shot_modes)

    print("[INFO] Selecting few-shot seeds (if applicable)")
    # 4. Optional: select few-shot seeds
    few_pos, few_neg = [], []
    if args.shot_mode == "few":
        few_pos, few_neg = exp.choose_fewshot(examples, num_per_class=3, style=args.prompt_style)

    print("[INFO] Running predictions on base and augmented data")
    # 4. Run predictions
    results, history = exp.prediction(
        aug_data=examples,
        base_data=base_data,
        styles=styles,
        shot_modes=shot_modes,
        few_pos=few_pos,
        few_neg=few_neg,
    )

    hash_str = exp.args_hash(args)
    metrics_filename = f"outputs_{args.model_name.replace('/', '_')}_{hash_str}.json"
    history_filename = f"prompts_{args.model_name.replace('/', '_')}_{hash_str}.json"

    print(f"[INFO] Saving results to {metrics_filename}")
    # Save results with file name
    with open(metrics_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {metrics_filename}")

    print(f"[INFO] Saving prompt/response history to {history_filename}")
    # Save prompt/response history
    with open(history_filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logging.info(f"Saved prompt/response history to {history_filename}")

    print("[INFO] Evaluating predictions and saving metrics")
    # 5. Evaluate
    exp.evaluate(results, args.out_json)


if __name__ == "__main__":
    main()
