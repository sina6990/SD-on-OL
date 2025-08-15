from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

class HFLocalLLM:
    """
    Local inference using Hugging Face Transformers.
    - Auto-downloads model and tokenizer from the Hub.
    - Uses HF token if provided or from env(HF_TOKEN).
    - Supports optional 4-bit quantization.
    - Returns plain string text generations.
    """

    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        dtype: Literal["auto", "bfloat16", "float16"] = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # Resolve auth token: explicit arg wins, else env HF_TOKEN if present
        token = hf_token or os.getenv("HF_TOKEN", None)

        # Choose dtype
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = "auto"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        # Load model with optional 4-bit quantization
        model_kwargs = dict(
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        if load_in_4bit:
            # 4-bit quantization configuration
            model_kwargs.update(
                dict(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype != "auto" else torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            )
            torch_dtype = None  # handled by bitsandbytes path
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            **model_kwargs,
        )

        # Simple text-generation pipeline works for prompt-style classification
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device_map,
        )

    def __call__(self, prompt: str) -> str:
        # Return only the generated continuation, not echo of the prompt
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            return_full_text=False,
        )
        return out[0]["generated_text"]
