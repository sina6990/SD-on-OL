from token_shap.base import LocalModel, HuggingFaceEmbeddings, ModelBase
from token_shap.token_shap import TokenSHAP, StringSplitter, Splitter
import re
from typing import List
import torch
import pandas as pd

class ManualSplitter(Splitter):
    """
    Splits text into pre-defined chunks.
    Useful for analyzing the importance of specific phrases or sentences as whole units.
    """
    def __init__(self, chunks: List[str]):
        """
        Args:
            chunks (List[str]): List of strings that make up the full text. 
                                MUST concatenate exactly to the input text.
        """
        self.chunks = chunks
        
    def split(self, text: str) -> List[str]:
        # Verify reconstruction
        reconstructed = "".join(self.chunks)
        # We strip to handle potential whitespace differences if the user wasn't precise,
        # but ideally they should match exactly.
        if reconstructed.strip() != text.strip():
            print(f"Warning: Manual chunks do not exactly match input text.")
            print(f"Chunks joined: '{reconstructed}'")
            print(f"Input text:    '{text}'")
        return self.chunks
    
    def join(self, tokens: List[str]) -> str:
        return "".join(tokens)

class SystemPromptWrapper(ModelBase):
    """
    Wraps a model to prepend a system prompt to every generation request.
    This ensures the system prompt stays fixed while TokenSHAP analyzes only the user input.
    """
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = system_prompt
        # Inherit attributes from the base model if needed for compatibility
        self.model_name = getattr(model, 'model_name', 'SystemPromptWrapper')

    def generate(self, prompt, **kwargs):
        """
        Uses the model's tokenizer to apply the chat template if available.
        This ensures specific models (like Llama-3) respect the system prompt.
        """
        # Access the tokenizer from the underlying LocalModel
        tokenizer = getattr(self.model, 'tokenizer', None)
        
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            # specific fix for llama-3 if needed, but apply_chat_template handles most
            # we simply format it as a string to pass to the underlying generate method
            try:
                full_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: Could not apply chat template: {e}. Falling back to simple concatenation.")
                full_prompt = f"{self.system_prompt}\n{prompt}"
        else:
            full_prompt = f"{self.system_prompt}\n{prompt}"
            
        return self.model.generate(full_prompt, **kwargs)

class TokenShapRunner:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", device=None, system_prompt=None, splitter=None):
        """
        Initialize the TokenShapRunner with a local Hugging Face model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use. Default is "gpt2".
            device (str): Device to run the model on ('cpu', 'cuda', 'mps'). If None, auto-detects.
            system_prompt (str, optional): Fixed system instructions to prepend to inputs.
            splitter (Splitter, optional): Custom splitter for tokenization. If None, uses StringSplitter.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        print(f"Initializing LocalModel with {model_name} on {device}...")
        # Limiting max_new_tokens to 10 to prevent long hallucinations for Yes/No tasks
        self.base_model = LocalModel(
            model_name=model_name, 
            device=device,
            max_new_tokens=10 
        )
        
        # If system prompt is provided, wrap the model
        if system_prompt:
            print("Wrapping model with system prompt...")
            self.model = SystemPromptWrapper(self.base_model, system_prompt)
        else:
            self.model = self.base_model
        
        print("Initializing HuggingFaceEmbeddings...")
        self.vectorizer = HuggingFaceEmbeddings(device=device)
        
        if splitter:
            print("Using custom splitter...")
            self.splitter = splitter
        else:
            print("Initializing default StringSplitter...")
            self.splitter = StringSplitter()
        
        print("Initializing TokenSHAP...")
        self.token_shap = TokenSHAP(
            model=self.model, 
            vectorizer=self.vectorizer, 
            splitter=self.splitter,
            debug=True
        )

    def analyze(self, text, sampling_ratio=0.5, max_combinations=1000):
        """
        Analyze the given text using TokenSHAP to calculate Shapley values.
        
        Args:
            text (str): The sentence or text to analyze.
            sampling_ratio (float): Ratio of combinations to sample (0-1). Default is 0.5.
            max_combinations (int): Maximum number of unique combinations to process (default 1000).
            
        Returns:
            pd.DataFrame: DataFrame containing 'token' and 'shap_value'.
        """
        print(f"Analyzing text: '{text}' with sampling_ratio={sampling_ratio}, max_combinations={max_combinations}")
        _ = self.token_shap.analyze(
            prompt=text, 
            sampling_ratio=sampling_ratio,
            max_combinations=max_combinations,
            print_highlight_text=True
        )
        
        # Create a DataFrame from the calculated Shapley values
        shap_values = self.token_shap.shapley_values
        df = pd.DataFrame(list(shap_values.items()), columns=['token_with_idx', 'shap_value'])
        
        # Separate the raw token from its unique index 
        # (TokenSHAP appends _index to handle duplicates like two 'in's)
        def parse_token(token_str):
            if '_' in token_str:
                # Split from right to handle underscores in the token itself
                token, idx = token_str.rsplit('_', 1)
                # If token is just whitespace, label it for visibility
                if token.strip() == '':
                    DisplayToken = f"[SPACE] ({len(token)})"
                else:
                    DisplayToken = token
                return token, idx, DisplayToken
            return token_str, "", token_str

        parsed = df['token_with_idx'].apply(parse_token)
        df['token'] = [p[0] for p in parsed]
        df['index'] = [p[1] for p in parsed]
        df['display_token'] = [p[2] for p in parsed]
        
        # Sort by absolute shap_value descending for better visibility
        df['abs_shap_value'] = df['shap_value'].abs()
        df = df.sort_values(by='abs_shap_value', ascending=False).drop(columns=['abs_shap_value', 'token_with_idx'])
        
        # Reorder columns
        df = df[['display_token', 'shap_value', 'index']]
        
        return df

    def print_colored_text(self):
        """Print the text with tokens colored by their Shapley values."""
        self.token_shap.print_colored_text()


    
    def set_splitter(self, splitter):
        """Update the splitter used by TokenSHAP."""
        print("Updating splitter...")
        self.splitter = splitter
        self.token_shap.splitter = splitter




