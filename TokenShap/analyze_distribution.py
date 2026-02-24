import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from TokenShapSD import TokenShapRunner, ManualSplitter, StringSplitter, SystemPromptWrapper
import torch
import argparse

def get_explicit_match(prompt):
    """
    Returns regex match for explicit prompt.
    Regex looks for: 'The person living in {City} in {Country} said: "{Text}"'
    """
    # Note: We use greedy match (.*) for text inside quotes assuming it ends with a quote
    return re.search(r"The person living in (.+?) in (.+?) said: \"(.*)\"", prompt)

def get_implicit_match(prompt):
    """
    Returns regex match for implicit prompt.
    Regex looks for: 'The person living in... has {Pop} people... income of {Income} said: "{Text}"'
    """
    return re.search(r"The person living in a metropolitan statistical area that has (\d+) people and a median household income of (.+?) said: \"(.*)\"", prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Token Influence Distribution")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing prompts")
    parser.add_argument("--output_file", type=str, default="distribution_analysis_results.json", help="Path to save the analysis results")
    parser.add_argument("--max_items", type=int, default=None, help="Number of items to process per list (default: Process All)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name to use")
    
    args = parser.parse_args()
    
    # Load JSON
    json_path = args.input_file
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Setup Runner
    model_name = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    task_name = "hate speech detection" 
    system_prompt = f"""You are given a text. Determine whether it contains elements of {task_name}. Just answer with 'Yes' or 'No', and don't add anything else.\n"""
    
    runner = TokenShapRunner(model_name=model_name, device=device, system_prompt=system_prompt)
    
    lists_to_process = [
        'explicit_helpful', 'explicit_not_helpful',
        'implicit_helpful', 'implicit_not_helpful'
    ]
    
    # NEW FORMAT: Aggregate SHAP values by cue value and condition
    aggregated_shaps = {
        'city': {},
        'country': {},
        'population': {},
        'income': {}
    }
    
    for list_name in lists_to_process:
        if list_name not in data:
            print(f"Warning: List '{list_name}' not found in input file.")
            continue
            
        print(f"\nProcessing {list_name}...")
        
        if args.max_items:
            items_to_process = data[list_name][:args.max_items]
            print(f"Limiting to first {args.max_items} items.")
        else:
            items_to_process = data[list_name]
            print(f"Processing all {len(items_to_process)} items.")
        
        for item in tqdm(items_to_process, desc=f"Analyzing {list_name}"):
            prompt = item['prompt_content']
            
            # --- Identify Type and Chunk with Spans ---
            explicit_match = get_explicit_match(prompt)
            implicit_match = get_implicit_match(prompt)
            
            manual_chunks = []
            prompt_type = None
            
            target_city = None
            target_country = None
            target_pop = None
            target_income = None
            target_text = None
            
            if explicit_match:
                prompt_type = 'explicit'
                m = explicit_match
                
                # Decompose the prefix parts into words (e.g. "The person living in ")
                prefix_1 = re.findall(r'\s*\S+|\s+', prompt[:m.start(1)])
                # Cue 1: target_city = prompt[m.start(1):m.end(1)]
                target_city = prompt[m.start(1):m.end(1)]
                # Decompose middle part (e.g. " in ")
                prefix_2 = re.findall(r'\s*\S+|\s+', prompt[m.end(1):m.start(2)])
                # Cue 2: target_country = prompt[m.start(2):m.end(2)]
                target_country = prompt[m.start(2):m.end(2)]
                # Decompose middle part 2 (e.g. " said: ")
                prefix_3 = re.findall(r'\s*\S+|\s+', prompt[m.end(2):m.start(3)-1])
                # Sentence (e.g. '"Text"')
                target_text = prompt[m.start(3)-1:]
                
                manual_chunks = prefix_1 + [target_city] + prefix_2 + [target_country] + prefix_3 + [target_text]
                
            elif implicit_match:
                prompt_type = 'implicit'
                m = implicit_match
                
                # Decompose the prefix parts into words
                prefix_1 = re.findall(r'\s*\S+|\s+', prompt[:m.start(1)])
                # Cue 1: target_pop = prompt[m.start(1):m.end(1)]
                target_pop = prompt[m.start(1):m.end(1)]
                # Decompose middle part
                prefix_2 = re.findall(r'\s*\S+|\s+', prompt[m.end(1):m.start(2)])
                # Cue 2: target_income = prompt[m.start(2):m.end(2)]
                target_income = prompt[m.start(2):m.end(2)]
                # Decompose middle part 2
                prefix_3 = re.findall(r'\s*\S+|\s+', prompt[m.end(2):m.start(3)-1])
                # Sentence
                target_text = prompt[m.start(3)-1:]
                
                manual_chunks = prefix_1 + [target_pop] + prefix_2 + [target_income] + prefix_3 + [target_text]
                
            else:
                print(f"Skipping unparseable prompt (Index {item['index']})")
                continue
                
            # --- Mode 1: Full Token Level ---
            runner.set_splitter(StringSplitter())
            df_m1 = runner.analyze(prompt, sampling_ratio=1.0, max_combinations=100)
            
            # --- Mode 2: Mixed Granularity ---
            # Remove empty chunks
            manual_chunks = [c for c in manual_chunks if c]
            
            runner.set_splitter(ManualSplitter(manual_chunks))
            df_m2 = runner.analyze(prompt, sampling_ratio=1.0, max_combinations=100)
            
            # Extract Negative Tokens
            neg_m1 = df_m1[df_m1['shap_value'] < 0]['display_token'].tolist()
            neg_m2 = df_m2[df_m2['shap_value'] < 0]['display_token'].tolist()
            
            if neg_m1:
                print(f"  [Mode 1] Tokens with negative SHAP: {neg_m1}")
            if neg_m2:
                print(f"  [Mode 2] Tokens with negative SHAP: {neg_m2}")
                
            # Calculate Total Absolute Denominators for Relative Contribution
            total_abs_m1 = df_m1['shap_value'].abs().sum()
            total_abs_m2 = df_m2['shap_value'].abs().sum()

            # Helper to append to aggregation dict
            def add_to_agg(category, key, list_nm, mode1_val, mode2_val):
                if key not in aggregated_shaps[category]:
                    aggregated_shaps[category][key] = {}
                if list_nm not in aggregated_shaps[category][key]:
                    aggregated_shaps[category][key][list_nm] = {'mode_1': [], 'mode_2': []}
                
                # Mode 1 Object
                m1_contrib = mode1_val / total_abs_m1 if total_abs_m1 > 0 else 0
                point_m1 = {
                    'raw_shap': mode1_val,
                    'relative_contribution': m1_contrib,
                    'negative_tokens_in_prompt': neg_m1
                }
                
                # Mode 2 Object
                m2_contrib = mode2_val / total_abs_m2 if total_abs_m2 > 0 else 0
                point_m2 = {
                    'raw_shap': mode2_val,
                    'relative_contribution': m2_contrib,
                    'negative_tokens_in_prompt': neg_m2
                }
                
                aggregated_shaps[category][key][list_nm]['mode_1'].append(point_m1)
                aggregated_shaps[category][key][list_nm]['mode_2'].append(point_m2)

            def get_mode1_sum(target_string):
                # Mode 1 splits by whitespace. A target like "New York" becomes "New" and "York".
                # We need to find the SHAP values of these specific tokens.
                # Since prompt tokenization is deterministic, we can look for the token text.
                target_words = target_string.split()
                total_shap = 0.0
                
                # To be precise, we find the sequence of target_words in display_tokens, 
                # but since we just need the total contribution, summing the values of those
                # display tokens assuming they don't appear elsewhere in the prefix is a simple approximation.
                # A robust way is to just grab the row where display_token == word
                for word in target_words:
                    word_rows = df_m1[df_m1['display_token'] == word]
                    if not word_rows.empty:
                        # If a word like "New" appears multiple times, this might grab the first or sum.
                        # For simplicity, we grab the first token matched (often safe for cues in this dataset layout).
                        total_shap += float(word_rows['shap_value'].values[0])
                return total_shap

            # Extract Cues Importances
            if prompt_type == 'explicit':
                city_row = df_m2[df_m2['display_token'] == target_city]
                if target_city and not city_row.empty:
                    m1_val = get_mode1_sum(target_city)
                    add_to_agg('city', target_city, list_name, m1_val, float(city_row['shap_value'].values[0]))
                    
                country_row = df_m2[df_m2['display_token'] == target_country]
                if target_country and not country_row.empty:
                    m1_val = get_mode1_sum(target_country)
                    add_to_agg('country', target_country, list_name, m1_val, float(country_row['shap_value'].values[0]))
            else:
                pop_row = df_m2[df_m2['display_token'] == target_pop]
                if target_pop and not pop_row.empty:
                    m1_val = get_mode1_sum(target_pop)
                    add_to_agg('population', target_pop, list_name, m1_val, float(pop_row['shap_value'].values[0]))
                    
                income_row = df_m2[df_m2['display_token'] == target_income]
                if target_income and not income_row.empty:
                    m1_val = get_mode1_sum(target_income)
                    add_to_agg('income', target_income, list_name, m1_val, float(income_row['shap_value'].values[0]))
        
    print("\nAggregating results...")
    
    # Optionally, we can merge helpful/not helpful together if we want overall explicit/implicit.
    # But since the lists keep them separate under `city[New York][explicit_helpful]`, we don't 
    # strictly need to combine the top level like before.
    
    # Save Results
    output_filename = args.output_file
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(output_filename, 'w') as f:
        json.dump(aggregated_shaps, f, cls=NpEncoder, indent=4)
        
    print(f"Analysis complete. Distribution results saved to {output_filename}")
