from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from datasets import load_dataset
import torch
import pandas as pd
import logging
import json
import os

class Model:
    def __init__(self, model_id, max_new_tokens, max_input_token_length, task):
        self.model_id = model_id
        if 'gpt' in model_id.lower():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch_dtype, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, verbose=False)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.use_default_system_prompt = False
        self.max_new_tokens = max_new_tokens
        self.max_input_token_length = max_input_token_length
        self.task = task 

    def _generate_batch(self, conversations, user_contents):
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}"
                "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                "{% if system_message != '' %}<<SYS>>\n{{ system_message }}\n<</SYS>>\n\n{% endif %}"
                "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                "{% elif message['role'] == 'assistant' %}"
                "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                "{% endif %}"
                "{% endfor %}"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        input_ids = self.tokenizer.apply_chat_template(conversations, return_tensors="pt", add_generation_prompt=True, padding=True)
        if input_ids.shape[1] > self.max_input_token_length:
            input_ids = input_ids[:, -self.max_input_token_length:]
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids) 
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

        generate_kwargs = dict(
            {"input_ids": input_ids},
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,  
            eos_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.model.generate(**generate_kwargs)
        # Decode only the new tokens
        new_tokens = outputs[:, input_ids.shape[1]:]
        res_list = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        results = []
        for i, res in enumerate(res_list):
            if i == 0:
                logging.info(f"System Prompt: {conversations[i][0]['content'].strip()}")
                logging.info(f"Input Text: {user_contents[i]}")
                logging.info(f"Response: {res}")
                logging.info("")

            if res.lower().startswith('yes'):
                results.append(1)
            elif res.lower().startswith('no'):
                results.append(0)
            else:
                if 'yes' in res.lower():
                    results.append(1)
                elif 'no' in res.lower():
                    results.append(0)
                else:
                    if i == 0:
                        logging.info('Error: response of the model is neither yes nor no.')
                    results.append(-1)
        return results, user_contents

    def zeroshot_prompting(self, texts):
        conversations = []
        user_contents = []
        for text in texts:
            system_prompt = f"""You are given a text. Determine whether it contains elements of {self.task.replace('_', ' ')}. Just answer with 'Yes' or 'No', and don't add anything else.\n""" 
            user_content = f'"{text}"'
            user_contents.append(user_content)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])
        return self._generate_batch(conversations, user_contents)
    
    def zeroshot_explicit_prompting(self, texts, cities, countries):
        conversations = []
        user_contents = []
        city = cities[0]
        country = countries[0]

        for text in texts:
            system_prompt = f"""You are given a text. Determine whether it contains elements of {self.task.replace('_', ' ')}. Just answer with 'Yes' or 'No', and don't add anything else.\n"""  
            prefix = f"The person living in {city} in {country} said:"
            user_content = f'{prefix} "{text}"'
            user_contents.append(user_content)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])
        return self._generate_batch(conversations, user_contents)

    def zeroshot_implicit_prompting(self, texts, populations, household_incomes):
        conversations = []
        user_contents = []

        population = populations[0]
        household_income = household_incomes[0]
        income_parts = household_income.split()
        income_value = income_parts[0]
        currency = ' '.join(income_parts[1:]) if len(income_parts) > 1 else ''

        for text in texts:
            system_prompt = f"""You are given a text. Determine whether it contains elements of {self.task.replace('_', ' ')}. Just answer with 'Yes' or 'No', and don't add anything else.\n"""  
            prefix = f"The person living in a metropolitan statistical area that has {population} people and a median household income of {income_value} {currency} said:"
            user_content = f'{prefix} "{text}"'
            user_contents.append(user_content)
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])
        return self._generate_batch(conversations, user_contents)

def read_data(task, number_of_samples): 
    random_seed = 42
    if task == 'hatespeech': 
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        dataframe = pd.DataFrame(ds['train'])
        dataframe = dataframe[['text', 'hatespeech']]  
        dataframe = dataframe.rename(columns={'hatespeech': 'labels'})
        dataframe['labels'] = dataframe['labels'].apply(lambda x: 0 if x == 0 else 1)
        
        dataframe = dataframe.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
        positive_samples = dataframe[dataframe['labels'] == 1].sample(n=int(number_of_samples/2), random_state=random_seed)  
        negative_samples = dataframe[dataframe['labels'] == 0].sample(n=int(number_of_samples/2), random_state=random_seed)  
        dataframe = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
        
        logging.info(f"-----{task.capitalize().replace('_', ' ')} Dataset Information-----")  
        logging.info(f"Total size = {len(dataframe)}, {task.capitalize().replace('_', ' ')}: {len(dataframe[dataframe['labels'] == 1])}, Non-{task.replace('_', ' ')}: {len(dataframe[dataframe['labels'] == 0])}")  
        logging.info("")
        
        max_input_token_length = dataframe['text'].str.len().max()
        return dataframe, max_input_token_length
    elif task == 'sarcasm':
        target_path = os.path.join('..', 'isarcasm.csv')
        dataframe = pd.read_csv(target_path)

        dataframe = dataframe[['tweet', 'sarcastic']]
        dataframe = dataframe.rename(columns={'tweet': 'text', 'sarcastic': 'labels'})

        dataframe = dataframe.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
        positive_samples = dataframe[dataframe['labels'] == 1].sample(n=int(number_of_samples/2), random_state=random_seed)  
        negative_samples = dataframe[dataframe['labels'] == 0].sample(n=int(number_of_samples/2), random_state=random_seed)  
        dataframe = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
        
        logging.info(f"-----{task.capitalize().replace('_', ' ')} Dataset Information-----")  
        logging.info(f"Total size = {len(dataframe)}, {task.capitalize().replace('_', ' ')}: {len(dataframe[dataframe['labels'] == 1])}, Non-{task.replace('_', ' ')}: {len(dataframe[dataframe['labels'] == 0])}")  
        logging.info("")
        
        max_input_token_length = dataframe['text'].str.len().max()
        return dataframe, max_input_token_length
    else:
        raise NotImplementedError(f"Dataset loading for task '{task}' is not implemented yet. Placeholder: Use specific dataset and column for {task}.")  

def prediction_original(model, data, batch_size):
    valid_processed_samples = [] # List of (index, pred_original, target)
    count = 0
    undecided = 0

    # Iterate over data in chunks of batch_size
    for i in range(0, len(data), batch_size):
        chunk = data.iloc[i:i+batch_size]
        current_batch_size = len(chunk)
        logging.info(f"-----Running {model.model_id} zeroshot prompting (Batch {i}-{i+current_batch_size}/{len(data)})-----")
        
        texts = chunk['text'].tolist()
        ground_truths = chunk['labels'].tolist()
        indices = chunk.index.tolist()
        
        # Call batched inference
        batch_preds, _ = model.zeroshot_prompting(texts)
        
        for idx_in_batch, pred_original in enumerate(batch_preds):
            ground_truth = int(ground_truths[idx_in_batch])
            original_idx = indices[idx_in_batch]
            
            if pred_original == -1:
                undecided += 1
            else:
                valid_processed_samples.append((original_idx, pred_original, ground_truth))
            logging.info(f"Predicted original: {pred_original}, explicit: N/A, implicit: N/A, ground truth: {ground_truth}")
            count += 1
    
    logging.info(f"\n----------{model.model_id} original prompting result----------")
    logging.info(f"Out of {len(data)} samples, undecided samples: {undecided}, processed samples: {len(valid_processed_samples)}")

    return valid_processed_samples

def prediction_cues(model, data, cities, countries, populations, household_incomes, processed_original, batch_size):
    valid_processed_explicit_samples = [] # List of (index, pred_explicit, target)
    valid_processed_implicit_samples = [] # List of (index, pred_implicit, target)
    valid_explicit_prompts = [] # List of prompt_content aligned with valid_processed_explicit_samples
    valid_implicit_prompts = [] # List of prompt_content aligned with valid_processed_implicit_samples
    
    count = 0
    undecided_explicit = 0
    undecided_implicit = 0
    original_dict = {idx: pred for idx, pred, _ in processed_original}

    # Iterate over data in chunks of batch_size
    for i in range(0, len(data), batch_size):
        chunk = data.iloc[i:i+batch_size]
        current_batch_size = len(chunk)
        logging.info(f"-----Running {model.model_id} cues zeroshot prompting (Batch {i}-{i+current_batch_size}/{len(data)})-----")
        
        texts = chunk['text'].tolist()
        ground_truths = chunk['labels'].tolist()
        indices = chunk.index.tolist()
        
        # Call batched inference
        batch_preds_explicit, batch_prompts_explicit = model.zeroshot_explicit_prompting(texts, cities, countries)
        batch_preds_implicit, batch_prompts_implicit = model.zeroshot_implicit_prompting(texts, populations, household_incomes)
        
        for idx_in_batch, (pred_explicit, pred_implicit) in enumerate(zip(batch_preds_explicit, batch_preds_implicit)):
            ground_truth = int(ground_truths[idx_in_batch])
            original_idx = indices[idx_in_batch]
            pred_original = original_dict.get(original_idx, "N/A")
            
            if pred_explicit == -1:
                undecided_explicit += 1
            else:
                valid_processed_explicit_samples.append((original_idx, pred_explicit, ground_truth))
                valid_explicit_prompts.append(batch_prompts_explicit[idx_in_batch])
            
            if pred_implicit == -1:
                undecided_implicit += 1
            else:
                valid_processed_implicit_samples.append((original_idx, pred_implicit, ground_truth))
                valid_implicit_prompts.append(batch_prompts_implicit[idx_in_batch])
                
            logging.info(f"Predicted original: {pred_original}, explicit: {pred_explicit}, implicit: {pred_implicit}, ground truth: {ground_truth}")
            count += 1

    logging.info(f"\n----------{model.model_id} cues prompting result----------")
    logging.info(f"Explicit: Out of {len(data)} samples, undecided samples: {undecided_explicit}, processed samples: {len(valid_processed_explicit_samples)}")
    logging.info(f"Implicit: Out of {len(data)} samples, undecided samples: {undecided_implicit}, processed samples: {len(valid_processed_implicit_samples)}")

    return valid_processed_explicit_samples, valid_processed_implicit_samples, valid_explicit_prompts, valid_implicit_prompts

def compute_metrics(preds, targets):
    precision = precision_score(targets, preds, zero_division=0) * 100
    recall = recall_score(targets, preds, zero_division=0) * 100
    accuracy = accuracy_score(targets, preds) * 100
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0) * 100
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0) * 100
    conf_matrix = confusion_matrix(targets, preds)

    metrics = {
        "precision": precision,  
        "recall": recall,  
        "accuracy": accuracy,  
        "micro_f1": micro_f1,  
        "macro_f1": macro_f1,  
        "confusion_matrix": conf_matrix.tolist() 
    }

    return metrics

def evaluate(valid_samples_original, valid_samples_explicit, valid_samples_implicit, json_filename, task, city_flips=None, same_size_predictions=None, avg_explicit_metrics=None, avg_implicit_metrics=None, dataset=None, explicit_prompts_list=None, implicit_prompts_list=None, city_predictions=None):  
    processed_original = [pred for idx, pred, target in valid_samples_original]
    processed_targets = [target for idx, pred, target in valid_samples_original]

    # Filter explicit/implicit samples to intersect with valid original samples
    valid_original_indices = set(idx for idx, _, _ in valid_samples_original)
    
    valid_samples_explicit_intersect = [item for item in valid_samples_explicit if item[0] in valid_original_indices]
    processed_explicit = [pred for idx, pred, target in valid_samples_explicit_intersect]
    processed_targets_explicit = [target for idx, pred, target in valid_samples_explicit_intersect]
    
    valid_samples_implicit_intersect = [item for item in valid_samples_implicit if item[0] in valid_original_indices]
    processed_implicit = [pred for idx, pred, target in valid_samples_implicit_intersect]
    processed_targets_implicit = [target for idx, pred, target in valid_samples_implicit_intersect]

    metrics_original = compute_metrics(processed_original, processed_targets)
    metrics_explicit = compute_metrics(processed_explicit, processed_targets_explicit)
    metrics_implicit = compute_metrics(processed_implicit, processed_targets_implicit)

    # Per-city flips aggregation
    flips_explicit_total = 0
    flips_0to1_explicit_total = 0
    flips_1to0_explicit_total = 0
    flips_0to1_helpful_explicit_total = 0      # Ground truth: 1 / Original: 0 / Explicit: 1
    flips_0to1_nothelpful_explicit_total = 0   # Ground truth: 0 / Original: 0 / Explicit: 1
    flips_1to0_helpful_explicit_total = 0      # Ground truth: 0 / Original: 1 / Explicit: 0
    flips_1to0_nothelpful_explicit_total = 0   # Ground truth: 1 / Original: 1 / Explicit: 0
    num_valid_explicit_total = 0
    
    flips_implicit_total = 0
    flips_0to1_implicit_total = 0
    flips_1to0_implicit_total = 0
    flips_0to1_helpful_implicit_total = 0      # Ground truth: 1 / Original: 0 / Explicit: 1
    flips_0to1_nothelpful_implicit_total = 0   # Ground truth: 0 / Original: 0 / Explicit: 1
    flips_1to0_helpful_implicit_total = 0      # Ground truth: 0 / Original: 1 / Explicit: 0
    flips_1to0_nothelpful_implicit_total = 0   # Ground truth: 1 / Original: 1 / Explicit: 0
    num_valid_implicit_total = 0
    
    city_flips_results = {}

    if city_flips:
        for city, flips_data in city_flips.items():
            flips_explicit_total += flips_data['flips_explicit']
            flips_0to1_explicit_total += flips_data['flips_0to1_explicit']
            flips_1to0_explicit_total += flips_data['flips_1to0_explicit']
            flips_0to1_helpful_explicit_total += flips_data['flips_0to1_helpful_explicit']
            flips_0to1_nothelpful_explicit_total += flips_data['flips_0to1_nothelpful_explicit']
            flips_1to0_helpful_explicit_total += flips_data['flips_1to0_helpful_explicit']
            flips_1to0_nothelpful_explicit_total += flips_data['flips_1to0_nothelpful_explicit']
            num_valid_explicit_total += flips_data['num_valid_explicit']
            flips_implicit_total += flips_data['flips_implicit']
            flips_0to1_implicit_total += flips_data['flips_0to1_implicit']
            flips_1to0_implicit_total += flips_data['flips_1to0_implicit']
            flips_0to1_helpful_implicit_total += flips_data['flips_0to1_helpful_implicit']
            flips_0to1_nothelpful_implicit_total += flips_data['flips_0to1_nothelpful_implicit']
            flips_1to0_helpful_implicit_total += flips_data['flips_1to0_helpful_implicit']
            flips_1to0_nothelpful_implicit_total += flips_data['flips_1to0_nothelpful_implicit']
            num_valid_implicit_total += flips_data['num_valid_implicit']
            city_flips_results[city] = {
                "flips_explicit": flips_data['flips_explicit'],
                "flips_explicit_percent": f'{(flips_data["flips_explicit"] / flips_data["num_valid_explicit"] * 100):.2f}' if flips_data["num_valid_explicit"] > 0 else "0.00",
                "flips_0to1_explicit": flips_data['flips_0to1_explicit'],
                "flips_0to1_explicit_percent": f'{(flips_data["flips_0to1_explicit"] / flips_data["num_valid_explicit"] * 100):.2f}' if flips_data["num_valid_explicit"] > 0 else "0.00",
                "flips_1to0_explicit": flips_data['flips_1to0_explicit'],
                "flips_1to0_explicit_percent": f'{(flips_data["flips_1to0_explicit"] / flips_data["num_valid_explicit"] * 100):.2f}' if flips_data["num_valid_explicit"] > 0 else "0.00",
                "flips_0to1_helpful_explicit": flips_data['flips_0to1_helpful_explicit'],
                "flips_0to1_helpful_explicit_percent": f'{(flips_data["flips_0to1_helpful_explicit"] / flips_data["num_valid_explicit"] * 100):.2f}' if flips_data["num_valid_explicit"] > 0 else "0.00",
                "flips_0to1_nothelpful_explicit": flips_data['flips_0to1_nothelpful_explicit'],
                "flips_0to1_nothelpful_explicit_percent": f'{(flips_data["flips_0to1_nothelpful_explicit"] / flips_data["num_valid_explicit"] * 100):.2f}' if flips_data["num_valid_explicit"] > 0 else "0.00",
                "flips_1to0_helpful_explicit": flips_data['flips_1to0_helpful_explicit'],
                "flips_1to0_helpful_explicit_percent": f'{(flips_data["flips_1to0_helpful_explicit"] / flips_data["num_valid_explicit"] * 100):.2f}' if flips_data["num_valid_explicit"] > 0 else "0.00",
                "flips_1to0_nothelpful_explicit": flips_data['flips_1to0_nothelpful_explicit'],
                "flips_1to0_nothelpful_explicit_percent": f'{(flips_data["flips_1to0_nothelpful_explicit"] / flips_data["num_valid_explicit"] * 100):.2f}' if flips_data["num_valid_explicit"] > 0 else "0.00",
                "num_valid_samples_explicit": flips_data['num_valid_explicit'],
                "flips_implicit": flips_data['flips_implicit'],
                "flips_implicit_percent": f'{(flips_data["flips_implicit"] / flips_data["num_valid_implicit"] * 100):.2f}' if flips_data["num_valid_implicit"] > 0 else "0.00",
                "flips_0to1_implicit": flips_data['flips_0to1_implicit'],
                "flips_0to1_implicit_percent": f'{(flips_data["flips_0to1_implicit"] / flips_data["num_valid_implicit"] * 100):.2f}' if flips_data["num_valid_implicit"] > 0 else "0.00",
                "flips_1to0_implicit": flips_data['flips_1to0_implicit'],
                "flips_1to0_implicit_percent": f'{(flips_data["flips_1to0_implicit"] / flips_data["num_valid_implicit"] * 100):.2f}' if flips_data["num_valid_implicit"] > 0 else "0.00",
                "flips_0to1_helpful_implicit": flips_data['flips_0to1_helpful_implicit'],
                "flips_0to1_helpful_implicit_percent": f'{(flips_data["flips_0to1_helpful_implicit"] / flips_data["num_valid_implicit"] * 100):.2f}' if flips_data["num_valid_implicit"] > 0 else "0.00",
                "flips_0to1_nothelpful_implicit": flips_data['flips_0to1_nothelpful_implicit'],
                "flips_0to1_nothelpful_implicit_percent": f'{(flips_data["flips_0to1_nothelpful_implicit"] / flips_data["num_valid_implicit"] * 100):.2f}' if flips_data["num_valid_implicit"] > 0 else "0.00",
                "flips_1to0_helpful_implicit": flips_data['flips_1to0_helpful_implicit'],
                "flips_1to0_helpful_implicit_percent": f'{(flips_data["flips_1to0_helpful_implicit"] / flips_data["num_valid_implicit"] * 100):.2f}' if flips_data["num_valid_implicit"] > 0 else "0.00",
                "flips_1to0_nothelpful_implicit": flips_data['flips_1to0_nothelpful_implicit'],
                "flips_1to0_nothelpful_implicit_percent": f'{(flips_data["flips_1to0_nothelpful_implicit"] / flips_data["num_valid_implicit"] * 100):.2f}' if flips_data["num_valid_implicit"] > 0 else "0.00",
                "num_valid_samples_implicit": flips_data['num_valid_implicit']
            }

    flips_explicit_percent_total = (flips_explicit_total / num_valid_explicit_total * 100) if num_valid_explicit_total > 0 else 0
    flips_0to1_explicit_percent_total = (flips_0to1_explicit_total / num_valid_explicit_total * 100) if num_valid_explicit_total > 0 else 0
    flips_1to0_explicit_percent_total = (flips_1to0_explicit_total / num_valid_explicit_total * 100) if num_valid_explicit_total > 0 else 0
    flips_0to1_helpful_explicit_percent_total = (flips_0to1_helpful_explicit_total / num_valid_explicit_total * 100) if num_valid_explicit_total > 0 else 0
    flips_0to1_nothelpful_explicit_percent_total = (flips_0to1_nothelpful_explicit_total / num_valid_explicit_total * 100) if num_valid_explicit_total > 0 else 0
    flips_1to0_helpful_explicit_percent_total = (flips_1to0_helpful_explicit_total / num_valid_explicit_total * 100) if num_valid_explicit_total > 0 else 0
    flips_1to0_nothelpful_explicit_percent_total = (flips_1to0_nothelpful_explicit_total / num_valid_explicit_total * 100) if num_valid_explicit_total > 0 else 0
    flips_implicit_percent_total = (flips_implicit_total / num_valid_implicit_total * 100) if num_valid_implicit_total > 0 else 0
    flips_0to1_implicit_percent_total = (flips_0to1_implicit_total / num_valid_implicit_total * 100) if num_valid_implicit_total > 0 else 0
    flips_1to0_implicit_percent_total = (flips_1to0_implicit_total / num_valid_implicit_total * 100) if num_valid_implicit_total > 0 else 0
    flips_0to1_helpful_implicit_percent_total = (flips_0to1_helpful_implicit_total / num_valid_implicit_total * 100) if num_valid_implicit_total > 0 else 0
    flips_0to1_nothelpful_implicit_percent_total = (flips_0to1_nothelpful_implicit_total / num_valid_implicit_total * 100) if num_valid_implicit_total > 0 else 0
    flips_1to0_helpful_implicit_percent_total = (flips_1to0_helpful_implicit_total / num_valid_implicit_total * 100) if num_valid_implicit_total > 0 else 0
    flips_1to0_nothelpful_implicit_percent_total = (flips_1to0_nothelpful_implicit_total / num_valid_implicit_total * 100) if num_valid_implicit_total > 0 else 0

    # Same-size city flips
    same_size_flips = {}
    target_dict = {idx: target for idx, _, target in valid_samples_original}
    if same_size_predictions:
        for city_pair, (city1_explicit, city1_implicit, city2_explicit, city2_implicit) in same_size_predictions.items():
            # Align explicit predictions between cities
            city1_explicit_dict = {idx: pred for idx, pred, _ in city1_explicit}
            city2_explicit_dict = {idx: pred for idx, pred, _ in city2_explicit}
            common_indices_same_size_explicit = sorted(set(city1_explicit_dict.keys()) & set(city2_explicit_dict.keys()) & set(target_dict.keys()))
            explicit1 = [city1_explicit_dict[i] for i in common_indices_same_size_explicit]
            explicit2 = [city2_explicit_dict[i] for i in common_indices_same_size_explicit]
            targets_same_size_explicit = [target_dict[i] for i in common_indices_same_size_explicit]
            num_valid_same_size_explicit = len(common_indices_same_size_explicit)
            flips_explicit_same_size = sum(e1 != e2 for e1, e2 in zip(explicit1, explicit2))
            flips_0to1_explicit_same_size = sum(e1 == 0 and e2 == 1 for e1, e2 in zip(explicit1, explicit2))
            flips_1to0_explicit_same_size = sum(e1 == 1 and e2 == 0 for e1, e2 in zip(explicit1, explicit2))
            flips_0to1_helpful_explicit_same_size = sum(e1 == 0 and e2 == 1 and t == 1 for e1, e2, t in zip(explicit1, explicit2, targets_same_size_explicit))
            flips_0to1_nothelpful_explicit_same_size = sum(e1 == 0 and e2 == 1 and t == 0 for e1, e2, t in zip(explicit1, explicit2, targets_same_size_explicit))
            flips_1to0_helpful_explicit_same_size = sum(e1 == 1 and e2 == 0 and t == 0 for e1, e2, t in zip(explicit1, explicit2, targets_same_size_explicit))
            flips_1to0_nothelpful_explicit_same_size = sum(e1 == 1 and e2 == 0 and t == 1 for e1, e2, t in zip(explicit1, explicit2, targets_same_size_explicit))
            flips_explicit_same_size_percent = (flips_explicit_same_size / num_valid_same_size_explicit * 100) if num_valid_same_size_explicit > 0 else 0
            flips_0to1_explicit_same_size_percent = (flips_0to1_explicit_same_size / num_valid_same_size_explicit * 100) if num_valid_same_size_explicit > 0 else 0
            flips_1to0_explicit_same_size_percent = (flips_1to0_explicit_same_size / num_valid_same_size_explicit * 100) if num_valid_same_size_explicit > 0 else 0
            flips_0to1_helpful_explicit_same_size_percent = (flips_0to1_helpful_explicit_same_size / num_valid_same_size_explicit * 100) if num_valid_same_size_explicit > 0 else 0
            flips_0to1_nothelpful_explicit_same_size_percent = (flips_0to1_nothelpful_explicit_same_size / num_valid_same_size_explicit * 100) if num_valid_same_size_explicit > 0 else 0
            flips_1to0_helpful_explicit_same_size_percent = (flips_1to0_helpful_explicit_same_size / num_valid_same_size_explicit * 100) if num_valid_same_size_explicit > 0 else 0
            flips_1to0_nothelpful_explicit_same_size_percent = (flips_1to0_nothelpful_explicit_same_size / num_valid_same_size_explicit * 100) if num_valid_same_size_explicit > 0 else 0

            # Align implicit predictions between cities
            city1_implicit_dict = {idx: pred for idx, pred, _ in city1_implicit}
            city2_implicit_dict = {idx: pred for idx, pred, _ in city2_implicit}
            common_indices_same_size_implicit = sorted(set(city1_implicit_dict.keys()) & set(city2_implicit_dict.keys()) & set(target_dict.keys()))
            implicit1 = [city1_implicit_dict[i] for i in common_indices_same_size_implicit]
            implicit2 = [city2_implicit_dict[i] for i in common_indices_same_size_implicit]
            targets_same_size_implicit = [target_dict[i] for i in common_indices_same_size_implicit]
            num_valid_same_size_implicit = len(common_indices_same_size_implicit)
            flips_implicit_same_size = sum(i1 != i2 for i1, i2 in zip(implicit1, implicit2))
            flips_0to1_implicit_same_size = sum(i1 == 0 and i2 == 1 for i1, i2 in zip(implicit1, implicit2))
            flips_1to0_implicit_same_size = sum(i1 == 1 and i2 == 0 for i1, i2 in zip(implicit1, implicit2))
            flips_0to1_helpful_implicit_same_size = sum(i1 == 0 and i2 == 1 and t == 1 for i1, i2, t in zip(implicit1, implicit2, targets_same_size_implicit))
            flips_0to1_nothelpful_implicit_same_size = sum(i1 == 0 and i2 == 1 and t == 0 for i1, i2, t in zip(implicit1, implicit2, targets_same_size_implicit))
            flips_1to0_helpful_implicit_same_size = sum(i1 == 1 and i2 == 0 and t == 0 for i1, i2, t in zip(implicit1, implicit2, targets_same_size_implicit))
            flips_1to0_nothelpful_implicit_same_size = sum(i1 == 1 and i2 == 0 and t == 1 for i1, i2, t in zip(implicit1, implicit2, targets_same_size_implicit))
            flips_implicit_same_size_percent = (flips_implicit_same_size / num_valid_same_size_implicit * 100) if num_valid_same_size_implicit > 0 else 0
            flips_0to1_implicit_same_size_percent = (flips_0to1_implicit_same_size / num_valid_same_size_implicit * 100) if num_valid_same_size_implicit > 0 else 0
            flips_1to0_implicit_same_size_percent = (flips_1to0_implicit_same_size / num_valid_same_size_implicit * 100) if num_valid_same_size_implicit > 0 else 0
            flips_0to1_helpful_implicit_same_size_percent = (flips_0to1_helpful_implicit_same_size / num_valid_same_size_implicit * 100) if num_valid_same_size_implicit > 0 else 0
            flips_0to1_nothelpful_implicit_same_size_percent = (flips_0to1_nothelpful_implicit_same_size / num_valid_same_size_implicit * 100) if num_valid_same_size_implicit > 0 else 0
            flips_1to0_helpful_implicit_same_size_percent = (flips_1to0_helpful_implicit_same_size / num_valid_same_size_implicit * 100) if num_valid_same_size_implicit > 0 else 0
            flips_1to0_nothelpful_implicit_same_size_percent = (flips_1to0_nothelpful_implicit_same_size / num_valid_same_size_implicit * 100) if num_valid_same_size_implicit > 0 else 0

            same_size_flips[city_pair] = {
                "flips_explicit_same_size": flips_explicit_same_size,
                "flips_explicit_same_size_percent": f'{flips_explicit_same_size_percent:.2f}',
                "flips_0to1_explicit_same_size": flips_0to1_explicit_same_size,
                "flips_0to1_explicit_same_size_percent": f'{flips_0to1_explicit_same_size_percent:.2f}',
                "flips_1to0_explicit_same_size": flips_1to0_explicit_same_size,
                "flips_1to0_explicit_same_size_percent": f'{flips_1to0_explicit_same_size_percent:.2f}',
                "flips_0to1_helpful_explicit_same_size": flips_0to1_helpful_explicit_same_size,
                "flips_0to1_helpful_explicit_same_size_percent": f'{flips_0to1_helpful_explicit_same_size_percent:.2f}',
                "flips_0to1_nothelpful_explicit_same_size": flips_0to1_nothelpful_explicit_same_size,
                "flips_0to1_nothelpful_explicit_same_size_percent": f'{flips_0to1_nothelpful_explicit_same_size_percent:.2f}',
                "flips_1to0_helpful_explicit_same_size": flips_1to0_helpful_explicit_same_size,
                "flips_1to0_helpful_explicit_same_size_percent": f'{flips_1to0_helpful_explicit_same_size_percent:.2f}',
                "flips_1to0_nothelpful_explicit_same_size": flips_1to0_nothelpful_explicit_same_size,
                "flips_1to0_nothelpful_explicit_same_size_percent": f'{flips_1to0_nothelpful_explicit_same_size_percent:.2f}',
                "num_valid_samples_explicit": num_valid_same_size_explicit,
                "flips_implicit_same_size": flips_implicit_same_size,
                "flips_implicit_same_size_percent": f'{flips_implicit_same_size_percent:.2f}',
                "flips_0to1_implicit_same_size": flips_0to1_implicit_same_size,
                "flips_0to1_implicit_same_size_percent": f'{flips_0to1_implicit_same_size_percent:.2f}',
                "flips_1to0_implicit_same_size": flips_1to0_implicit_same_size,
                "flips_1to0_implicit_same_size_percent": f'{flips_1to0_implicit_same_size_percent:.2f}',
                "flips_0to1_helpful_implicit_same_size": flips_0to1_helpful_implicit_same_size,
                "flips_0to1_helpful_implicit_same_size_percent": f'{flips_0to1_helpful_implicit_same_size_percent:.2f}',
                "flips_0to1_nothelpful_implicit_same_size": flips_0to1_nothelpful_implicit_same_size,
                "flips_0to1_nothelpful_implicit_same_size_percent": f'{flips_0to1_nothelpful_implicit_same_size_percent:.2f}',
                "flips_1to0_helpful_implicit_same_size": flips_1to0_helpful_implicit_same_size,
                "flips_1to0_helpful_implicit_same_size_percent": f'{flips_1to0_helpful_implicit_same_size_percent:.2f}',
                "flips_1to0_nothelpful_implicit_same_size": flips_1to0_nothelpful_implicit_same_size,
                "flips_1to0_nothelpful_implicit_same_size_percent": f'{flips_1to0_nothelpful_implicit_same_size_percent:.2f}',
                "num_valid_samples_implicit": num_valid_same_size_implicit
            }

    task_label = task.replace('_', ' ').capitalize()  
    logging.info("\n----------Original Prompting Metrics----------")
    logging.info(f"Total valid samples = {len(processed_original)}")
    logging.info(f"Total {task_label.lower()} samples = {len([v for v in processed_targets if v == 1])}, non-{task_label.lower()} samples = {len([v for v in processed_targets if v == 0])}")  
    logging.info(f"Predicted {task_label.lower()} samples = {len([v for v in processed_original if v == 1])}, non-{task_label.lower()} samples = {len([v for v in processed_original if v == 0])}")  
    logging.info(f"- Precision = {metrics_original['precision']:.2f}")  
    logging.info(f"- Recall = {metrics_original['recall']:.2f}")  
    logging.info(f"- Accuracy = {metrics_original['accuracy']:.2f}") 
    logging.info(f"- Micro F1-Score = {metrics_original['micro_f1']:.2f}")  
    logging.info(f"- Macro F1-Score = {metrics_original['macro_f1']:.2f}")  
    logging.info(f"- Confusion Matrix = \n{metrics_original['confusion_matrix']}")

    logging.info("\n----------Explicit Cues Prompting Metrics----------")
    logging.info(f"Total valid samples = {len(processed_explicit)}")
    logging.info(f"Predicted {task_label.lower()} samples = {len([v for v in processed_explicit if v == 1])}, non-{task_label.lower()} samples = {len([v for v in processed_explicit if v == 0])}")  
    logging.info(f"- Precision = {metrics_explicit['precision']:.2f}")  
    logging.info(f"- Recall = {metrics_explicit['recall']:.2f}")  
    logging.info(f"- Accuracy = {metrics_explicit['accuracy']:.2f}")  
    logging.info(f"- Micro F1-Score = {metrics_explicit['micro_f1']:.2f}")  
    logging.info(f"- Macro F1-Score = {metrics_explicit['macro_f1']:.2f}")  
    logging.info(f"- Confusion Matrix = \n{metrics_explicit['confusion_matrix']}")

    logging.info("\n----------Implicit Cues Prompting Metrics----------")
    logging.info(f"Total valid samples = {len(processed_implicit)}")
    logging.info(f"Predicted {task_label.lower()} samples = {len([v for v in processed_implicit if v == 1])}, non-{task_label.lower()} samples = {len([v for v in processed_implicit if v == 0])}")  
    logging.info(f"- Precision = {metrics_implicit['precision']:.2f}")  
    logging.info(f"- Recall = {metrics_implicit['recall']:.2f}")  
    logging.info(f"- Accuracy = {metrics_implicit['accuracy']:.2f}")  
    logging.info(f"- Micro F1-Score = {metrics_implicit['micro_f1']:.2f}")  
    logging.info(f"- Macro F1-Score = {metrics_implicit['macro_f1']:.2f}")  
    logging.info(f"- Confusion Matrix = \n{metrics_implicit['confusion_matrix']}")

    logging.info("\n----------Flips----------")
    if city_flips_results:
        for city, flips_data in city_flips_results.items():
            logging.info(f"Flips (original vs explicit, {city}): {flips_data['flips_explicit']} ({flips_data['flips_explicit_percent']}%) over {flips_data['num_valid_samples_explicit']} valid samples")
            logging.info(f"  From non-{task_label.lower()} (0) to {task_label.lower()} (1): {flips_data['flips_0to1_explicit']} ({flips_data['flips_0to1_explicit_percent']}%)")  
            logging.info(f"    - Helpful (true {task_label.lower()}/1): {flips_data['flips_0to1_helpful_explicit']} ({flips_data['flips_0to1_helpful_explicit_percent']}%)")  
            logging.info(f"    - Not helpful (true non-{task_label.lower()}/0): {flips_data['flips_0to1_nothelpful_explicit']} ({flips_data['flips_0to1_nothelpful_explicit_percent']}%)")  
            logging.info(f"  From {task_label.lower()} (1) to non-{task_label.lower()} (0): {flips_data['flips_1to0_explicit']} ({flips_data['flips_1to0_explicit_percent']}%)")  
            logging.info(f"    - Helpful (true non-{task_label.lower()}/0): {flips_data['flips_1to0_helpful_explicit']} ({flips_data['flips_1to0_helpful_explicit_percent']}%)")  
            logging.info(f"    - Not helpful (true {task_label.lower()}/1): {flips_data['flips_1to0_nothelpful_explicit']} ({flips_data['flips_1to0_nothelpful_explicit_percent']}%)")  
            logging.info(f"Flips (original vs implicit, {city}): {flips_data['flips_implicit']} ({flips_data['flips_implicit_percent']}%) over {flips_data['num_valid_samples_implicit']} valid samples")
            logging.info(f"  From non-{task_label.lower()} (0) to {task_label.lower()} (1): {flips_data['flips_0to1_implicit']} ({flips_data['flips_0to1_implicit_percent']}%)")  
            logging.info(f"    - Helpful (true {task_label.lower()}/1): {flips_data['flips_0to1_helpful_implicit']} ({flips_data['flips_0to1_helpful_implicit_percent']}%)")  
            logging.info(f"    - Not helpful (true non-{task_label.lower()}/0): {flips_data['flips_0to1_nothelpful_implicit']} ({flips_data['flips_0to1_nothelpful_implicit_percent']}%)") 
            logging.info(f"  From {task_label.lower()} (1) to non-{task_label.lower()} (0): {flips_data['flips_1to0_implicit']} ({flips_data['flips_1to0_implicit_percent']}%)")  
            logging.info(f"    - Helpful (true non-{task_label.lower()}/0): {flips_data['flips_1to0_helpful_implicit']} ({flips_data['flips_1to0_helpful_implicit_percent']}%)")  
            logging.info(f"    - Not helpful (true {task_label.lower()}/1): {flips_data['flips_1to0_nothelpful_implicit']} ({flips_data['flips_1to0_nothelpful_implicit_percent']}%)")  
        logging.info(f"Total Flips (original vs explicit): {flips_explicit_total} ({flips_explicit_percent_total:.2f}%) over {num_valid_explicit_total} valid samples")
        logging.info(f"  From non-{task_label.lower()} (0) to {task_label.lower()} (1): {flips_0to1_explicit_total} ({flips_0to1_explicit_percent_total:.2f}%)")  
        logging.info(f"    - Helpful (true {task_label.lower()}/1): {flips_0to1_helpful_explicit_total} ({flips_0to1_helpful_explicit_percent_total:.2f}%)")  
        logging.info(f"    - Not helpful (true non-{task_label.lower()}/0): {flips_0to1_nothelpful_explicit_total} ({flips_0to1_nothelpful_explicit_percent_total:.2f}%)")  
        logging.info(f"  From {task_label.lower()} (1) to non-{task_label.lower()} (0): {flips_1to0_explicit_total} ({flips_1to0_explicit_percent_total:.2f}%)")  
        logging.info(f"    - Helpful (true non-{task_label.lower()}/0): {flips_1to0_helpful_explicit_total} ({flips_1to0_helpful_explicit_percent_total:.2f}%)")  
        logging.info(f"    - Not helpful (true {task_label.lower()}/1): {flips_1to0_nothelpful_explicit_total} ({flips_1to0_nothelpful_explicit_percent_total:.2f}%)")  
        logging.info(f"Total Flips (original vs implicit): {flips_implicit_total} ({flips_implicit_percent_total:.2f}%) over {num_valid_implicit_total} valid samples")
        logging.info(f"  From non-{task_label.lower()} (0) to {task_label.lower()} (1): {flips_0to1_implicit_total} ({flips_0to1_implicit_percent_total:.2f}%)")  
        logging.info(f"    - Helpful (true {task_label.lower()}/1): {flips_0to1_helpful_implicit_total} ({flips_0to1_helpful_implicit_percent_total:.2f}%)")  
        logging.info(f"    - Not helpful (true non-{task_label.lower()}/0): {flips_0to1_nothelpful_implicit_total} ({flips_0to1_nothelpful_implicit_percent_total:.2f}%)")  
        logging.info(f"  From {task_label.lower()} (1) to non-{task_label.lower()} (0): {flips_1to0_implicit_total} ({flips_1to0_implicit_percent_total:.2f}%)")  
        logging.info(f"    - Helpful (true non-{task_label.lower()}/0): {flips_1to0_helpful_implicit_total} ({flips_1to0_helpful_implicit_percent_total:.2f}%)")  
        logging.info(f"    - Not helpful (true {task_label.lower()}/1): {flips_1to0_nothelpful_implicit_total} ({flips_1to0_nothelpful_implicit_percent_total:.2f}%)")  

    if same_size_flips:
        logging.info("\n----------Same-Size City Flips----------")
        for city_pair, flips_data in same_size_flips.items():
            logging.info(f"Flips (explicit vs {city_pair}): {flips_data['flips_explicit_same_size']} ({flips_data['flips_explicit_same_size_percent']}%) over {flips_data['num_valid_samples_explicit']} valid samples")
            logging.info(f"  From non-{task_label.lower()} (0) to {task_label.lower()} (1): {flips_data['flips_0to1_explicit_same_size']} ({flips_data['flips_0to1_explicit_same_size_percent']}%)")  
            logging.info(f"    - Helpful (true {task_label.lower()}/1): {flips_data['flips_0to1_helpful_explicit_same_size']} ({flips_data['flips_0to1_helpful_explicit_same_size_percent']}%)") 
            logging.info(f"    - Not helpful (true non-{task_label.lower()}/0): {flips_data['flips_0to1_nothelpful_explicit_same_size']} ({flips_data['flips_0to1_nothelpful_explicit_same_size_percent']}%)")  
            logging.info(f"  From {task_label.lower()} (1) to non-{task_label.lower()} (0): {flips_data['flips_1to0_explicit_same_size']} ({flips_data['flips_1to0_explicit_same_size_percent']}%)")  
            logging.info(f"    - Helpful (true non-{task_label.lower()}/0): {flips_data['flips_1to0_helpful_explicit_same_size']} ({flips_data['flips_1to0_helpful_explicit_same_size_percent']}%)")  
            logging.info(f"    - Not helpful (true {task_label.lower()}/1): {flips_data['flips_1to0_nothelpful_explicit_same_size']} ({flips_data['flips_1to0_nothelpful_explicit_same_size_percent']}%)")  
            logging.info(f"Flips (implicit vs {city_pair}): {flips_data['flips_implicit_same_size']} ({flips_data['flips_implicit_same_size_percent']}%) over {flips_data['num_valid_samples_implicit']} valid samples")
            logging.info(f"  From non-{task_label.lower()} (0) to {task_label.lower()} (1): {flips_data['flips_0to1_implicit_same_size']} ({flips_data['flips_0to1_implicit_same_size_percent']}%)")  
            logging.info(f"    - Helpful (true {task_label.lower()}/1): {flips_data['flips_0to1_helpful_implicit_same_size']} ({flips_data['flips_0to1_helpful_implicit_same_size_percent']}%)")  
            logging.info(f"    - Not helpful (true non-{task_label.lower()}/0): {flips_data['flips_0to1_nothelpful_implicit_same_size']} ({flips_data['flips_0to1_nothelpful_implicit_same_size_percent']}%)")  
            logging.info(f"  From {task_label.lower()} (1) to non-{task_label.lower()} (0): {flips_data['flips_1to0_implicit_same_size']} ({flips_data['flips_1to0_implicit_same_size_percent']}%)")  
            logging.info(f"    - Helpful (true non-{task_label.lower()}/0): {flips_data['flips_1to0_helpful_implicit_same_size']} ({flips_data['flips_1to0_helpful_implicit_same_size_percent']}%)")  
            logging.info(f"    - Not helpful (true {task_label.lower()}/1): {flips_data['flips_1to0_nothelpful_implicit_same_size']} ({flips_data['flips_1to0_nothelpful_implicit_same_size_percent']}%)")  

    
    if avg_explicit_metrics:
        logging.info("\n----------Average Explicit Metrics for Size Category----------")
        logging.info(f"- Precision = {avg_explicit_metrics['precision']}")
        logging.info(f"- Recall = {avg_explicit_metrics['recall']}")
        logging.info(f"- Accuracy = {avg_explicit_metrics['accuracy']}")
        logging.info(f"- Micro F1-Score = {avg_explicit_metrics['micro_f1']}")
        logging.info(f"- Macro F1-Score = {avg_explicit_metrics['macro_f1']}")
        logging.info(f"- Confusion Matrix = \n{avg_explicit_metrics['confusion_matrix']}")

    if avg_implicit_metrics:
        logging.info("\n----------Average Implicit Metrics for Size Category----------")
        logging.info(f"- Precision = {avg_implicit_metrics['precision']}")
        logging.info(f"- Recall = {avg_implicit_metrics['recall']}")
        logging.info(f"- Accuracy = {avg_implicit_metrics['accuracy']}")
        logging.info(f"- Micro F1-Score = {avg_implicit_metrics['micro_f1']}")
        logging.info(f"- Macro F1-Score = {avg_implicit_metrics['macro_f1']}")
        logging.info(f"- Confusion Matrix = \n{avg_implicit_metrics['confusion_matrix']}")

    # Save predictions in the JSON file
    # Save predictions in the JSON file
    results = {
        "original": {k: str(v) if isinstance(v, (int, float)) else v for k, v in metrics_original.items()},  
        "avg_explicit_metrics": avg_explicit_metrics,  
        "avg_implicit_metrics": avg_implicit_metrics,  
        "city_flips": city_flips_results,
        # ... Flips stats kept as is for summary ...
        "flips_explicit_total": flips_explicit_total,
        "flips_explicit_percent_total": f'{flips_explicit_percent_total:.2f}',
        "flips_0to1_explicit_total": flips_0to1_explicit_total,
        "flips_0to1_explicit_percent_total": f'{flips_0to1_explicit_percent_total:.2f}',
        "flips_1to0_explicit_total": flips_1to0_explicit_total,
        "flips_1to0_explicit_percent_total": f'{flips_1to0_explicit_percent_total:.2f}',
        "flips_0to1_helpful_explicit_total": flips_0to1_helpful_explicit_total,
        "flips_0to1_helpful_explicit_percent_total": f'{flips_0to1_helpful_explicit_percent_total:.2f}',
        "flips_0to1_nothelpful_explicit_total": flips_0to1_nothelpful_explicit_total,
        "flips_0to1_nothelpful_explicit_percent_total": f'{flips_0to1_nothelpful_explicit_percent_total:.2f}',
        "flips_1to0_helpful_explicit_total": flips_1to0_helpful_explicit_total,
        "flips_1to0_helpful_explicit_percent_total": f'{flips_1to0_helpful_explicit_percent_total:.2f}',
        "flips_1to0_nothelpful_explicit_total": flips_1to0_nothelpful_explicit_total,
        "flips_1to0_nothelpful_explicit_percent_total": f'{flips_1to0_nothelpful_explicit_percent_total:.2f}',
        "num_valid_samples_explicit_total": num_valid_explicit_total,
        "flips_implicit_total": flips_implicit_total,
        "flips_implicit_percent_total": f'{flips_implicit_percent_total:.2f}',
        "flips_0to1_implicit_total": flips_0to1_implicit_total,
        "flips_0to1_implicit_percent_total": f'{flips_0to1_implicit_percent_total:.2f}',
        "flips_1to0_implicit_total": flips_1to0_implicit_total,
        "flips_1to0_implicit_percent_total": f'{flips_1to0_implicit_percent_total:.2f}',
        "flips_0to1_helpful_implicit_total": flips_0to1_helpful_implicit_total,
        "flips_0to1_helpful_implicit_percent_total": f'{flips_0to1_helpful_implicit_percent_total:.2f}',
        "flips_0to1_nothelpful_implicit_total": flips_0to1_nothelpful_implicit_total,
        "flips_0to1_nothelpful_implicit_percent_total": f'{flips_0to1_nothelpful_implicit_percent_total:.2f}',
        "flips_1to0_helpful_implicit_total": flips_1to0_helpful_implicit_total,
        "flips_1to0_helpful_implicit_percent_total": f'{flips_1to0_helpful_implicit_percent_total:.2f}',
        "flips_1to0_nothelpful_implicit_total": flips_1to0_nothelpful_implicit_total,
        "flips_1to0_nothelpful_implicit_percent_total": f'{flips_1to0_nothelpful_implicit_percent_total:.2f}',
        "num_valid_samples_implicit_total": num_valid_implicit_total,
        "same_size_flips": same_size_flips,
        "predictions": {
            "original": processed_original,
            "targets": processed_targets,
            "indices": [idx for idx, _, _ in valid_samples_original], # Explicitly saving indices for original
        }
    }
    
    # Add per-city predictions
    if city_predictions:
        for city, (valid_explicit, valid_implicit) in city_predictions.items():
             results["predictions"][city] = {
                 "explicit": [pred for idx, pred, _ in valid_explicit],
                 "indices_explicit": [idx for idx, pred, _ in valid_explicit],
                 "targets_explicit": [target for idx, pred, target in valid_explicit],
                 "implicit": [pred for idx, pred, _ in valid_implicit],
                 "indices_implicit": [idx for idx, pred, _ in valid_implicit],
                 "targets_implicit": [target for idx, pred, target in valid_implicit]
             }
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved evaluation metrics and predictions to {json_filename}")

    # FLIP STORAGE LOGIC
    if dataset is not None and explicit_prompts_list is not None and implicit_prompts_list is not None:
        flips_storage = {
            "explicit_helpful": [],
            "explicit_not_helpful": [],
            "implicit_helpful": [],
            "implicit_not_helpful": []
        }
        
        original_texts_map = dataset['text'].to_dict() # Assumes index alignment if no reset_index was weird

        original_preds = {idx: pred for idx, pred, _ in valid_samples_original}
        target_map = {idx: target for idx, _, target in valid_samples_original}
        
        # Explicit Flips
        for (idx, pred, _), prompt_content in zip(valid_samples_explicit, explicit_prompts_list):
            if idx in original_preds and idx in target_map:
                orig_pred = original_preds[idx]
                target = target_map[idx]
                # Check for FLIP
                if orig_pred != pred:
                    entry = {
                        "index": idx,
                        "original_text": original_texts_map.get(idx, ""),
                        "prompt_content": prompt_content,
                        "original_pred": orig_pred,
                        "new_pred": pred,
                        "target": target
                    }
                    
                    # Helpful: (0->1 for target=1) OR (1->0 for target=0)
                    # i.e. New prediction matches target
                    if pred == target:
                        flips_storage["explicit_helpful"].append(entry)
                    else:
                        flips_storage["explicit_not_helpful"].append(entry)

        # Implicit Flips
        for (idx, pred, _), prompt_content in zip(valid_samples_implicit, implicit_prompts_list):
            if idx in original_preds and idx in target_map:
                orig_pred = original_preds[idx]
                target = target_map[idx]
                # Check for FLIP
                if orig_pred != pred:
                    entry = {
                        "index": idx,
                        "original_text": original_texts_map.get(idx, ""),
                        "prompt_content": prompt_content,
                        "original_pred": orig_pred,
                        "new_pred": pred,
                        "target": target
                    }
                    
                    if pred == target:
                        flips_storage["implicit_helpful"].append(entry)
                    else:
                        flips_storage["implicit_not_helpful"].append(entry)
                        
        return flips_storage
    return None