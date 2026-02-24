from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from datasets import load_dataset
import torch
import pandas as pd
import logging
import json
import os
import random
import numpy as np

class Model:
    def __init__(self, model_id, max_new_tokens, max_input_token_length, task):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")
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
        return results

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
    
    def zeroshot_random_prompting(self, texts, num_tokens):
        conversations = []
        user_contents = []

        vocab_size = self.tokenizer.vocab_size
        
        for text in texts:
            # Generate random token IDs
            random_ids = [random.randint(0, vocab_size - 1) for _ in range(num_tokens)]
            random_prefix = self.tokenizer.decode(random_ids, skip_special_tokens=True)
            
            system_prompt = f"""You are given a text. Determine whether it contains elements of {self.task.replace('_', ' ')}. Just answer with 'Yes' or 'No', and don't add anything else.\n"""  
            
            user_content = f'{random_prefix} "{text}"'
            
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
        batch_preds = model.zeroshot_prompting(texts)
        
        for idx_in_batch, pred_original in enumerate(batch_preds):
            ground_truth = int(ground_truths[idx_in_batch])
            original_idx = indices[idx_in_batch]
            
            if pred_original == -1:
                undecided += 1
            else:
                valid_processed_samples.append((original_idx, pred_original, ground_truth))
            logging.info(f"Predicted original: {pred_original}, random: N/A, ground truth: {ground_truth}")
            count += 1
    
    logging.info(f"\n----------{model.model_id} original prompting result----------")
    logging.info(f"Out of {len(data)} samples, undecided samples: {undecided}, processed samples: {len(valid_processed_samples)}")

    return valid_processed_samples

def prediction_random(model, data, processed_original, batch_size, num_tokens):
    valid_processed_random_samples = [] # List of (index, pred_random, target)
    count = 0
    undecided_random = 0
    original_dict = {idx: pred for idx, pred, _ in processed_original}

    # Iterate over data in chunks of batch_size
    for i in range(0, len(data), batch_size):
        chunk = data.iloc[i:i+batch_size]
        current_batch_size = len(chunk)
        logging.info(f"-----Running {model.model_id} random prompting (Batch {i}-{i+current_batch_size}/{len(data)})-----")
        
        texts = chunk['text'].tolist()
        ground_truths = chunk['labels'].tolist()
        indices = chunk.index.tolist()
        
        # Call batched inference with random tokens
        batch_preds_random = model.zeroshot_random_prompting(texts, num_tokens)
        
        for idx_in_batch, pred_random in enumerate(batch_preds_random):
            ground_truth = int(ground_truths[idx_in_batch])
            original_idx = indices[idx_in_batch]
            pred_original = original_dict.get(original_idx, "N/A")
            
            if pred_random == -1:
                undecided_random += 1
            else:
                valid_processed_random_samples.append((original_idx, pred_random, ground_truth))
            
            logging.info(f"Predicted original: {pred_original}, random: {pred_random}, ground truth: {ground_truth}")
            count += 1

    logging.info(f"\n----------{model.model_id} random prompting result----------")
    logging.info(f"Random: Out of {len(data)} samples, undecided samples: {undecided_random}, processed samples: {len(valid_processed_random_samples)}")

    return valid_processed_random_samples

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

def evaluate(valid_samples_original, valid_samples_random, json_filename, task):  
    processed_original = [pred for idx, pred, target in valid_samples_original]
    processed_targets = [target for idx, pred, target in valid_samples_original]

    # Filter random samples to intersect with valid original samples
    valid_original_indices = set(idx for idx, _, _ in valid_samples_original)
    
    valid_samples_random_intersect = [item for item in valid_samples_random if item[0] in valid_original_indices]
    processed_random = [pred for idx, pred, target in valid_samples_random_intersect]
    processed_targets_random = [target for idx, pred, target in valid_samples_random_intersect]
    
    random_dict = {idx: pred for idx, pred, _ in valid_samples_random_intersect}
    original_dict = {idx: pred for idx, pred, _ in valid_samples_original}
    
    common_indices = sorted(set(original_dict.keys()) & set(random_dict.keys()))
    
    aligned_original = [original_dict[i] for i in common_indices]
    aligned_random = [random_dict[i] for i in common_indices]
    # Targets should be same from any source
    target_dict = {idx: target for idx, _, target in valid_samples_original}
    aligned_targets = [target_dict[i] for i in common_indices]
    
    metrics_original = compute_metrics(aligned_original, aligned_targets)
    metrics_random = compute_metrics(aligned_random, aligned_targets)

    # Flips Calculation
    flips_random = sum(o != r for o, r in zip(aligned_original, aligned_random))
    flips_0to1_random = sum(o == 0 and r == 1 for o, r in zip(aligned_original, aligned_random))
    flips_1to0_random = sum(o == 1 and r == 0 for o, r in zip(aligned_original, aligned_random))
    
    flips_0to1_helpful_random = sum(o == 0 and r == 1 and t == 1 for o, r, t in zip(aligned_original, aligned_random, aligned_targets))
    flips_0to1_nothelpful_random = sum(o == 0 and r == 1 and t == 0 for o, r, t in zip(aligned_original, aligned_random, aligned_targets))
    flips_1to0_helpful_random = sum(o == 1 and r == 0 and t == 0 for o, r, t in zip(aligned_original, aligned_random, aligned_targets))
    flips_1to0_nothelpful_random = sum(o == 1 and r == 0 and t == 1 for o, r, t in zip(aligned_original, aligned_random, aligned_targets))

    num_valid = len(common_indices)
    
    def calc_percent(num):
        return (num / num_valid * 100) if num_valid > 0 else 0

    task_label = task.replace('_', ' ').capitalize()  
    logging.info("\n----------Original Prompting Metrics (Aligned)----------")
    logging.info(f"Total valid intersected samples = {num_valid}")
    logging.info(f"- Precision = {metrics_original['precision']:.2f}")  
    logging.info(f"- Recall = {metrics_original['recall']:.2f}")  
    logging.info(f"- Accuracy = {metrics_original['accuracy']:.2f}") 
    logging.info(f"- Micro F1-Score = {metrics_original['micro_f1']:.2f}")  
    logging.info(f"- Macro F1-Score = {metrics_original['macro_f1']:.2f}")  

    logging.info("\n----------Random Prompting Metrics (Aligned)----------")
    logging.info(f"- Precision = {metrics_random['precision']:.2f}")  
    logging.info(f"- Recall = {metrics_random['recall']:.2f}")  
    logging.info(f"- Accuracy = {metrics_random['accuracy']:.2f}")  
    logging.info(f"- Micro F1-Score = {metrics_random['micro_f1']:.2f}")  
    logging.info(f"- Macro F1-Score = {metrics_random['macro_f1']:.2f}")  

    logging.info("\n----------Flips (Original vs Random)----------")
    logging.info(f"Total Flips: {flips_random} ({calc_percent(flips_random):.2f}%)")
    logging.info(f"  From non-{task_label.lower()} (0) to {task_label.lower()} (1): {flips_0to1_random} ({calc_percent(flips_0to1_random):.2f}%)")  
    logging.info(f"    - Helpful (true 1): {flips_0to1_helpful_random} ({calc_percent(flips_0to1_helpful_random):.2f}%)")  
    logging.info(f"    - Not helpful (true 0): {flips_0to1_nothelpful_random} ({calc_percent(flips_0to1_nothelpful_random):.2f}%)")  
    logging.info(f"  From {task_label.lower()} (1) to non-{task_label.lower()} (0): {flips_1to0_random} ({calc_percent(flips_1to0_random):.2f}%)")  
    logging.info(f"    - Helpful (true 0): {flips_1to0_helpful_random} ({calc_percent(flips_1to0_helpful_random):.2f}%)")  
    logging.info(f"    - Not helpful (true 1): {flips_1to0_nothelpful_random} ({calc_percent(flips_1to0_nothelpful_random):.2f}%)")  

    # Save predictions in the JSON file
    results = {
        "original_metrics": {k: str(v) if isinstance(v, (int, float)) else v for k, v in metrics_original.items()},  
        "random_metrics": {k: str(v) if isinstance(v, (int, float)) else v for k, v in metrics_random.items()},  
        "flips_stats": {
            "total": flips_random,
            "percent": f"{calc_percent(flips_random):.2f}",
            "0to1": flips_0to1_random,
            "0to1_percent": f"{calc_percent(flips_0to1_random):.2f}",
            "1to0": flips_1to0_random,
            "1to0_percent": f"{calc_percent(flips_1to0_random):.2f}",
            "0to1_helpful": flips_0to1_helpful_random,
            "0to1_nothelpful": flips_0to1_nothelpful_random,
            "1to0_helpful": flips_1to0_helpful_random,
            "1to0_nothelpful": flips_1to0_nothelpful_random
        },
        "predictions": {
            "indices": common_indices,
            "original": aligned_original,
            "random": aligned_random,
            "targets": aligned_targets
        }
    }
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved evaluation metrics and predictions to {json_filename}")