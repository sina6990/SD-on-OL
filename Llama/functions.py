# zeroshot_functions.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from threading import Thread
from datasets import load_dataset
import torch
import pandas as pd
import logging
import csv
import json

class LlamaModel:
    def __init__(self, model_id, temperature, top_p, top_k, repetition_penalty, max_new_tokens, max_input_token_length):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, verbose=False)
        self.tokenizer.use_default_system_prompt = False
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.max_input_token_length = max_input_token_length

    def zeroshot_prompting(self, text):
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

        conversation = []
        system_prompt = """I will provide you with a text. Please determine if it contains elements of hate speech. Just answer with 'Yes' or 'No', and don't add anything else.\n"""
        user_content = " " + text
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_content})

        input_ids = self.tokenizer.apply_chat_template(conversation, 
                                                       return_tensors="pt",
                                                       add_generation_prompt=True,
                                                       padding=True)  
        if input_ids.shape[1] > self.max_input_token_length:
            input_ids = input_ids[:, -self.max_input_token_length:]
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids) 

        streamer = TextIteratorStreamer(self.tokenizer, timeout=100.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,  
            eos_token_id=self.tokenizer.eos_token_id,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
        res = ''.join(outputs)

        logging.info(f"System Prompt: {system_prompt.strip()}")
        logging.info(f"Input Text: {user_content}")
        logging.info(f"Response: {res}")
        logging.info("")

        if res.lower().startswith('yes'):
            return 1
        elif res.lower().startswith('no'):
            return 0
        else:
            if 'yes' in res.lower():
                return 1
            elif 'no' in res.lower():
                return 0
            else:
                logging.info('Error: response of Llama is neither yes nor no.')
                return -1
            
    def zeroshot_explicit_prompting(self, text, cities, index):
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

        conversation = []
        system_prompt = """I will provide you with a text. Please determine if it contains elements of hate speech. Just answer with 'Yes' or 'No', and don't add anything else.\n"""
        city = cities[index % len(cities)] 
        prefix = f"The person living in {city} said: "
        user_content = prefix + text
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_content})

        input_ids = self.tokenizer.apply_chat_template(conversation, 
                                                       return_tensors="pt",
                                                       add_generation_prompt=True,
                                                       padding=True)  
        if input_ids.shape[1] > self.max_input_token_length:
            input_ids = input_ids[:, -self.max_input_token_length:]
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids) 

        streamer = TextIteratorStreamer(self.tokenizer, timeout=100.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,  
            eos_token_id=self.tokenizer.eos_token_id,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
        res = ''.join(outputs)

        logging.info(f"System Prompt: {system_prompt.strip()}")
        logging.info(f"Input Text: {user_content}")
        logging.info(f"Response: {res}")
        logging.info("")

        if res.lower().startswith('yes'):
            return 1
        elif res.lower().startswith('no'):
            return 0
        else:
            if 'yes' in res.lower():
                return 1
            elif 'no' in res.lower():
                return 0
            else:
                logging.info('Error: response of Llama is neither yes nor no.')
                return -1

    def zeroshot_implicit_prompting(self, text, populations, household_incomes, index):
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

        conversation = []
        system_prompt = """I will provide you with a text. Please determine if it contains elements of hate speech. Just answer with 'Yes' or 'No', and don't add anything else.\n"""
        population = populations[index % len(populations)]
        household_income = household_incomes[index % len(household_incomes)]
        income_parts = household_income.split()
        income_value = income_parts[0]
        currency = ' '.join(income_parts[1:]) if len(income_parts) > 1 else ''
        prefix = f"The person living in the metropolitan statistical area that has {population} people and a median household income of {income_value} {currency} said: "
        user_content = prefix + text
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_content})

        input_ids = self.tokenizer.apply_chat_template(conversation, 
                                                       return_tensors="pt",
                                                       add_generation_prompt=True,
                                                       padding=True)  
        if input_ids.shape[1] > self.max_input_token_length:
            input_ids = input_ids[:, -self.max_input_token_length:]
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids) 

        streamer = TextIteratorStreamer(self.tokenizer, timeout=100.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,  
            eos_token_id=self.tokenizer.eos_token_id,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
        res = ''.join(outputs)

        logging.info(f"System Prompt: {system_prompt.strip()}")
        logging.info(f"Input Text: {user_content}")
        logging.info(f"Response: {res}")
        logging.info("")

        if res.lower().startswith('yes'):
            return 1
        elif res.lower().startswith('no'):
            return 0
        else:
            if 'yes' in res.lower():
                return 1
            elif 'no' in res.lower():
                return 0
            else:
                logging.info('Error: response of Llama is neither yes nor no.')
                return -1

def read_data():
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    dataframe = pd.DataFrame(ds['train'])
    dataframe = dataframe[['text', 'hatespeech']]
    dataframe = dataframe.rename(columns={'hatespeech': 'labels'})
    dataframe['labels'] = dataframe['labels'].apply(lambda x: 0 if x == 0 else 1)
    
    random_seed = 42
    dataframe = dataframe.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    hate_samples = dataframe[dataframe['labels'] == 1].sample(n=100, random_state=random_seed)
    non_hate_samples = dataframe[dataframe['labels'] == 0].sample(n=100, random_state=random_seed)
    dataframe = pd.concat([hate_samples, non_hate_samples]).reset_index(drop=True)
    
    logging.info(f"-----HateSpeech Dataset Information-----")
    logging.info(f"Total size = {len(dataframe)}, Hate: {len(dataframe[dataframe['labels'] == 1])}, Non-hate: {len(dataframe[dataframe['labels'] == 0])}")
    logging.info("")
    
    max_input_token_length = dataframe['text'].str.len().max()
    return dataframe, max_input_token_length

def prediction_original(model, data):
    targets = [int(v) for v in data['labels'].values]
    preds_original = []
    count = 0

    for idx, row in data.iterrows():
        count += 1
        logging.info(f"-----Running {model.model_id} zeroshot prompting ({count}/{len(data)})-----")
        text = row['text']
        ground_truth = int(row['labels'])
        pred_original = model.zeroshot_prompting(text)
        preds_original.append(pred_original)
        logging.info(f"Predicted original: {pred_original}, explicit: N/A, implicit: N/A, ground truth: {ground_truth}")

    processed_original = []
    processed_targets = []
    corrupted = 0

    for o, t in zip(preds_original, targets):
        if o == -1:
            corrupted += 1
        else:
            processed_original.append(o)
            processed_targets.append(t)

    logging.info(f"\n----------{model.model_id} original prompting result----------")
    logging.info(f"Out of {len(preds_original)} samples, corrupted samples: {corrupted}, processed samples: {len(processed_original)}")

    return processed_original, processed_targets

def prediction_cues(model, data, cities, populations, household_incomes, processed_original, processed_targets):
    targets = [int(v) for v in data['labels'].values]
    preds_explicit = []
    preds_implicit = []
    count = 0

    for idx, row in data.iterrows():
        count += 1
        logging.info(f"-----Running {model.model_id} zeroshot prompting ({count}/{len(data)})-----")
        text = row['text']
        ground_truth = int(row['labels'])
        pred_explicit = model.zeroshot_explicit_prompting(text, cities, idx)
        pred_implicit = model.zeroshot_implicit_prompting(text, populations, household_incomes, idx)
        preds_explicit.append(pred_explicit)
        preds_implicit.append(pred_implicit)
        # Use processed_original[count-1] for logging if available, else N/A
        pred_original = processed_original[count-1] if count-1 < len(processed_original) else "N/A"
        logging.info(f"Predicted original: {pred_original}, explicit: {pred_explicit}, implicit: {pred_implicit}, ground truth: {ground_truth}")

    processed_explicit = []
    processed_implicit = []
    processed_targets_cues = []
    corrupted = 0

    for e, i, t in zip(preds_explicit, preds_implicit, targets):
        if -1 in (e, i):
            corrupted += 1
        else:
            processed_explicit.append(e)
            processed_implicit.append(i)
            processed_targets_cues.append(t)

    logging.info(f"\n----------{model.model_id} cues prompting result----------")
    logging.info(f"Out of {len(preds_explicit)} samples, corrupted samples: {corrupted}, processed samples: {len(processed_explicit)}")

    return processed_explicit, processed_implicit, processed_targets_cues

def compute_metrics(preds, targets):
    precision = precision_score(targets, preds, zero_division=0) * 100
    recall = recall_score(targets, preds, zero_division=0) * 100
    accuracy = accuracy_score(targets, preds) * 100
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0) * 100
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0) * 100
    conf_matrix = confusion_matrix(targets, preds)

    metrics = {
        "precision": f'{precision:.2f}',
        "recall": f'{recall:.2f}',
        "accuracy": f'{accuracy:.2f}',
        "micro_f1": f'{micro_f1:.2f}',
        "macro_f1": f'{macro_f1:.2f}',
        "confusion_matrix": conf_matrix.tolist() 
    }
    return metrics

def evaluate(processed_original, processed_explicit, processed_implicit, processed_targets, json_filename):
    metrics_original = compute_metrics(processed_original, processed_targets)
    metrics_explicit = compute_metrics(processed_explicit, processed_targets)
    metrics_implicit = compute_metrics(processed_implicit, processed_targets)

    flips_explicit = sum(o != e for o, e in zip(processed_original, processed_explicit))
    flips_implicit = sum(o != i for o, i in zip(processed_original, processed_implicit))
    num_samples = len(processed_original)
    flips_explicit_percent = (flips_explicit / num_samples * 100) if num_samples > 0 else 0
    flips_implicit_percent = (flips_implicit / num_samples * 100) if num_samples > 0 else 0

    logging.info("\n----------Original Prompting Metrics----------")
    logging.info(f"Total manipulative samples = {len([v for v in processed_targets if v == 1])}, non-manipulative samples = {len([v for v in processed_targets if v == 0])}")
    logging.info(f"Predicted manipulative samples = {len([v for v in processed_original if v == 1])}, non-manipulative samples = {len([v for v in processed_original if v == 0])}")
    logging.info(f"- Precision = {metrics_original['precision']}")
    logging.info(f"- Recall = {metrics_original['recall']}")
    logging.info(f"- Accuracy = {metrics_original['accuracy']}")
    logging.info(f"- Micro F1-Score = {metrics_original['micro_f1']}")
    logging.info(f"- Macro F1-Score = {metrics_original['macro_f1']}")
    logging.info(f"- Confusion Matrix = \n{confusion_matrix(processed_targets, processed_original)}")

    logging.info("\n----------Explicit Cues Prompting Metrics----------")
    logging.info(f"Predicted manipulative samples = {len([v for v in processed_explicit if v == 1])}, non-manipulative samples = {len([v for v in processed_explicit if v == 0])}")
    logging.info(f"- Precision = {metrics_explicit['precision']}")
    logging.info(f"- Recall = {metrics_explicit['recall']}")
    logging.info(f"- Accuracy = {metrics_explicit['accuracy']}")
    logging.info(f"- Micro F1-Score = {metrics_explicit['micro_f1']}")
    logging.info(f"- Macro F1-Score = {metrics_explicit['macro_f1']}")
    logging.info(f"- Confusion Matrix = \n{confusion_matrix(processed_targets, processed_explicit)}")

    logging.info("\n----------Implicit Cues Prompting Metrics----------")
    logging.info(f"Predicted manipulative samples = {len([v for v in processed_implicit if v == 1])}, non-manipulative samples = {len([v for v in processed_implicit if v == 0])}")
    logging.info(f"- Precision = {metrics_implicit['precision']}")
    logging.info(f"- Recall = {metrics_implicit['recall']}")
    logging.info(f"- Accuracy = {metrics_implicit['accuracy']}")
    logging.info(f"- Micro F1-Score = {metrics_implicit['micro_f1']}")
    logging.info(f"- Macro F1-Score = {metrics_implicit['macro_f1']}")
    logging.info(f"- Confusion Matrix = \n{confusion_matrix(processed_targets, processed_implicit)}")

    logging.info("\n----------Flips----------")
    logging.info(f"Flips (original vs explicit): {flips_explicit} ({flips_explicit_percent:.2f}%)")
    logging.info(f"Flips (original vs implicit): {flips_implicit} ({flips_implicit_percent:.2f}%)")

    results = {
        "original": metrics_original,
        "explicit": metrics_explicit,
        "implicit": metrics_implicit,
        "flips_explicit": flips_explicit,
        "flips_explicit_percent": f'{flips_explicit_percent:.2f}',
        "flips_implicit": flips_implicit,
        "flips_implicit_percent": f'{flips_implicit_percent:.2f}',
        "predictions": {
            "original": processed_original,
            "explicit": processed_explicit,
            "implicit": processed_implicit,
            "targets": processed_targets
        }
    }
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved evaluation metrics to {json_filename}")

