from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from threading import Thread
from datasets import load_dataset
import torch
import pandas as pd
import logging
import json

class Model:
    #def __init__(self, model_id, temperature, top_p, top_k, repetition_penalty, max_new_tokens, max_input_token_length):
    def __init__(self, model_id, max_new_tokens, max_input_token_length, task):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, verbose=False)
        self.tokenizer.use_default_system_prompt = False
        #self.temperature = temperature
        #self.top_p = top_p
        #self.top_k = top_k
        #self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.max_input_token_length = max_input_token_length
        self.task = task 

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
        system_prompt = f"""You are given a text. Determine whether it contains elements of {self.task.replace('_', ' ')}. Just answer with 'Yes' or 'No', and don't add anything else.\n""" 
        user_content = " " + text
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_content})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True, padding=True)
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
            # top_p=self.top_p,
            # top_k=self.top_k,
            # temperature=self.temperature,
            # repetition_penalty=self.repetition_penalty,
            num_beams=1,
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
                logging.info('Error: response of the model is neither yes nor no.')
                return -1
    
    def zeroshot_explicit_prompting(self, text, cities, countries):
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
        system_prompt = f"""You are given a text. Determine whether it contains elements of {self.task.replace('_', ' ')}. Just answer with 'Yes' or 'No', and don't add anything else.\n"""  
        city = cities[0]
        country = countries[0]
        prefix = f"The person living in {city} in {country} said: "
        user_content = prefix + text
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_content})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True, padding=True)  
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
            # top_p=self.top_p,
            # top_k=self.top_k,
            # temperature=self.temperature,
            # repetition_penalty=self.repetition_penalty,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,  
            eos_token_id=self.tokenizer.eos_token_id,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text_stream in streamer:
            outputs.append(text_stream)
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
                logging.info('Error: response of the model is neither yes nor no.')
                return -1
            
    def zeroshot_implicit_prompting(self, text, populations, household_incomes):
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
        system_prompt = f"""You are given a text. Determine whether it contains elements of {self.task.replace('_', ' ')}. Just answer with 'Yes' or 'No', and don't add anything else.\n"""  # Changed: Dynamically insert task into prompt
        population = populations[0]
        household_income = household_incomes[0]
        income_parts = household_income.split()
        income_value = income_parts[0]
        currency = ' '.join(income_parts[1:]) if len(income_parts) > 1 else ''
        prefix = f"The person living in the metropolitan statistical area that has {population} people and a median household income of {income_value} {currency} said: "
        user_content = prefix + text
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": user_content})

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True, padding=True)  
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
            # top_p=self.top_p,
            # top_k=self.top_k,
            # temperature=self.temperature,
            # repetition_penalty=self.repetition_penalty,
            num_beams=1,
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
                logging.info('Error: response of the model is neither yes nor no.')
                return -1

def read_data(task, number_of_samples): 
    if task == 'hatespeech': 
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        dataframe = pd.DataFrame(ds['train'])
        dataframe = dataframe[['text', 'hatespeech']]  
        dataframe = dataframe.rename(columns={'hatespeech': 'labels'})
        dataframe['labels'] = dataframe['labels'].apply(lambda x: 0 if x == 0 else 1)
        
        random_seed = 42
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

def prediction_original(model, data):
    valid_processed_samples = [] # List of (index, pred_original, target)
    count = 0
    undecided = 0

    for idx, row in data.iterrows():
        count += 1
        logging.info(f"-----Running {model.model_id} zeroshot prompting ({count}/{len(data)})-----")
        text = row['text']
        ground_truth = int(row['labels'])
        pred_original = model.zeroshot_prompting(text)
        if pred_original == -1:
            undecided += 1
        else:
            valid_processed_samples.append((idx, pred_original, ground_truth))
        logging.info(f"Predicted original: {pred_original}, explicit: N/A, implicit: N/A, ground truth: {ground_truth}")
    
    logging.info(f"\n----------{model.model_id} original prompting result----------")
    logging.info(f"Out of {len(data)} samples, undecided samples: {undecided}, processed samples: {len(valid_processed_samples)}")

    return valid_processed_samples

def prediction_cues(model, data, cities, countries, populations, household_incomes, processed_original):
    valid_processed_explicit_samples = [] # List of (index, pred_explicit, target)
    valid_processed_implicit_samples = [] # List of (index, pred_implicit, target)
    count = 0
    undecided_explicit = 0
    undecided_implicit = 0
    original_dict = {idx: pred for idx, pred, _ in processed_original}

    for idx, row in data.iterrows():
        count += 1
        logging.info(f"-----Running {model.model_id} zeroshot prompting ({count}/{len(data)})-----")
        text = row['text']
        ground_truth = int(row['labels'])
        pred_explicit = model.zeroshot_explicit_prompting(text, cities, countries)
        pred_implicit = model.zeroshot_implicit_prompting(text, populations, household_incomes)
        pred_original = original_dict.get(idx, "N/A")
        
        if pred_explicit == -1:
            undecided_explicit += 1
        else:
            valid_processed_explicit_samples.append((idx, pred_explicit, ground_truth))
        
        if pred_implicit == -1:
            undecided_implicit += 1
        else:
            valid_processed_implicit_samples.append((idx, pred_implicit, ground_truth))
        logging.info(f"Predicted original: {pred_original}, explicit: {pred_explicit}, implicit: {pred_implicit}, ground truth: {ground_truth}")

    logging.info(f"\n----------{model.model_id} cues prompting result----------")
    logging.info(f"Explicit: Out of {len(data)} samples, undecided samples: {undecided_explicit}, processed samples: {len(valid_processed_explicit_samples)}")
    logging.info(f"Implicit: Out of {len(data)} samples, undecided samples: {undecided_implicit}, processed samples: {len(valid_processed_implicit_samples)}")

    return valid_processed_explicit_samples, valid_processed_implicit_samples

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

    # metrics = {
    #     "precision": f'{precision:.2f}',
    #     "recall": f'{recall:.2f}',
    #     "accuracy": f'{accuracy:.2f}',
    #     "micro_f1": f'{micro_f1:.2f}',
    #     "macro_f1": f'{macro_f1:.2f}',
    #     "confusion_matrix": conf_matrix.tolist() 
    # }
    return metrics

def evaluate(valid_samples_original, valid_samples_explicit, valid_samples_implicit, json_filename, task, city_flips=None, same_size_predictions=None, avg_explicit_metrics=None, avg_implicit_metrics=None ):  
    processed_original = [pred for idx, pred, target in valid_samples_original]
    processed_targets = [target for idx, pred, target in valid_samples_original]
    processed_explicit = [pred for idx, pred, target in valid_samples_explicit]
    processed_targets_explicit = [target for idx, pred, target in valid_samples_explicit]
    processed_implicit = [pred for idx, pred, target in valid_samples_implicit]
    processed_targets_implicit = [target for idx, pred, target in valid_samples_implicit]

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
            common_indices_same_size_explicit = sorted(set(city1_explicit_dict.keys()) & set(city2_explicit_dict.keys()))
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
            common_indices_same_size_implicit = sorted(set(city1_implicit_dict.keys()) & set(city2_implicit_dict.keys()))
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
    results = {
        "original": {k: str(v) if isinstance(v, (int, float)) else v for k, v in metrics_original.items()},  
        "explicit": {k: str(v) if isinstance(v, (int, float)) else v for k, v in metrics_explicit.items()},  
        "implicit": {k: str(v) if isinstance(v, (int, float)) else v for k, v in metrics_implicit.items()},  
        "avg_explicit": avg_explicit_metrics,  
        "avg_implicit": avg_implicit_metrics,  
        "city_flips": city_flips_results,
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
            "explicit": processed_explicit,
            "targets_explicit": processed_targets_explicit,
            "implicit": processed_implicit,
            "targets_implicit": processed_targets_implicit
        }
    }
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved evaluation metrics and predictions to {json_filename}")