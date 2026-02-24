import logging
import argparse
import re
import os
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='zeroshot prompting')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Model name from HuggingFace')
    parser.add_argument('--num_of_samples', type=int, default=1500, help='Number of samples in the dataset')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Max new tokens for generation')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logging')
    parser.add_argument('--task', type=str, default='hatespeech', help='Detection task (e.g., hatespeech, sarcasm, etc)')
    parser.add_argument('--random_tokens_count', type=int, default=3, help='Number of random tokens to prefix')
    return parser.parse_args()

def set_logging(args, description):
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    match = re.search(r'([^/\\]+)$', args.model) # Extract model name from args.model (e.g., 'meta-llama/Llama-3.2-3B-Instruct' -> 'Llama-3.2-3B-Instruct')
    model_short_name = match.group() if match else args.model.replace('/', '_') 
    run_id = f"{timestamp}"
    data = args.task
    random_tokens_count = args.random_tokens_count
    log_path = os.path.join(args.log_dir, model_short_name)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    json_filename = log_path + '/' + description + '_' + model_short_name + '_' + data + '_' + str(random_tokens_count) + '_Random_' + run_id + '.json'
    log_filename = log_path + '/' + description + '_' + model_short_name + '_' + data + '_' + str(random_tokens_count) + '_Random_' + run_id + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("\x1b[38;20m" + ' %(message)s' + "\x1b[0m"))
    logging.getLogger().addHandler(console)
    logging.info("----------Arguments-----------")
    for arg in vars(args):
        logging.info(str(arg) + ' = ' + str(getattr(args, arg)))
    logging.info("------------------------------\n")
    return json_filename