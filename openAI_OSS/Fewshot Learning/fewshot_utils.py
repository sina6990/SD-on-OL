import logging
import argparse
import re
import os
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='fewshot prompting')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Model name from HuggingFace')
    #parser.add_argument('--temperature', type=float, default=0.001, help='Temperature for generation')
    #parser.add_argument('--top_p', type=float, default=0.5, help='Top-p sampling for generation')
    #parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling for generation')
    #parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty for generation')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Max new tokens for generation')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logging')
    return parser.parse_args()

def set_logging(args, description):
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    # Extract model name from args.model (e.g., 'meta-llama/Llama-3.2-3B-Instruct' -> 'Llama-3.2-3B-Instruct')
    match = re.search(r'([^/\\]+)$', args.model)
    model_short_name = match.group() if match else args.model.replace('/', '_')  # Fallback: replace slashes with underscores
    run_id = f"{timestamp}"
    data = "hate_speech"
    log_path = os.path.join(args.log_dir, model_short_name)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    json_filename = log_path + '/' + description + '_' + model_short_name + '_' + data + '_' + run_id + '.json'
    log_filename = log_path + '/' + description + '_' + model_short_name + '_' + data + '_' + run_id + '.log'
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
        logging.info(str(arg)+' = '+str(getattr(args, arg)))
    logging.info("------------------------------\n")
    return json_filename