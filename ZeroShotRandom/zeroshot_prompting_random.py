import warnings
warnings.filterwarnings('ignore')
from zeroshot_functions_random import *
from zeroshot_utils_random import *
import json
import numpy as np 

if __name__ == '__main__':
    args = parse_args()
    json_filename = set_logging(args, 'Zeroshot')

    dataset, max_input_token_length = read_data(args.task, args.num_of_samples)

    model = Model(model_id=args.model,
                  max_new_tokens=args.max_new_tokens,
                  max_input_token_length=max_input_token_length,
                  task=args.task)

    # Run original prompting once for the entire dataset
    logging.info("----- Running original prompting once for the entire dataset -----")
    valid_samples_original = prediction_original(model, dataset, args.batch_size)
    logging.info(f"Original prompting completed: {len(valid_samples_original)} valid samples")
    
    # Process original predictions for lookup
    processed_original = [pred for idx, pred, target in valid_samples_original]

    # Run random prompting once
    logging.info(f"----- Running random prompting (tokens={args.random_tokens_count}) -----")
    valid_samples_random = prediction_random(
        model, 
        dataset, 
        valid_samples_original, 
        args.batch_size, 
        args.random_tokens_count
    )
    logging.info(f"Random prompting completed: {len(valid_samples_random)} valid samples")

    # Evaluate and Save
    evaluate(
        valid_samples_original=valid_samples_original,
        valid_samples_random=valid_samples_random,
        json_filename=json_filename,
        task=args.task
    )