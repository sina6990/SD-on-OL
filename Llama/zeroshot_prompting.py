import warnings
warnings.filterwarnings('ignore')
from functions import *
from utils import *
import json

if __name__ == '__main__':
    args = parse_args()
    json_filename = set_logging(args, 'zeroshot')

    dataset, max_input_token_length = read_data()

    modelLlama = LlamaModel(model_id=args.model,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            max_new_tokens=args.max_new_tokens,
                            max_input_token_length=max_input_token_length)

    with open('MSAs.json', 'r') as f:
        msas = json.load(f)
    
    # Run original prompting once for the entire dataset
    logging.info("----- Running original prompting once for the entire dataset -----")
    processed_original, processed_targets = prediction_original(modelLlama, dataset)
    logging.info(f"Original prompting completed: {len(processed_original)} valid samples")

    # Iterate over countries and city classes
    for country in ['United States', 'Canada', 'Australia']:
        for class_size in ['Large', 'Medium', 'Small']:
            cities_dict = msas[country][class_size]
            cities = list(cities_dict.keys())
            logging.info(f"----- Running aggregated for {country} - {class_size} ({', '.join(cities)}) -----")

            # Initialize lists to aggregate results across cities
            all_processed_original = []
            all_processed_explicit = []
            all_processed_implicit = []
            all_processed_targets = []

            # Run explicit and implicit prompting for each city
            for city in cities:
                population = cities_dict[city]['population']
                household_income = cities_dict[city]['householdIncome']
                logging.info(f"----- Processing city: {city} (explicit and implicit only) -----")
                # Run only explicit and implicit prompting, passing precomputed original results
                processed_explicit, processed_implicit, processed_targets_city = prediction_cues(
                    modelLlama, dataset, [city], [population], [household_income],
                    processed_original, processed_targets
                )
                # Extend aggregated lists
                all_processed_original.extend(processed_original)
                all_processed_explicit.extend(processed_explicit)
                all_processed_implicit.extend(processed_implicit)
                all_processed_targets.extend(processed_targets_city)

            # Generate unique JSON filename for the class
            unique_json = f"{json_filename.rsplit('.', 1)[0]}_{country}_{class_size}.json"
            evaluate(all_processed_original, all_processed_explicit, all_processed_implicit, all_processed_targets, unique_json)
