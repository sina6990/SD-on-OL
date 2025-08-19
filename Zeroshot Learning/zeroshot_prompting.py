import warnings
warnings.filterwarnings('ignore')
from zeroshot_functions import *
from zeroshot_utils import *
import json

if __name__ == '__main__':
    args = parse_args()
    json_filename = set_logging(args, 'zeroshot')

    dataset, max_input_token_length = read_data()

    model = Model(model_id=args.model,
                            # temperature=args.temperature,
                            # top_p=args.top_p,
                            # top_k=args.top_k,
                            # repetition_penalty=args.repetition_penalty,
                            max_new_tokens=args.max_new_tokens,
                            max_input_token_length=max_input_token_length)

    with open('../MSAs.json', 'r') as f:
        msas = json.load(f)
    
    # Run original prompting once for the entire dataset
    logging.info("----- Running original prompting once for the entire dataset -----")
    valid_samples_original = prediction_original(model, dataset)
    logging.info(f"Original prompting completed: {len(valid_samples_original)} valid samples")

    # Extract predictions and targets for original
    processed_original = [pred for idx, pred, target in valid_samples_original]
    processed_targets = [target for idx, pred, target in valid_samples_original]

    # Iterate over countries and city classes (9 experiments)
    for country in ['United States', 'Canada', 'Australia']:
        for class_size in ['Large', 'Medium', 'Small']:
            cities_dict = msas[country][class_size]
            cities = list(cities_dict.keys())
            logging.info(f"----- Running aggregated for {country} - {class_size} ({', '.join(cities)}) -----")

            # Initialize for per-city flips and aggregated metrics
            city_flips = {}
            city_predictions = {}
            all_valid_samples_explicit = []
            all_valid_samples_implicit = []

            # Compute flips for each city
            for city in cities:
                population = cities_dict[city]['population']
                household_income = cities_dict[city]['householdIncome']
                logging.info(f"----- Processing city: {city} (explicit and implicit only) -----")
                valid_samples_explicit, valid_samples_implicit = prediction_cues(
                    model, dataset, [city], [population], [household_income], [country],
                    valid_samples_original
                )

                # Compute flips for this city
                original_dict = {idx: pred for idx, pred, _ in valid_samples_original}
                explicit_dict = {idx: pred for idx, pred, _ in valid_samples_explicit}
                implicit_dict = {idx: pred for idx, pred, _ in valid_samples_implicit}

                common_indices_explicit = sorted(set(original_dict.keys()) & set(explicit_dict.keys()))
                common_indices_implicit = sorted(set(original_dict.keys()) & set(implicit_dict.keys()))

                processed_original_explicit = [original_dict[i] for i in common_indices_explicit]
                processed_explicit = [explicit_dict[i] for i in common_indices_explicit]
                processed_original_implicit = [original_dict[i] for i in common_indices_implicit]
                processed_implicit = [implicit_dict[i] for i in common_indices_implicit]

                flips_explicit = sum(o != e for o, e in zip(processed_original_explicit, processed_explicit))
                num_valid_explicit = len(common_indices_explicit)
                flips_implicit = sum(o != i for o, i in zip(processed_original_implicit, processed_implicit))
                num_valid_implicit = len(common_indices_implicit)

                city_flips[city] = (flips_explicit, num_valid_explicit, flips_implicit, num_valid_implicit)
                city_predictions[city] = (valid_samples_explicit, valid_samples_implicit)

                # Aggregate predictions for metrics
                all_valid_samples_explicit.extend(valid_samples_explicit)
                all_valid_samples_implicit.extend(valid_samples_implicit)

            # Extract predictions and targets for explicit and implicit
            processed_explicit = [pred for idx, pred, target in all_valid_samples_explicit]
            processed_targets_explicit = [target for idx, pred, target in all_valid_samples_explicit]
            processed_implicit = [pred for idx, pred, target in all_valid_samples_implicit]
            processed_targets_implicit = [target for idx, pred, target in all_valid_samples_implicit]

            # Generate unique JSON filename for the class
            unique_json = f"{json_filename.rsplit('.', 1)[0]}_{country}_{class_size}.json"

            # Prepare same-size city flips (if exactly 2 cities)
            same_size_predictions = None
            if len(cities) == 2:
                city1, city2 = cities
                pair_name = f"{country}_{class_size}_{city1.replace(' ', '_')}_vs_{city2.replace(' ', '_')}"
                same_size_predictions = {pair_name: (city_predictions[city1][0], city_predictions[city1][1], 
                                                    city_predictions[city2][0], city_predictions[city2][1])}

            # Evaluate with per-city flips and same-size predictions
            evaluate(
                valid_samples_original=valid_samples_original,
                valid_samples_explicit=all_valid_samples_explicit,
                valid_samples_implicit=all_valid_samples_implicit,
                json_filename=unique_json,
                city_flips=city_flips,
                same_size_predictions=same_size_predictions
            )