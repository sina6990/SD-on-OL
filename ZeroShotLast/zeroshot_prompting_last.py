import warnings
warnings.filterwarnings('ignore')
from zeroshot_functions_last import *
from zeroshot_utils_last import *
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

    with open('../MSAs.json', 'r') as f:
        msas = json.load(f)
    
    # Run original prompting once for the entire dataset
    logging.info("----- Running original prompting once for the entire dataset -----")
    valid_samples_original = prediction_original(model, dataset, args.batch_size)
    logging.info(f"Original prompting completed: {len(valid_samples_original)} valid samples")

    # Extract predictions and targets for original
    processed_original = [pred for idx, pred, target in valid_samples_original]
    processed_targets = [target for idx, pred, target in valid_samples_original]

    # Iterate over countries and city classes (9 experiments)
    aggregated_flips_storage = {
        "explicit_helpful": [],
        "explicit_not_helpful": [],
        "implicit_helpful": [],
        "implicit_not_helpful": []
    }
    
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
            all_explicit_prompts = []
            all_implicit_prompts = []
            per_city_explicit_metrics = []  
            per_city_implicit_metrics = []  

            # Compute flips for each city — each city is run on ALL samples
            for city in cities:
                population = cities_dict[city]['population']
                household_income = cities_dict[city]['householdIncome']
                logging.info(f"----- Processing city: {city} (explicit and implicit only) -----")

                valid_samples_explicit, valid_samples_implicit, valid_explicit_prompts, valid_implicit_prompts = prediction_cues(
                    model, dataset,
                    [city],                 
                    [country],
                    [population],          
                    [household_income],             
                    valid_samples_original,
                    args.batch_size
                )

                # Compute flips for this city
                original_dict = {idx: pred for idx, pred, _ in valid_samples_original}
                target_dict = {idx: target for idx, _, target in valid_samples_original}
                explicit_dict = {idx: pred for idx, pred, _ in valid_samples_explicit}
                implicit_dict = {idx: pred for idx, pred, _ in valid_samples_implicit}

                common_indices_explicit = sorted(set(original_dict.keys()) & set(explicit_dict.keys()))
                common_indices_implicit = sorted(set(original_dict.keys()) & set(implicit_dict.keys()))

                processed_original_explicit = [original_dict[i] for i in common_indices_explicit]
                processed_explicit = [explicit_dict[i] for i in common_indices_explicit]
                targets_explicit = [target_dict[i] for i in common_indices_explicit]
                processed_original_implicit = [original_dict[i] for i in common_indices_implicit]
                processed_implicit = [implicit_dict[i] for i in common_indices_implicit]
                targets_implicit = [target_dict[i] for i in common_indices_implicit]

                flips_explicit = sum(o != e for o, e in zip(processed_original_explicit, processed_explicit))
                flips_0to1_explicit = sum(o == 0 and e == 1 for o, e in zip(processed_original_explicit, processed_explicit))
                flips_1to0_explicit = sum(o == 1 and e == 0 for o, e in zip(processed_original_explicit, processed_explicit))
                flips_0to1_helpful_explicit = sum(o == 0 and e == 1 and t == 1 for o, e, t in zip(processed_original_explicit, processed_explicit, targets_explicit))
                flips_0to1_nothelpful_explicit = sum(o == 0 and e == 1 and t == 0 for o, e, t in zip(processed_original_explicit, processed_explicit, targets_explicit))
                flips_1to0_helpful_explicit = sum(o == 1 and e == 0 and t == 0 for o, e, t in zip(processed_original_explicit, processed_explicit, targets_explicit))
                flips_1to0_nothelpful_explicit = sum(o == 1 and e == 0 and t == 1 for o, e, t in zip(processed_original_explicit, processed_explicit, targets_explicit))
                num_valid_explicit = len(common_indices_explicit)
                flips_implicit = sum(o != i for o, i in zip(processed_original_implicit, processed_implicit))
                flips_0to1_implicit = sum(o == 0 and i == 1 for o, i in zip(processed_original_implicit, processed_implicit))
                flips_1to0_implicit = sum(o == 1 and i == 0 for o, i in zip(processed_original_implicit, processed_implicit))
                flips_0to1_helpful_implicit = sum(o == 0 and i == 1 and t == 1 for o, i, t in zip(processed_original_implicit, processed_implicit, targets_implicit))
                flips_0to1_nothelpful_implicit = sum(o == 0 and i == 1 and t == 0 for o, i, t in zip(processed_original_implicit, processed_implicit, targets_implicit))
                flips_1to0_helpful_implicit = sum(o == 1 and i == 0 and t == 0 for o, i, t in zip(processed_original_implicit, processed_implicit, targets_implicit))
                flips_1to0_nothelpful_implicit = sum(o == 1 and i == 0 and t == 1 for o, i, t in zip(processed_original_implicit, processed_implicit, targets_implicit))
                num_valid_implicit = len(common_indices_implicit)

                city_flips[city] = {
                    'flips_explicit': flips_explicit,
                    'flips_0to1_explicit': flips_0to1_explicit,
                    'flips_1to0_explicit': flips_1to0_explicit,
                    'flips_0to1_helpful_explicit': flips_0to1_helpful_explicit,
                    'flips_0to1_nothelpful_explicit': flips_0to1_nothelpful_explicit,
                    'flips_1to0_helpful_explicit': flips_1to0_helpful_explicit,
                    'flips_1to0_nothelpful_explicit': flips_1to0_nothelpful_explicit,
                    'num_valid_explicit': num_valid_explicit,
                    'flips_implicit': flips_implicit,
                    'flips_0to1_implicit': flips_0to1_implicit,
                    'flips_1to0_implicit': flips_1to0_implicit,
                    'flips_0to1_helpful_implicit': flips_0to1_helpful_implicit,
                    'flips_0to1_nothelpful_implicit': flips_0to1_nothelpful_implicit,
                    'flips_1to0_helpful_implicit': flips_1to0_helpful_implicit,
                    'flips_1to0_nothelpful_implicit': flips_1to0_nothelpful_implicit,
                    'num_valid_implicit': num_valid_implicit
                }
                city_predictions[city] = (valid_samples_explicit, valid_samples_implicit)

                # Changed: Compute and log per-city metrics
                metrics_explicit_per_city = compute_metrics(processed_explicit, targets_explicit)
                metrics_implicit_per_city = compute_metrics(processed_implicit, targets_implicit)
                per_city_explicit_metrics.append(metrics_explicit_per_city)
                per_city_implicit_metrics.append(metrics_implicit_per_city)

                # Log per-city metrics
                logging.info(f"\n----------Per-City Explicit Metrics for {city}----------")
                logging.info(f"- Precision = {metrics_explicit_per_city['precision']}")
                logging.info(f"- Recall = {metrics_explicit_per_city['recall']}")
                logging.info(f"- Accuracy = {metrics_explicit_per_city['accuracy']}")
                logging.info(f"- Micro F1-Score = {metrics_explicit_per_city['micro_f1']}")
                logging.info(f"- Macro F1-Score = {metrics_explicit_per_city['macro_f1']}")
                logging.info(f"- Confusion Matrix = \n{metrics_explicit_per_city['confusion_matrix']}")

                logging.info(f"\n----------Per-City Implicit Metrics for {city}----------")
                logging.info(f"- Precision = {metrics_implicit_per_city['precision']}")
                logging.info(f"- Recall = {metrics_implicit_per_city['recall']}")
                logging.info(f"- Accuracy = {metrics_implicit_per_city['accuracy']}")
                logging.info(f"- Micro F1-Score = {metrics_implicit_per_city['micro_f1']}")
                logging.info(f"- Macro F1-Score = {metrics_implicit_per_city['macro_f1']}")
                logging.info(f"- Confusion Matrix = \n{metrics_implicit_per_city['confusion_matrix']}")

                # Aggregate predictions for metrics (across all cities in the group)
                all_valid_samples_explicit.extend(valid_samples_explicit)
                all_valid_samples_implicit.extend(valid_samples_implicit)
                all_explicit_prompts.extend(valid_explicit_prompts)
                all_implicit_prompts.extend(valid_implicit_prompts)

            if per_city_explicit_metrics:
                avg_explicit_metrics = {
                    key: f'{np.mean([m[key] for m in per_city_explicit_metrics if isinstance(m[key], (int, float))]):.2f}' 
                    for key in per_city_explicit_metrics[0].keys() if key != 'confusion_matrix'
                }
                
                conf_matrices_explicit = [np.array(m['confusion_matrix']) for m in per_city_explicit_metrics]
                avg_conf_explicit = np.mean(conf_matrices_explicit, axis=0).tolist()
                avg_explicit_metrics['confusion_matrix'] = avg_conf_explicit

                avg_implicit_metrics = {
                    key: f'{np.mean([m[key] for m in per_city_implicit_metrics if isinstance(m[key], (int, float))]):.2f}' 
                    for key in per_city_implicit_metrics[0].keys() if key != 'confusion_matrix'
                }
                conf_matrices_implicit = [np.array(m['confusion_matrix']) for m in per_city_implicit_metrics]
                avg_conf_implicit = np.mean(conf_matrices_implicit, axis=0).tolist()
                avg_implicit_metrics['confusion_matrix'] = avg_conf_implicit

            # Extract predictions and targets for explicit and implicit (aggregated across cities)
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
            local_flips = evaluate(
                valid_samples_original=valid_samples_original,
                valid_samples_explicit=all_valid_samples_explicit,
                valid_samples_implicit=all_valid_samples_implicit,
                json_filename=unique_json,
                task=args.task,
                city_flips=city_flips,
                same_size_predictions=same_size_predictions,
                avg_explicit_metrics=avg_explicit_metrics if per_city_explicit_metrics else None,  
                avg_implicit_metrics=avg_implicit_metrics if per_city_implicit_metrics else None,
                dataset=dataset,
                explicit_prompts_list=all_explicit_prompts,
                implicit_prompts_list=all_implicit_prompts,
                city_predictions=city_predictions # Pass the dict of per-city predictions
            )
            
            if local_flips:
                for key in aggregated_flips_storage:
                    aggregated_flips_storage[key].extend(local_flips[key])
                    
    # Save aggregated flips
    aggregated_flips_filename = f"{json_filename.rsplit('.', 1)[0]}_all_flips_prompts.json"
    with open(aggregated_flips_filename, 'w') as f:
        json.dump(aggregated_flips_storage, f, indent=4)
    logging.info(f"Saved aggregated flip data to {aggregated_flips_filename}")
