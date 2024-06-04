import json
import os
import multiprocessing as mp
from timeit import default_timer as timer


def summarize_stats(stats):
    summary = {
        "all_statements": 0,
        "above_statements": 0,
        "below_statements": 0,
        "between_statements": 0,
        "near_statements": 0,
        "closest_statements": 0,
        "farthest_statements": 0,
        "on_statements": 0,
        "in_statements": 0,
        "color_references": 0,
        "size_references": 0,
        "color_statements": 0,
        "size_statements": 0,
        "unique_objects": 0,
    }

    max_all_statements = {"key": None, "value": 0}
    min_all_statements = {"key": None, "value": float('inf')}

    for key, value in stats.items():
        for sub_key in summary.keys():
            summary[sub_key] += value[sub_key]

        if value["all_statements"] > max_all_statements["value"]:
            max_all_statements = {"key": key, "value": value["all_statements"]}

        if value["all_statements"] < min_all_statements["value"]:
            min_all_statements = {"key": key, "value": value["all_statements"]}

    return summary, max_all_statements, min_all_statements


def generate_stats(target_file, stats, object_set):
    with open(target_file) as f:
        data = json.load(f)
    
    stats[data['scene_name']] = {
        "all_statements": 0,
        "above_statements": 0,
        "below_statements": 0,
        "between_statements": 0,
        "near_statements": 0,
        "on_statements": 0,
        "in_statements": 0,
        "closest_statements": 0,
        "farthest_statements": 0,
        "color_references": 0,
        "size_references": 0,
        "color_statements": 0,
        "size_statements": 0,
        "unique_objects": 0,
    }

    scene_stats = stats[data['scene_name']]
    
    for regions, statements in data['regions'].items():
        scene_stats['all_statements'] += len(statements)

        for statement, labels in statements.items():
            if statement == 'region':
                continue

            for label in labels:
                scene_stats[f"{label['relation']}_statements"] += 1

                if label['target_color_used']:
                    scene_stats['color_references'] += 1
                if label['target_size_used']:
                    scene_stats['size_references'] += 1

                object_set.add(label['target_class'])

                if label['relation_type'] == 'ternary':
                    if label['anchor1_color_used']:
                        scene_stats['color_references'] += 1
                    if label['anchor1_size_used']:
                        scene_stats['size_references'] += 1
                    if label['anchor2_color_used']:
                        scene_stats['color_references'] += 1
                    if label['anchor2_size_used']:
                        scene_stats['size_references'] += 1
                    if label['target_color_used'] or label['anchor1_color_used'] or label['anchor2_color_used']:
                        scene_stats['color_statements'] += 1
                    if label['target_size_used'] or label['anchor1_size_used'] or label['anchor2_size_used']:
                        scene_stats['size_statements'] += 1

                    object_set.add(label['anchor1_class'])
                    object_set.add(label['anchor2_class'])

                else:
                    if label['anchor_color_used']:
                        scene_stats['color_references'] += 1
                    if label['anchor_size_used']:
                        scene_stats['size_references'] += 1
                    if label['target_color_used'] or label['anchor_color_used']:
                        scene_stats['color_statements'] += 1
                    if label['target_size_used'] or label['anchor_size_used']:
                        scene_stats['size_statements'] += 1

                    object_set.add(label['anchor_class'])



    

if __name__ == '__main__':

    total_start_time = timer()

    target_dir = "data/test_dataset"

    target_paths = []


    # Open the JSON file
    scene_dirs = next(os.walk(target_dir))[1]
    for dir in scene_dirs:
        scene_path = os.path.join(target_dir, dir)
        file_path = os.path.join(scene_path,  f'{dir}_label_data_new.json')
        target_paths.append(file_path)


    stats = {}
    object_set = set()

    for target_file in target_paths:
        generate_stats(target_file, stats, object_set)

    summary, max_all_statements, min_all_statements = summarize_stats(stats)

    print("Summary:", summary)
    print("Max all_statements:", max_all_statements)
    print("Min all_statements:", min_all_statements)

    stats["unique_objects"] = len(object_set)

    print("Unique objects:", len(object_set))

    print(f"Took {timer( ) - total_start_time} to preprocess")

    log_output_dir = "logger/output"

    # Save the stats to a JSON file
    with open(os.path.join(log_output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)

    # Save the summary to a separate JSON file
    with open(os.path.join(log_output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'summary': summary,
            'max_all_statements': max_all_statements,
            'min_all_statements': min_all_statements
        }, f)