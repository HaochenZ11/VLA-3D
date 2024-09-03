import json
import os
from pathlib import Path
import multiprocessing as mp
from timeit import default_timer as timer
from tqdm import tqdm


def summarize_stats(stats):
    summary = {
        "all_statements": 0,
        "above_statements": 0,
        "below_statements": 0,
        "between_statements": 0,
        "near_statements": 0,
        "closest_statements": 0,
        "second_closest_statements": 0,
        "third_closest_statements": 0,
        "farthest_statements": 0,
        "second_farthest_statements": 0,
        "third_farthest_statements": 0,
        "on_statements": 0,
        "in_statements": 0,
        "color_references": 0,
        "size_references": 0,
        "color_statements": 0,
        "size_statements": 0,
        "unique_objects": 0,
    }

    if generation_configs["generate_false_statements"]:
        summary.update({
            "false_all_statements": 0,
            "false_target_statements": 0,
            "false_anchor_statements": 0,
            "false_color_statements": 0,
            "false_class_statements": 0,
            "false_target_color_statements": 0,
            "false_target_class_statements": 0,
            "false_anchor_color_statements": 0,
            "false_anchor_class_statements": 0,
        })

        if generation_configs["generate_false_statements"]:
            summary.update({
                f"false_all_statements": 0,
            })

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


def generate_stats(target_file, stats, object_set, colors_set, sizes_set):
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
        "second_closest_statements": 0,
        "third_closest_statements": 0,
        "farthest_statements": 0,
        "second_farthest_statements": 0,
        "third_farthest_statements": 0,
        "color_references": 0,
        "size_references": 0,
        "color_statements": 0,
        "size_statements": 0,
        "unique_objects": 0,
    }

    if generation_configs["generate_false_statements"]:
        stats[data['scene_name']].update({
            "false_all_statements": 0,
            "false_target_statements": 0,
            "false_anchor_statements": 0,
            "false_color_statements": 0,
            "false_class_statements": 0,
            "false_target_color_statements": 0,
            "false_target_class_statements": 0,
            "false_anchor_color_statements": 0,
            "false_anchor_class_statements": 0,
        })

        for relationship in relationships:
            stats[data['scene_name']].update({
                f"false_{relationship}_statements": 0,
                f"false_color_{relationship}_statements": 0,
                f"false_class_{relationship}_statements": 0,
                f"false_target_{relationship}_statements": 0,
                f"false_target_color_{relationship}_statements": 0,
                f"false_target_class_{relationship}_statements": 0,
                f"false_anchor_{relationship}_statements": 0,
                f"false_anchor_color_{relationship}_statements": 0,
                f"false_anchor_class_{relationship}_statements": 0,
            })

    scene_stats = stats[data['scene_name']]

    for regions, statements in data['regions'].items():
        scene_stats['all_statements'] += len(statements)

        for statement, labels in statements.items():
            if statement == 'region':
                continue

            for label in labels:
                scene_stats[f"{label['relation']}_statements"] += 1

                color_statement = False
                size_statement = False

                if len(label['target_color_used']) > 0:
                    scene_stats['color_references'] += 1
                    color_statement = True
                    colors_set.add(label['target_color_used'])
                if len(label['target_size_used']) > 0:
                    scene_stats['size_references'] += 1
                    size_statement = True
                    sizes_set.add(label['target_size_used'])
                object_set.add(label['target_class'])

                for anchor, anchor_info in label['anchors'].items():
                    if len(anchor_info['color_used']) > 0:
                        scene_stats['color_references'] += 1
                        color_statement = True
                        colors_set.add(anchor_info['color_used'])
                    if len(anchor_info['size_used']) > 0:
                        scene_stats['size_references'] += 1
                        size_statement = True
                        sizes_set.add(anchor_info['size_used'])
                    object_set.add(anchor_info['class'])

                if color_statement:
                    scene_stats['color_statements'] += 1
                if size_statement:
                    scene_stats['size_statements'] += 1

                if "false_statements" in label:
                    if "false_target_color" in label['false_statements']:
                        scene_stats['false_all_statements'] += 1
                        scene_stats['false_target_statements'] += 1
                        scene_stats['false_color_statements'] += 1
                        scene_stats['false_target_color_statements'] += 1
                        scene_stats[f'false_{label["relation"]}_statements'] += 1
                        scene_stats[f'false_color_{label["relation"]}_statements'] += 1
                        scene_stats[f'false_target_{label["relation"]}_statements'] += 1
                        scene_stats[f'false_target_color_{label["relation"]}_statements'] += 1

                    if "false_target_class" in label['false_statements']:
                        scene_stats['false_all_statements'] += 1
                        scene_stats['false_target_statements'] += 1
                        scene_stats['false_class_statements'] += 1
                        scene_stats['false_target_class_statements'] += 1
                        scene_stats[f'false_{label["relation"]}_statements'] += 1
                        scene_stats[f'false_class_{label["relation"]}_statements'] += 1
                        scene_stats[f'false_target_{label["relation"]}_statements'] += 1
                        scene_stats[f'false_target_class_{label["relation"]}_statements'] += 1

                    if "false_anchors" in label['false_statements']:
                        for anchor, false_anchor_statements in label['anchors'].items():
                            if "false_anchor_color" in false_anchor_statements:
                                scene_stats['false_all_statements'] += 1
                                scene_stats['false_anchor_statements'] += 1
                                scene_stats['false_color_statements'] += 1
                                scene_stats['false_anchor_color_statements'] += 1
                                scene_stats[f'false_{label["relation"]}_statements'] += 1
                                scene_stats[f'false_color_{label["relation"]}_statements'] += 1
                                scene_stats[f'false_anchor_{label["relation"]}_statements'] += 1
                                scene_stats[f'false_anchor_color_{label["relation"]}_statements'] += 1

                            if "false_anchor_class" in false_anchor_statements:
                                scene_stats['false_all_statements'] += 1
                                scene_stats['false_anchor_statements'] += 1
                                scene_stats['false_class_statements'] += 1
                                scene_stats['false_anchor_class_statements'] += 1
                                scene_stats[f'false_{label["relation"]}_statements'] += 1
                                scene_stats[f'false_class_{label["relation"]}_statements'] += 1
                                scene_stats[f'false_anchor_{label["relation"]}_statements'] += 1
                                scene_stats[f'false_anchor_class_{label["relation"]}_statements'] += 1

if __name__ == '__main__':

    total_start_time = timer()

    configs_path = Path(__file__).parents[1].resolve() / 'configs' / 'language_generation_configs.json'
    print(configs_path)
    with open(configs_path) as c:
        generation_configs = json.load(c)

    target_dir = generation_configs["scene_data_root"]

    relationships = generation_configs["relationships"]

    target_paths = []

    dataset_dirs = next(os.walk(target_dir))[1]
    for dataset in dataset_dirs:
        scene_dirs = next(os.walk(os.path.join(target_dir, dataset)))[1]
        for scene_dir in scene_dirs:
            scene_path = os.path.join(target_dir, dataset, scene_dir)
            file_path = os.path.join(scene_path, f'{scene_dir}_referential_statements.json')
            target_paths.append(file_path)

    stats = {}
    object_set = set()
    colors_set = set()
    sizes_set = set()

    for target_file in tqdm(target_paths):
        generate_stats(target_file, stats, object_set, colors_set, sizes_set)

    summary, max_all_statements, min_all_statements = summarize_stats(stats)

    print("Summary:", summary)
    print("Max all_statements:", max_all_statements)
    print("Min all_statements:", min_all_statements)

    stats["unique_objects"] = len(object_set)
    stats["unique_colors"] = len(colors_set)
    stats["unique_sizes"] = len(sizes_set)

    summary["unique_objects"] = len(object_set)
    summary["unique_colors"] = len(colors_set)
    summary["unique_sizes"] = len(sizes_set)

    print("Unique objects:", len(object_set))
    print("Unique colors:", len(colors_set))
    print("Unique sizes:", len(sizes_set))

    print(f"Took {timer() - total_start_time} to preprocess")

    log_output_dir = "logger/output"

    # Save the stats to a JSON file
    with open(os.path.join(log_output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)

    # Save the summary to a separate JSON file
    with open(os.path.join(log_output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'summary': summary,
            'max_all_statements': max_all_statements,
            'min_all_statements': min_all_statements,
            'colors_used': list(colors_set),
            'sizes_used': list(sizes_set),
            'object_set': list(object_set)
        }, f)