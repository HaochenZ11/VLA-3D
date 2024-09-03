import json
import os
import multiprocessing as mp
from timeit import default_timer as timer
from pathlib import Path


def group_by_nyu_label(relation_dict):
    # Initialize a dictionary to hold the grouped relationships
    grouped_by_anchor_relationships = {}
    grouped_by_target_relationships = {}

    # Iterate over all regions
    for region_id, region in relation_dict['regions'].items():
        grouped_by_anchor_relationships[region_id] = {}
        grouped_by_target_relationships[region_id] = {}
        # Extract the objects and their labels
        objects = {obj['object_id']: obj['nyu_label'] for obj in region['objects']}

        # Iterate over the relationships
        for relation, related_objects in region['relationships'].items():
            # breakpoint()
            if relation == 'between':
                if relation not in grouped_by_anchor_relationships:
                    grouped_by_anchor_relationships[region_id][relation] = {}

                for target_id, anchors_ids in related_objects.items():
                    if len(anchors_ids) == 0:
                        continue
                    # Get the label of the object
                    target_label = objects[target_id]

                    if target_label not in grouped_by_anchor_relationships[region_id][relation]:
                        grouped_by_anchor_relationships[region_id][relation][target_label] = {}

                    grouped_by_anchor_relationships[region_id][relation][target_label][target_id] = anchors_ids

            else:
                # Iterate over the related objects
                if relation not in grouped_by_anchor_relationships:
                    grouped_by_anchor_relationships[region_id][relation] = {}
                if relation not in grouped_by_target_relationships:
                    grouped_by_target_relationships[region_id][relation] = {}

                for anchor_id, target_ids in related_objects.items():

                    if len(target_ids) == 0:
                        continue
                    # Get the label of the object
                    anchor_label = objects[anchor_id]

                    if anchor_label not in grouped_by_anchor_relationships[region_id][relation]:
                        grouped_by_anchor_relationships[region_id][relation][anchor_label] = {}

                    grouped_by_anchor_relationships[region_id][relation][anchor_label][anchor_id] = {}

                    # Iterate over the related object ids
                    for target_id in target_ids:
                        # Get the label of the related object
                        target_label = objects[target_id]

                        if target_label not in grouped_by_target_relationships[region_id][relation]:
                            grouped_by_target_relationships[region_id][relation][target_label] = {}

                        if target_id not in grouped_by_target_relationships[region_id][relation][target_label]:
                            grouped_by_target_relationships[region_id][relation][target_label][target_id] = {}

                        if anchor_label not in grouped_by_target_relationships[region_id][relation][target_label][target_id]:
                            grouped_by_target_relationships[region_id][relation][target_label][target_id][anchor_label] = []

                        grouped_by_target_relationships[region_id][relation][target_label][target_id][anchor_label].append(anchor_id)

                        # Add the related object label to the grouped relationships
                        if target_label not in grouped_by_anchor_relationships[region_id][relation][anchor_label][anchor_id]:
                            grouped_by_anchor_relationships[region_id][relation][anchor_label][anchor_id][target_label] = []

                        grouped_by_anchor_relationships[region_id][relation][anchor_label][anchor_id][target_label].append(
                            target_id)

    return grouped_by_anchor_relationships, grouped_by_target_relationships


def index_by_object_id_and_class(relation_dict):
    object_index = {}
    object_class = {}

    regions = relation_dict['regions']

    for region_id, region in regions.items():
        if region_id not in object_index:
            object_index[region_id] = {}
        if region_id not in object_class:
            object_class[region_id] = {}

        for obj in region['objects']:
            object_index[region_id][obj['object_id']] = obj

            obj['largest_face_area'] = max(
                obj['size'][0] * obj['size'][1],
                obj['size'][0] * obj['size'][2],
                obj['size'][1] * obj['size'][2])

            if obj['nyu_label'] not in object_class[region_id]:
                object_class[region_id][obj['nyu_label']] = []

            object_class[region_id][obj['nyu_label']].append(obj['object_id'])

    return object_index, object_class


def process_file(target_file):
    with open(target_file, 'r') as file:
        # Load the JSON data from the file
        relation_dict = json.load(file)

    print(target_file)
    grouped_by_anchor_relationships, grouped_by_target_relationships = group_by_nyu_label(relation_dict)
    object_index, object_class = index_by_object_id_and_class(relation_dict)

    # Save the output to a JSON file
    grouped_by_target_output_file = target_file.replace('_scene_graph.json', '_grouped_by_target.json')
    with open(grouped_by_target_output_file, 'w') as file:
        json.dump(grouped_by_target_relationships, file)

    grouped_by_anchor_output_file = target_file.replace('_scene_graph.json', '_grouped_by_anchor.json')
    with open(grouped_by_anchor_output_file, 'w') as file:
        json.dump(grouped_by_anchor_relationships, file)

    # Save the object data to a JSON file
    object_output_file = target_file.replace('_scene_graph.json', '_object_data.json')
    with open(object_output_file, 'w') as file:
        json.dump(object_index, file)

    object_class_output_file = target_file.replace('_scene_graph.json', '_object_class.json')
    with open(object_class_output_file, 'w') as file:
        json.dump(object_class, file)


if __name__ == '__main__':

    total_start_time = timer()

    configs_path = Path(__file__).parents[1].resolve() / 'configs' / 'language_generation_configs.json'
    print(configs_path)
    with open(configs_path) as c:
        generation_configs = json.load(c)

    target_dir = generation_configs["scene_data_root"]
    print(target_dir)
    # relations = generation_configs["relations"]

    target_paths = []


    print(target_dir)
    # Open the JSON file
    dataset_dirs = next(os.walk(target_dir))[1]
    for dataset in dataset_dirs:
        scene_dirs = next(os.walk(os.path.join(target_dir, dataset)))[1]
        for scene_dir in scene_dirs:
            scene_path = os.path.join(target_dir, dataset, scene_dir)
            for file in os.listdir(scene_path):
                if (file != f'{scene_dir}_scene_graph.json'):
                    continue

                file_path = os.path.join(scene_path, file)
                target_paths.append(file_path)

    # with mp.Pool(mp.cpu_count()) as pool:
    #     pool.map(process_file, target_paths)
    # process_file(target_paths[2])
    for target_file in target_paths:
        process_file(target_file)

    print(f"Took {timer() - total_start_time} to preprocess")