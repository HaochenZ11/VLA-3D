import argparse
import json
import os
import multiprocessing as mp
from tqdm import tqdm

from relationship_classes.Binary_Relation import Binary
from relationship_classes.Ordered_Relation import Ordered
from relationship_classes.Ternary_Relation import Ternary
from object_filtering.ObjectFilter import ObjectFilter
from logger.Logger import Logger
from utils import purge_existing_language_data
from timeit import default_timer as timer



def parse_args():
    """
    Parses arguments from command line for input scene graph and language template
        Returns:
            args(ArgumentParser): the ArguementParser object with the necessary arguments added
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='configs/language_generation_configs.json',
                        help="File path to the generation configs")

    args = parser.parse_args()

    return args

def create_relationship_statement_generator(relationship_template, spatial_relations, objects, objects_class_region, language_template, object_filter):
    with open(relationship_template) as r:
        relationship_template_dict = json.load(r)

    relation_type = relationship_template_dict["relation_type"]

    if relation_type == 'binary':
        generator = Binary(spatial_relations, relationship_template_dict, objects, objects_class_region, language_template, object_filter)
    elif relation_type == 'ternary':
        generator = Ternary(spatial_relations, relationship_template_dict, objects, objects_class_region, language_template, object_filter)
    elif relation_type == 'ordered':
        generator = Ordered(spatial_relations, relationship_template_dict, objects, objects_class_region, language_template, object_filter)
    else:
        raise Exception("relationship type must be binary, ternary, or ordered")

    return generator


def get_region_statements(region, spatial_relations, objects, objects_class_region, relationship_templates, generation_configs):
    statements = {}

    if len(region['objects']) == 0:
        return statements


    object_filter = ObjectFilter(spatial_relations, objects, objects_class_region)

    relation_statement_generators = []
    for rt in relationship_templates:
        generator = create_relationship_statement_generator(rt, spatial_relations, objects, objects_class_region, language_template, object_filter)
        relation_statement_generators.append(generator)

    # Generate the statements for each relation
    for statement_generator in relation_statement_generators:
        # cur_time = timer()
        statements.update(statement_generator.generate_statements(generation_configs))
        # print(f'{type(statement_generator)}: {timer() - cur_time}')

    return statements

def get_scene_statement(scene_graph, spatial_relations_file, objects_file, object_class_file, relationship_templates, logger, generation_configs):
    # Generate language data by iterating through all regions in a scene
    scene_language_data = {}

    # Add some metadata for the output JSON
    scene_language_data["scene_name"] = scene_graph["scene_name"]
    scene_language_data["regions"] = {}
    logger[scene_graph["scene_name"]] = {}

    scene_logger = logger[scene_graph["scene_name"]]

    scene_start_time = timer()

    scene_logger['regions'] = {}
    scene_logger_region = scene_logger['regions']

    with open(spatial_relations_file) as sr:
        spatial_relations = json.load(sr)
    with open(objects_file) as o:
        objects = json.load(o)
    with open(object_class_file) as oc:
        object_classes = json.load(oc)

    for region_idx in scene_graph['regions']:
        scene_graph_region = scene_graph['regions'][region_idx]
        spatial_relations_region = spatial_relations[region_idx]
        objects_region = objects[region_idx]
        objects_class_region = object_classes[region_idx]

        region_statements = get_region_statements(scene_graph_region, spatial_relations_region, objects_region, objects_class_region, relationship_templates, generation_configs)

        scene_language_data["regions"][region_idx] = region_statements

        scene_language_data["regions"][region_idx]['region'] = scene_graph_region["region_name"]


    scene_logger['generation_time'] = timer()-scene_start_time


    return scene_language_data




def process_file(scene_path):


    subdir, file_path = scene_path

    scene_name = os.path.basename(file_path).replace("_scene_graph.json", "")
        # if not os.path.exists(os.path.join(subdir, f"{scene_name}_label_data.json")):

        # print(f"Skipping {scene_name} ({counter.value}/{len(scene_paths)})")

    with open(file_path) as sg:
        scene_graph = json.load(sg)

    spatial_relations_file = file_path.replace("_scene_graph.json", "_grouped.json")
    objects_file = file_path.replace("_scene_graph.json", "_object_data.json")
    object_class_file = file_path.replace("_scene_graph.json", "_object_class.json")


    scene_language_data = get_scene_statement(scene_graph, spatial_relations_file, objects_file, object_class_file, relationship_templates, logger.log_buffer, generation_configs)


    # Save dataset of statements and object ground truth data
    with open(os.path.join(subdir, f"{scene_name}_referential_statements.json"), "w") as outfile:
        json.dump(scene_language_data, outfile, indent=4)

    # Save dataset of just statements
    # all_statements = []
    # for region in scene_language_data['regions']:
    #     for statement in scene_language_data['regions'][region]:
    #         if statement != "region":
    #             all_statements.append(statement)
    # with open(os.path.join(subdir, f"{scene_name}_statement.json"), "w") as outfile:
    #     json.dump(all_statements, outfile)
    
    print(f"Completed {scene_path[1]}")


if __name__ == '__main__':

    total_start_time = timer()
    total_scene_count = 0

#Load the scene graph and language templates from their JSON files into dicts
    args = parse_args()

    with open(args.configs) as c:
        generation_configs = json.load(c)

    with open(generation_configs['language_template']) as lt:
        language_template = json.load(lt)

    relationship_templates = generation_configs["relationship_templates"]

    single_file = generation_configs["single_file"]
    relationship_templates = generation_configs["relationship_templates"]
    scene_data_root = generation_configs["scene_data_root"]

    output_dir = generation_configs["log_output_dir"]

    logger = Logger(output_dir)

    if single_file:
        with open(scene_data_root) as sg:
            scene_graph = json.load(sg)

        spatial_relations_file = scene_data_root.replace("_scene_graph.json", "_grouped.json")
        objects_file = scene_data_root.replace("_scene_graph.json", "_object_data.json")

        scene_language_data = get_scene_statement(scene_graph, spatial_relations_file, objects_file, relationship_templates, logger.log_buffer, generation_configs)


        # Save dataset of statements and object ground truth data
        with open(f"./output/statement_label_data.json", "w") as outfile:
            json.dump(scene_language_data, outfile)

            # Save dataset of just statements
        all_statements = []
        for region in scene_language_data['regions']:
            for statement in scene_language_data['regions'][region]:
                if statement != "region":
                    all_statements.append(statement)

        with open(f"output/statements.json", "w") as outfile:
            json.dump(all_statements, outfile)

        total_scene_count+=1


    else:

        if generation_configs["purge_existing_data"]:
            purge_existing_language_data(scene_data_root)

        scene_paths = []

        scene_dirs = next(os.walk(scene_data_root))[1]
        for dataset in scene_dirs:
            for dir in os.listdir(os.path.join(scene_data_root, dataset)):
                scene_path = os.path.join(scene_data_root, dataset, dir)
                for file in os.listdir(scene_path):

                    already_processed = False
                    if not generation_configs["purge_existing_data"]:
                        for file1 in os.listdir(os.path.join(scene_data_root, dir)):
                            if 'label_data_new' in file1 or 'statement_new' in file1:
                                already_processed = True
                                break

                    if already_processed:
                        break

                    if (file != f'{dir}_scene_graph.json'):
                        continue


                    file_path = os.path.join(scene_path, file)
                    scene_paths.append((scene_path, file_path))
        
        print(f"Processing a total of {len(scene_paths)} scenes")

        with mp.Pool(mp.cpu_count()) as pool:
            for _ in tqdm(pool.imap(process_file, scene_paths), total=len(scene_paths)):
                pass

        # process_file(scene_paths[0])


        logger.generate_logs(scene_data_root=scene_data_root)

    print(f"Took {timer()-total_start_time} to complete")

