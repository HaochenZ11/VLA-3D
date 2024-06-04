import pandas as pd
import argparse
import json
import os

def parse_args():
    """
    Parses arguments from command line for input scene graph and language template
        Returns:
            args(ArgumentParser): the ArguementParser object with the necessary arguments added
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_file', default='./logger/output_full.json',
                        help="File path to the input lof schema")

    parser.add_argument('--stats_type', default='scene_centric',
                        help="scene_centric or language_centric statistics")

    args = parser.parse_args()

    return args

class Logger:
    def __init__(self, output_dir):
        self.log_buffer = {}
        self.output_dir = output_dir

    def generate_logs(self, scene_data_root):
        for subdir, dirs, files in os.walk(scene_data_root):
            for file in files:
                if (file.endswith('label_data.json')):
                    continue

        with open(f"{self.output_dir}/logs.json", "w") as out:
            json.dump(self.log_buffer, out)


    def publish_log_summary(self, full_set):
        total_statement_count = 0
        above_statement_count = 0
        below_statement_count = 0
        between_statement_count = 0
        near_statement_count = 0
        closest_statements_count = 0
        farthest_statements_count = 0
        color_statements_count = 0
        size_statements_count = 0
        color_references_count = 0
        size_references_count = 0

        max_statement_scene = None
        max_statement_scene_count = 0

        for scene_key in self.log_buffer:
            scene = self.log_buffer[scene_key]
            scene_total = 0

            for region_key in scene['regions']:
                region=scene['regions'][region_key]
                if "color_references" in region:
                    color_references_count+=region["color_references"]
                if "size_references" in region:
                    size_references_count+=region["size_references"]
                if "color_statements" in region:
                    color_statements_count+=region["color_statements"]
                if "size_statements" in region:
                    size_statements_count+=region["size_statements"]
                if "total_statements" in region:
                    scene_total+=region["total_statements"]
                if "above_statements" in region:
                    above_statement_count+=region["above_statements"]
                if "below_statements" in region:
                    below_statement_count+=region["below_statements"]
                if "between_statements" in region:
                    between_statement_count+=region["between_statements"]
                if "near_statements" in region:
                    near_statement_count+=region["near_statements"]
                if "closest_statements" in region:
                    closest_statements_count+=region["closest_statements"]
                if "farthest_statements" in region:
                    farthest_statements_count+=region["farthest_statements"]

            if scene_total > max_statement_scene_count:
                max_statement_scene_count = scene_total
                max_statement_scene = scene_key
            total_statement_count+=scene_total

        f = open(f"{self.output_dir}/log_summary.txt", "w")
        f.write(f"Total amount of all statements: {total_statement_count} \n"
                f"Total amount of above statements: {above_statement_count} \n"
                f"Total amount of below statements: {below_statement_count} \n"
                f"Total amount of between statements: {between_statement_count} \n"
                f"Total amount of near statements: {near_statement_count} \n"
                f"Total amount of closest statements: {closest_statements_count} \n"
                f"Total amount of farthest statements: {farthest_statements_count} \n"
                f"Total amount of color references: {color_references_count} \n"
                f"Total amount of size references: {size_references_count} \n"
                f"Total amount of color statements: {color_statements_count} \n"
                f"Total amount of size statements: {size_statements_count} \n"
                f"Total amount of unique objects: {len(full_set)} \n"
                f"Scene with most statements: {max_statement_scene} with a total of {max_statement_scene_count} statements\n")
        f.close()



if __name__ == '__main__':
    # Load the scene graph and language templates from their JSON files into dicts
    args = parse_args()
    logger = Logger(args.log_schema)

    print(logger.log_buffer)
