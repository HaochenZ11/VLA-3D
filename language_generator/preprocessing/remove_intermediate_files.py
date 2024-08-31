import json
from pathlib import Path
import os
import shutil

if __name__ == '__main__':


    configs_path = Path(__file__).parents[1].resolve() / 'configs' / 'language_generation_configs.json'
    print(configs_path)
    with open(configs_path) as c:
        generation_configs = json.load(c)

    target_dir = generation_configs["scene_data_root"]

    for dataset in os.listdir(target_dir):
        if not os.path.isdir(os.path.join(target_dir, dataset)):
            continue
        print(dataset)
        for scene in os.listdir(os.path.join(target_dir, dataset)):
            for suffix in ['grouped', 'grouped_by_anchor', 'grouped_by_target', 'object_class', 'object_data', 'objects']:
                try:
                    os.remove(os.path.join(target_dir, dataset, scene, f'{scene}_{suffix}.json'))
                    print(os.path.join(target_dir, dataset, scene, f'{scene}_{suffix}.json'))
                except:
                    pass
