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
            try:
                os.remove(os.path.join(target_dir, dataset, scene, f'{scene}_grouped.json'))
                os.remove(os.path.join(target_dir, dataset, scene, f'{scene}_object_class.json'))
                os.remove(os.path.join(target_dir, dataset, scene, f'{scene}_object_data.json'))
                print(os.path.join(target_dir, dataset, scene))
            except:
                pass
