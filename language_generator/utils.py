import os

def purge_existing_language_data(scene_data_root):
    for subdir, dirs, files in os.walk(scene_data_root):
        for file in files:
            if (file.endswith('statement.json')):
                os.remove(os.path.join(subdir, file))
            elif (file.endswith('label_data.json')):
                os.remove(os.path.join(subdir, file))
