import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def get_scenes(args):
    scannet_scenes = [x for x in os.listdir(args.data_path) if x.startswith('scene')]
    hm3d_scenes = [x for x in os.listdir(args.data_path) if "-" in x and os.path.isdir(os.path.join(args.data_path, x))]
    unity_names = ["arabic_room", "chinese_room", "home_building_1", "home_building_2", "home_building_3", "hotel_room_1", 
                    "hotel_room_2", "hotel_room_3", "japanese_room", "livingroom_1", "livingroom_2", "livingroom_3", "livingroom_4", 
                    "loft", "office_1", "office_2", "office_3", "studio"]
    unity_scenes = [x for x in os.listdir(args.data_path) if x in unity_names]
    matterport_scenes = [x for x in os.listdir(args.data_path) if x not in scannet_scenes and x not in hm3d_scenes and x not in unity_scenes 
                         and os.path.isdir(os.path.join(args.data_path, x))]
    arkit_scenes = [x for x in os.listdir(args.data_path_arkit) if os.path.isdir(os.path.join(args.data_path_arkit, x))]

    return {
        "scannet":scannet_scenes, 
        "hm3d":hm3d_scenes, 
        "unity":unity_scenes, 
        "matterport":matterport_scenes,
        "arkit": arkit_scenes
        }
    # return {"unity":unity_scenes}

def get_counts(args, dataset, scene_list):
    '''
    Save metadata in a json file
    '''

    relations = [
        'above',
        'below',
        'closest',
        'farthest',
        'between',
        'near',
        'in',
        'on',
        'hanging_on'
    ]

    scene_totals = {
        scene: {
            'Number of Regions': 0,
            'Number of Objects': 0,
            'Number of Relations': {
                relation: 0 for relation in relations
            },
            'Number of Statements': {
                relation: 0 for relation in relations
            },
            'Regions': {

            }
        } for scene in scene_list
    }

    for scene in tqdm(scene_list, desc=dataset):
        scene_path = os.path.join(args.data_path_arkit if dataset=='arkit' else args.data_path, scene)

        if os.path.isdir(scene_path + '/'):

            object_file = os.path.join(scene_path, scene + '_object_result.csv')
            object_df = pd.read_csv(object_file)

            relation_json = os.path.join(scene_path, f'{scene}.json')
            with open(relation_json, 'r') as f:
                relation_dict = json.load(f)
            


            scene_totals[scene]['Number of Objects'] = int(object_df['object_id'].count())
            scene_totals[scene]['Number of Regions'] = int(object_df['region_id'].nunique())

            if scene_totals[scene]['Number of Regions'] == 0:
                print(scene)

            object_df = object_df.drop(object_df[object_df['region_id'] == -1].index)
            region_counts = object_df['region_id'].value_counts()

            for region_id, num_objects in region_counts.items():
                scene_totals[scene]['Regions'][region_id] = {
                    'Number of Objects': num_objects,
                    'Number of Relations': {
                        relation: 0 for relation in relations
                    },
                    'Number of Statements': {
                        relation: 0 for relation in relations
                    },
                }


            for relation in relations:
                for region in scene_totals[scene]['Regions'].keys():
                    for anchor_obj_relations in relation_dict['regions'][str(region)]['relationships'][relation].values():
                        scene_totals[scene]['Regions'][region]['Number of Relations'][relation] += len(anchor_obj_relations)
                        scene_totals[scene]['Number of Relations'][relation] += len(anchor_obj_relations)

    print([[region['Number of Relations'][relation] for relation in relations]
                    for region in scene_totals[scene]['Regions'].values()])

    scene_stats = {
        'Total Number of Scenes': len(scene_list),

        'Objects': {
            'Total': int(np.sum([scene_totals[scene]['Number of Objects'] for scene in scene_list])),
            'Mean Per Scene': np.mean([scene_totals[scene]['Number of Objects'] for scene in scene_list]),
            'Std Per Scene': np.std([scene_totals[scene]['Number of Objects'] for scene in scene_list]),
        },

        'Relations': {
            'Total': int(np.sum([
                [scene_totals[scene]['Number of Relations'][relation] for relation in relations]
                for scene in scene_list])), 
            'Mean Per Scene': np.mean([
                [scene_totals[scene]['Number of Relations'][relation] for relation in relations]
                for scene in scene_list]), 
            'Std Per Scene': np.std([
                [scene_totals[scene]['Number of Relations'][relation] for relation in relations]
                for scene in scene_list]), 
            'Types': {
                relation: {
                    'Total': int(np.sum([scene_totals[scene]['Number of Relations'][relation] for scene in scene_list])),
                    'Mean Per Scene': np.mean([scene_totals[scene]['Number of Relations'][relation] for scene in scene_list]),
                    'Std Per Scene': np.std([scene_totals[scene]['Number of Relations'][relation] for scene in scene_list]),
                } for relation in relations
            }
        },

        'Regions': {
            'Mean Per Scene': np.mean([scene_totals[scene]['Number of Regions'] for scene in scene_list]),
            'Std Per Scene': np.std([scene_totals[scene]['Number of Regions'] for scene in scene_list]),
            'Objects': {
                'Mean Per Region': np.mean(np.concatenate([
                    [region['Number of Objects'] for region in scene_totals[scene]['Regions'].values()]
                    for scene in scene_list])),
                'Std Per Region': np.std(np.concatenate([
                    [region['Number of Objects'] for region in scene_totals[scene]['Regions'].values()]
                    for scene in scene_list]))
            },
            'Relations': {
                'Mean Per Region': np.mean(np.concatenate([
                    np.concatenate([[region['Number of Relations'][relation] for relation in relations]
                    for region in scene_totals[scene]['Regions'].values()])
                    for scene in scene_list])), 
                'Std Per Region': np.std(np.concatenate([
                    np.concatenate([[region['Number of Relations'][relation] for relation in relations]
                    for region in scene_totals[scene]['Regions'].values()])
                    for scene in scene_list])), 
                'Types': {
                    relation: {
                        'Mean Per Region': np.mean(np.concatenate([
                            [region['Number of Relations'][relation] for region in scene_totals[scene]['Regions'].values()] 
                            for scene in scene_list])),
                        'Std Per Region': np.std(np.concatenate([
                            [region['Number of Relations'][relation] for region in scene_totals[scene]['Regions'].values()] 
                            for scene in scene_list])),
                    } for relation in relations
                }
            }
        }
    }

    #plot_chart(regions, obj_counts, dataset, num_regions)
    # plot_hist(regions, obj_counts, dataset, num_regions)

    print(scene_stats)

    return scene_totals, scene_stats


def plot_chart(x_list, y_list, dataset, num_regions):

    fig, ax = plt.subplots()


    ax.bar(x_list, y_list)

    plt.xticks([])
    plt.ylim(0, 400)
    ax.set_ylabel('num objects')
    ax.set_xlabel('regions (' + str(num_regions) + ' total)') 
    ax.set_title('Number of objects per region in ' + dataset)

    plt.show()
    plt.savefig(dataset+'_obj_counts.png')


def plot_hist(x_list, y_list, dataset, num_regions):

    counts, bins = np.histogram(y_list, bins=40, range=(0, 400))
    print(counts)
    print(bins)
    plt.bar(bins[:-1], counts, color='C0')

    #plt.xticks([])
    plt.ylim(0, 750)
    plt.ylabel('number of regions')
    plt.xlabel('number of objects') 
    plt.title('Histogram for ' + dataset)

    plt.show()
    plt.savefig(dataset+'_obj_hist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='VLA_Dataset')
    parser.add_argument('--data_path_arkit', default='VLA_Dataset_more')
    parser.add_argument('--class_file', default='NYU_Object_Classes.csv')
    parser.add_argument('--pad_idx', default=900)

    args = parser.parse_args()

    all_scenes = get_scenes(args)

    scene_totals = {}
    scene_stats = {}
    for dataset, scene_list in all_scenes.items():
        scene_totals_dataset, scene_stats_dataset = get_counts(args, dataset, scene_list)
        scene_totals[dataset] = scene_totals_dataset
        scene_stats[dataset] = scene_stats_dataset
    
    with open('scene_totals.json', 'w') as f:
        json.dump(scene_totals, f, indent=4)

    with open('scene_stats.json', 'w') as f:
        json.dump(scene_stats, f, indent=4)
