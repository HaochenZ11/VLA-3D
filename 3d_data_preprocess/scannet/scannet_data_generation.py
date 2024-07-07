from referit3d.in_out.neural_net_oriented import load_scan_related_data
from referit3d.utils import unpickle_data
from referit3d.in_out.scan_2cad import load_scan2cad_meta_data, load_has_front_meta_data
from referit3d.in_out.scan_2cad import register_scan2cad_bboxes, register_front_direction
from scipy.spatial import KDTree
import os
import os.path as osp
import numpy as np
import csv
import math
import open3d as o3d
import matplotlib
import argparse
from tqdm import tqdm
import sys
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.freespace_generation_new import generate_free_space
from utils.dominant_colors_new_lab import judge_color, generate_color_anchors
from utils.bbox_utils import calculate_bbox_hull
from utils.headers import OBJECT_HEADER, REGION_HEADER
from utils.pointcloud_utils import write_ply_file, sort_pointcloud, get_regions, get_objects

class ScannetPreprocessor:
    def __init__(
        self,
        class_mapping_file: str, # the path containing the scannet class mapping file
        id_mapping_file: str, # the path containing the nyu class and id mapping file
        scan2cad_meta_file: str, # object segmentation file
        bad_scan2cad_mappings_file: str, # bad mapping file indicating which object should be discarded
        has_front_file: str , # the file containing the front direction
        all_scans_file: str, # the file containing the information of all scans
        scans_directory: str, # the path of the scannet dataset
        output_directory: str , # the path to store the processed results
        floor_height: float, # the floor height for generating free space
        color_standard: str, # colors standars to use for domain color calculation (css21, css3, html4, css2)

        generate_freespace=False # whether generate free space or not
    ):
        self.class_mapping_file = class_mapping_file 
        self.id_mapping_file = id_mapping_file
        self.scan2cad_meta_file = scan2cad_meta_file
        self.bad_scan2cad_mappings_file = bad_scan2cad_mappings_file
        self.has_front_file = has_front_file
        self.all_scans_file = all_scans_file
        self.scans_directory = scans_directory
        self.output_directory = output_directory
        self.floor_height = floor_height
        self.anchor_colors_array, self.anchor_colors_array_hsv, self.anchor_colors_name = generate_color_anchors(color_standard)
        self.tree = KDTree(self.anchor_colors_array_hsv)
        self.generate_freespace = generate_freespace

    def create_scannet(self):
        # front direction use the heading from center to the point and the most similar to one of the face direction.

        mapping = {}
        with open(class_mapping_file) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            next(tsv_file)
            for line in tsv_file:
                if line[6] == '':
                    line[6] = 'unknown'
                mapping[line[1]] = (line[4], line[6], line[7])

        mapping_id = {}
        with open(id_mapping_file) as file:
            csv_file = csv.reader(file, delimiter=",")
            idx = 1
            for line in csv_file:
                mapping_id[line[0]] = idx
                idx += 1

        scannet, all_scans = unpickle_data(self.all_scans_file)
        print("Loaded all scans")
        scan2CAD = load_scan2cad_meta_data(self.scan2cad_meta_file)
        register_scan2cad_bboxes(all_scans, scan2CAD, self.bad_scan2cad_mappings_file)
        has_front = load_has_front_meta_data(self.has_front_file)
        register_front_direction(all_scans, scan2CAD, has_front)
        pbar = tqdm(all_scans)
        for i, scan in enumerate(pbar):
            scan_name = all_scans[i].scan_id
            pbar.set_description(f'Processing {scan_name}')
            scan_path = osp.join(self.output_directory, scan_name)
            os.chdir(scan_path)
            region_and_align_info = osp.join(self.scans_directory, scan_name, scan_name + '.txt')
            with open(region_and_align_info) as f:
                for row in f.readlines():
                    line = row.strip().split(' = ')
                    if line[0] == 'axisAlignment':
                        align_rot = line[1].split(' ')
                    elif line[0] == 'sceneType':
                        region_label = line[1]

            region_center = (np.max(all_scans[i].pc, axis=0) + np.min(all_scans[i].pc, axis=0))/2
            region_size = np.max(all_scans[i].pc, axis=0) - np.min(all_scans[i].pc, axis=0)
            region_min = np.min(all_scans[i].pc, axis=0)
            region_max = np.max(all_scans[i].pc, axis=0)

            floor_sizes = []
            object_file_name = scan_name + '_object_result.csv'
            with open(object_file_name, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(OBJECT_HEADER)
                object_id = 0
                for obj in all_scans[i].three_d_objects:
                    obj.get_axis_align_bbox()
                    if obj.instance_label:
                        raw_label = obj.instance_label
                        label = mapping[obj.instance_label][1]
                        label_id = mapping_id[label]
                        label40 = mapping[obj.instance_label][2]
                        label40_id = mapping[obj.instance_label][0]
                    else:
                        raise KeyError('No Object Lable')

                    center, size, heading = calculate_bbox_hull(obj.pc.astype(np.float64))
                    
                    if object_id == 0:
                        vertices = obj.pc
                        vertex_colors = obj.color*255
                        obj_ids = np.repeat(np.array([[object_id]]), len(obj.pc), axis = 0)
                        region_ids = np.repeat(np.array([[0]]), len(obj.pc), axis = 0)
                    else:
                        vertices = np.vstack((vertices, obj.pc))
                        vertex_colors = np.vstack((vertex_colors, obj.color*255))
                        obj_ids = np.vstack((obj_ids, np.repeat(np.array([[object_id]]), len(obj.pc), axis = 0)))
                        region_ids = np.vstack((region_ids, np.repeat(np.array([[0]]), len(obj.pc), axis = 0)))
                    
                    if obj.has_front_direction:
                        front_direction = np.dot(np.array(align_rot, dtype=float).reshape(4,4), np.concatenate((obj.front_direction, np.array([1.0]))))[0:2] - center[0:2]
                        front_heading = [math.atan2(front_direction[1], front_direction[0])]
                    else:
                        front_heading = ['_']

                    

                    colors = obj.color
                    color_3 = judge_color(colors, self.tree, self.anchor_colors_array, self.anchor_colors_name)

                    region_id = 0
                    if label40 == 'floor':
                        floor_sizes.append(center+size+[heading, region_id])
                    info = []
                    info.append(object_id)
                    object_id += 1
                    info.append(region_id)
                    info.append(raw_label)
                    info.append(label_id)
                    info.append(label40_id)
                    info.append(label)
                    info.append(label40)
                    info += center
                    info += size
                    # info += rot
                    info.append(heading)
                    info += front_heading
                    info += color_3
                    writer.writerow(info)
            
            if self.generate_freespace:
                generate_free_space(scan_name, vertices.astype(float), floor_sizes, region_ids.reshape(-1, 1).astype(np.int32), self.floor_height)


            region_file_name = scan_name + '_region_result.csv'
            with open(region_file_name, 'w', newline='') as f_region:
                region_writer = csv.writer(f_region, delimiter=',')
                region_writer.writerow(REGION_HEADER)
                # region_id is defined before
                region_heading = 0
                region_info = []
                region_info.append(region_id)
                region_info.append(region_label)
                region_info += list(region_center)
                region_info += list(region_size)
                region_info.append(region_heading)
                region_writer.writerow(region_info)

            # region_folder = osp.join(scan_path, 'regions')
            # if not os.path.exists(region_folder):
            #     os.makedirs(region_folder)
            # get_regions(vertices.astype(float), vertex_colors.astype(np.int32), region_ids.reshape(-1, 1).astype(np.int32), obj_ids.reshape(-1, 1).astype(np.int32), region_folder, scan_name)

            # object_folder = osp.join(scan_path, 'objects')
            # if not os.path.exists(object_folder):
            #     os.makedirs(object_folder)
            # get_objects(vertices.astype(float), vertex_colors.astype(np.int32), region_ids.reshape(-1, 1).astype(np.int32), obj_ids.reshape(-1, 1).astype(np.int32), object_folder, scan_name)

            vertex = np.concatenate( [
                vertices,
                vertex_colors,
                obj_ids.reshape(-1, 1),
                region_ids.reshape(-1, 1)
            ], axis=1, dtype=np.float32)

            vertex, region_indices_out, object_indices_out = sort_pointcloud(torch.from_numpy(vertex))
            # print(vertex.shape)

            torch.save(region_indices_out, f'{scan_name}_region_split.npy')
            torch.save(object_indices_out, f'{scan_name}_object_split.npy')
            write_ply_file(vertex[:, :6], f'{scan_name}_pc_result.ply')

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_freespace', action='store_true')
    parser.add_argument('--scannet_directory', default='/home/navigation/Dataset/scannet')
    parser.add_argument('--output_directory', default='/home/navigation/Dataset/VLA_Dataset')
    parser.add_argument('--floor_height', default=0.2)
    parser.add_argument('--color_standard', default='css3')

    args = parser.parse_args()

    scannet_directory = args.scannet_directory
    scannet_data = osp.join(scannet_directory, 'data')
    output_directory = args.output_directory
    scans_directory = osp.join(scannet_directory, 'scans')

    class_mapping_file = osp.join(scannet_data, 'scannetv2-labels.combined.tsv')
    id_mapping_file = osp.join(scannet_data, 'nyu_label.csv')
    scan2cad_meta_file = osp.join(scannet_data, 'object_oriented_bboxes.json')
    bad_scan2cad_mappings_file = osp.join(scannet_data, 'bad_mappings.json')
    has_front_file = osp.join(scannet_data, 'shapenet_has_front.csv')
    all_scans_file = osp.join(
        scannet_data, 
        'keep_all_points_with_global_scan_alignment',
        'keep_all_points_with_global_scan_alignment.pkl')

    print('====================Start processing scannet====================')
    ScannetPreprocessor(
        class_mapping_file,
        id_mapping_file,
        scan2cad_meta_file,
        bad_scan2cad_mappings_file,
        has_front_file,
        all_scans_file,
        scans_directory,
        output_directory,
        args.floor_height,
        args.color_standard,
        generate_freespace=args.generate_freespace
    ).create_scannet()
    print('====================Start processing scannet====================')
