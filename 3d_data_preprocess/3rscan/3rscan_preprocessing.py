import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.spatial import KDTree
from webcolors import hex_to_rgb
import argparse
import multiprocessing as mp
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.glb_to_pointcloud import (
    load_meshes, 
    subdivide_mesh, 
    sample_semantic_pointcloud_from_uv_mesh, 
    SEED, 
    DEVICE
)
from utils.bbox_utils import calculate_bbox, calculate_bbox_hull, calculate_axis_aligned_bbox
from utils.pointcloud_utils import sort_pointcloud, write_ply_file
from utils.freespace_generation_new import generate_free_space
from utils.dominant_colors_new_lab import judge_color, generate_color_anchors
from utils.headers import REGION_HEADER, OBJECT_HEADER

class ThreeRScanPreprocessor:
    def __init__(
        self,
        scan_directory: str,
        category_mappings_path: str,
        output_directory: str,
        num_pointcloud_samples=1000000,
        sampling_density=None,
        num_region_samples=70000,
        filter_objs_less_than=10,
        seed=SEED,
        device=DEVICE
        ):

        # Color Tree

        self.floor_height = 0.1
        self.color_standard = 'css3'
        self.anchor_colors_array, self.anchor_colors_array_hsv, self.anchor_colors_name = generate_color_anchors(self.color_standard)
        self.tree = KDTree(self.anchor_colors_array_hsv)

        # Variable Initialization

        self.scan_directory = scan_directory
        self.category_mappings_path = category_mappings_path
        self.output_directory = output_directory
        self.num_pointcloud_samples = num_pointcloud_samples
        self.num_region_samples = num_region_samples
        self.filter_objs_less_than = filter_objs_less_than
        self.sampling_density = sampling_density
        self.seed = seed
        self.device = device 

        with open('skipped_scans.json', 'r') as f:
            skipped_scans = json.load(f)
        self.meshes = [mesh for mesh in os.listdir(self.scan_directory) if mesh not in skipped_scans]

        # with open('3RScan/3RScan.json', 'r') as f:
        #     rscan_dict = json.load(f)

        # transform_dict = {}
        # for base_scan in rscan_dict:
        #     transform_dict[base_scan['reference']] = []
        #     for linked_scan in base_scan['scans']:
        #         if 'transform' in linked_scan:
        #             transform_dict[linked_scan['reference']] = linked_scan['transform']
        #         else:
        #             transform_dict[linked_scan['reference']] = []
        
        self.category_mappings = pd.read_csv(category_mappings_path, index_col=0)
        self.scene_transformations_path = Path(scan_directory) / '3RScan.json'





    def find_points_and_calculate_bbox_and_top3_color3(
            self,
            object_points: torch.Tensor
    ):

        # point_filter = points[:, 6] == object_id

        # object_points = points[point_filter]

        if len(object_points) < self.filter_objs_less_than:
            return None, None, None, None

        center, size, heading = calculate_bbox_hull(object_points[:, :3].cpu().numpy())

        colors = object_points[:, 3:6]/255
        color_3 = judge_color(colors.cpu().numpy(), self.tree, self.anchor_colors_array, self.anchor_colors_name)

        return center, size, heading, color_3
    
    def create_object_csv(self, points, scan_name):

        points = points[torch.argsort(points[:, 6])]
        object_indices, object_splits = torch.unique_consecutive(points[:, 6], return_counts=True)
        object_points_list = torch.split(points, object_splits.tolist())
        object_indices = object_indices.int()

        object_csv_rows = []
        object_points_out = []

        object_id = 0
        for object_points in tqdm(object_points_list, desc='Objects: '):
        # for i in range(self.nobjects):
            object_line = []

            object_line.extend([
                object_id,
                0
            ])

            (
                center, 
                size, 
                heading, 
                color_3, 
            ) = self.find_points_and_calculate_bbox_and_top3_color3(object_points)
            
            if center is None or (size < np.array([1e-5, 1e-5, 1e-5])).any():
                continue

            assert (object_points[:, 6:] == object_points[0, 6:]).all()

            global_id = int(object_points[0, 7])

            try:
                category_mappings_row = self.category_mappings.loc[global_id]
            except KeyError as e:
                continue

            raw_label = category_mappings_row['Label']

            nyu_id, nyu_label = (int(category_mappings_row.nyuId), category_mappings_row.nyuClass) \
            if not pd.isnull(category_mappings_row.nyuId) else (20, 'unknown')

            nyu40_id, nyu40_label = category_mappings_row.nyu40id, category_mappings_row.nyu40class

            object_line.extend([raw_label, nyu_id, nyu40_id, nyu_label, nyu40_label])

            object_line += list(center)
            object_line += list(size)
            object_line.append(heading)
            object_line.append('_')

            object_line += color_3

            object_csv_rows.append(object_line)

            object_points[:, 6] = object_id

            object_id += 1

            object_points_out.append(object_points)
        
        object_file_name = Path(self.output_directory) / scan_name /  f'{scan_name}_object_result.csv'

        object_out_df = pd.DataFrame(object_csv_rows, columns=OBJECT_HEADER)
        object_out_df['region_id'] -= object_out_df['region_id'].min()
        object_out_df.set_index('object_id', inplace=True)
        object_out_df.to_csv(object_file_name)

        return torch.vstack(object_points_out)


    def create_region_csv(self, points, scan_name):
                
        region_list = []

        region_label = 'Room'

        region_center, region_size = calculate_axis_aligned_bbox(points[:, :3])

        region_info = [0, region_label]
        region_info += list(region_center)
        region_info += list(region_size)
        region_info.append(0)

        region_list.append(region_info)

        region_out_df = pd.DataFrame(region_list, columns=REGION_HEADER)
        region_out_df.set_index('region_id', inplace=True)
        region_file_name = Path(self.output_directory) / scan_name /  f'{scan_name}_region_result.csv'
        region_out_df.to_csv(region_file_name)


    def create_3rscan(self):

        pbar = tqdm(self.meshes)
        for scan_name in pbar:
            pbar.set_description(f'Processing {scan_name}')

            color_mesh_path = Path(self.scan_directory) / scan_name / f'mesh.refined.v2.obj'
            semantic_mesh_path = Path(self.scan_directory) / scan_name / f'labels.instances.annotated.v2.ply'
            object_json_path = Path(self.scan_directory) / scan_name / f'semseg.v2.json'
            output_path = Path(self.output_directory) / scan_name


            if not os.path.exists(Path(self.output_directory) / scan_name ):
                os.makedirs(Path(self.output_directory) / scan_name )

            self.mesh_objects, self.semantic_mesh_objects = load_meshes(
                str(color_mesh_path), 
                str(semantic_mesh_path),
                ['objectId', 'globalId', 'red', 'green', 'blue'])

            # self.mesh_objects = subdivide_mesh(self.mesh_objects, 0.1)
            
            points, point_triangle_indices = sample_semantic_pointcloud_from_uv_mesh(
                self.mesh_objects,
                self.semantic_mesh_objects,
                n=self.num_pointcloud_samples,
                sampling_density=self.sampling_density,
                seed=self.seed,
            )

            filtered_points = self.create_object_csv(points, scan_name)
            self.create_region_csv(filtered_points, scan_name)

            vertex = torch.cat([
                filtered_points[:, :7],
                torch.zeros((filtered_points.shape[0], 1), device=DEVICE)
            ], dim=1)

            vertex, region_indices_out, object_indices_out = sort_pointcloud(vertex)

            torch.save(region_indices_out, output_path / f'{scan_name}_region_split.npy')
            torch.save(object_indices_out, output_path / f'{scan_name}_object_split.npy')
            write_ply_file(vertex[:, :6], output_path / f'{scan_name}_pc_result.ply')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--scan_directory', 
        default='/media/navigation/easystore/Original_dataset/3RScan/scans')
    parser.add_argument('--category_mappings_path', default='3rscan_full_mapping.csv')
    parser.add_argument('--output_directory',
        default='/home/navigation/Dataset/VLA_Dataset_3rscan')
    parser.add_argument('--num_points', type=int, default=5000000)
    parser.add_argument('--sampling_density', type=float, default=1e-4)
    parser.add_argument('--num_region_points', type=int, default=500000)
    parser.add_argument('--filter_objs_less_than', type=int, default=10)
    # parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()

    print(f'Device: {DEVICE}')

    ThreeRScanPreprocessor(
        args.scan_directory,
        args.category_mappings_path,
        args.output_directory,
        args.num_points,
        args.sampling_density,
        args.num_region_points,
        args.filter_objs_less_than
        # args.device
    ).create_3rscan()
