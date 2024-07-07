import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys
import scipy.spatial
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy.spatial import KDTree
from webcolors import hex_to_rgb
import argparse
import multiprocessing as mp

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

class HM3DPreprocessor:
    def __init__(
        self,
        scan_name: str,
        color_mesh_directory: str,
        semantic_mesh_directory: str,
        category_mappings_path: str,
        raw_category_mappings_path: str,
        region_categories_path: str,
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

        self.scan_name = scan_name
        self.color_mesh_directory = color_mesh_directory
        self.semantic_mesh_directory = semantic_mesh_directory
        self.category_mappings_path = category_mappings_path
        self.raw_category_mappings_path = raw_category_mappings_path
        self.region_categories_path = region_categories_path
        self.output_directory = output_directory
        self.num_pointcloud_samples = num_pointcloud_samples
        self.num_region_samples = num_region_samples
        self.filter_objs_less_than = filter_objs_less_than
        self.seed = seed
        self.device=device
        
        self.scan_prefix, self.scan_suffix = self.scan_name.split('-')

        self.color_mesh_path = Path(color_mesh_directory) / self.scan_name / f'{self.scan_suffix}.glb'
        self.semantic_mesh_path = Path(semantic_mesh_directory) / self.scan_name / f'{self.scan_suffix}.semantic.glb'
        self.semantic_config_path = Path(semantic_mesh_directory) / self.scan_name / f'{self.scan_suffix}.semantic.txt'

        self.output_path = Path(output_directory) / self.scan_name

        if not os.path.exists(Path(output_directory) / self.scan_name ):
            os.makedirs(Path(output_directory) / self.scan_name )

        self.mesh_objects, self.semantic_mesh_objects = load_meshes(str(self.color_mesh_path), str(self.semantic_mesh_path))

        # self.mesh_objects = subdivide_mesh(self.mesh_objects, 0.1)
        
        self.points, self.point_triangle_indices = sample_semantic_pointcloud_from_uv_mesh(
            self.mesh_objects,
            self.semantic_mesh_objects,
            n=num_pointcloud_samples,
            sampling_density=sampling_density,
            seed=seed,
        )

        self.semantic_colors = self.points[:, 6:9].int()

        self.point_region_ids = -torch.ones_like(self.points[:, 0], dtype=torch.int32)

        self.point_object_ids = -torch.ones_like(self.points[:, 0], dtype=torch.int32)

        self.semantic_config = pd.read_csv(
            self.semantic_config_path, 
            header=0, 
            names=['object_id', 'semantic_color', 'object_label', 'region_id'],
            index_col=0)
        self.semantic_config.index -= 1

        self.category_mappings = pd.read_csv(category_mappings_path, index_col=1)

        self.raw_mappings = pd.read_csv(raw_category_mappings_path, sep='\t', index_col=0)

        self.region_categories = pd.read_csv(region_categories_path)

        self.nobjects = len(self.semantic_config)


        self.region_remapping_dict = {region_id: i for i, region_id in enumerate(list(set(self.semantic_config.region_id)))}
        self.semantic_config.region_id = self.semantic_config.region_id.map(self.region_remapping_dict)


    def find_object_points(
            self,
            semantic_colors: torch.Tensor,
            semantic_row: pd.Series,      
    ):
        semantic_color = torch.tensor(
            [elt for elt in hex_to_rgb(f'#{semantic_row.semantic_color}')],
            device=torch.device(self.device))

        point_filter = torch.all(semantic_colors == semantic_color, dim=1)

        return point_filter

    def find_points_and_calculate_bbox_and_top3_color3(
            self,
            semantic_colors: torch.Tensor,
            semantic_row: pd.Series,
            points: torch.Tensor
    ):

        point_filter = self.find_object_points(semantic_colors, semantic_row)

        object_points = points[point_filter]

        if len(object_points) < self.filter_objs_less_than:
            return None, None, None, None, None, None

        try:
            center, size, heading = calculate_bbox_hull(object_points[:, :3].cpu().numpy())
        except scipy.spatial._qhull.QhullError as e:
            print(e)
            print(object_points)
            return None, None, None, None, None, None

        colors = object_points[:, 3:6]/255
        color_3 = judge_color(colors.cpu().numpy(), self.tree, self.anchor_colors_array, self.anchor_colors_name)

        return center, size, heading, color_3, point_filter, object_points


    def find_points_and_calculate_bbox_region(
            self,
            target_region_id: int, 
            point_region_ids: torch.Tensor, 
            points: torch.Tensor):

        point_filter = point_region_ids == target_region_id

        region_points = points[point_filter]

        center, size = calculate_axis_aligned_bbox(region_points[:, :3])
        
        return center, size, region_points, point_filter
    
    def create_object_csv(self):

        object_out_list = []

        object_id = 0
        for i in tqdm(range(self.nobjects), desc='Objects: '):
        # for i in range(self.nobjects):
            object_line = []

            semantic_row = self.semantic_config.loc[i]

            object_line.extend([
                object_id,
                semantic_row.region_id
            ])

            (
                center, 
                size, 
                heading, 
                color_3, 
                point_filter,
                object_points
            ) = self.find_points_and_calculate_bbox_and_top3_color3(
                self.semantic_colors,
                semantic_row,
                self.points)
            
            if point_filter is None or (size < np.array([1e-5, 1e-5, 1e-5])).any():
                continue

            self.point_region_ids[point_filter] = semantic_row.region_id

            if (self.point_object_ids[point_filter] != -1).any():
                ids, counts = torch.unique(self.point_object_ids[point_filter], return_counts=True)
                print(ids, counts)
                print("Sum: ", counts.sum())
                print("Object ID:", object_id)
                continue

            self.point_object_ids[point_filter] = object_id


            try:
                category_mappings_row = self.category_mappings.loc[semantic_row.object_label]
            except KeyError as e:
                processed_label = self.raw_mappings.loc[semantic_row.object_label].category
                category_mappings_row = self.category_mappings.loc[processed_label]

            raw_label = semantic_row.object_label

            nyu_id, nyu_label = (int(category_mappings_row.nyuId), category_mappings_row.nyuClass) \
            if not pd.isnull(category_mappings_row.nyuId) else (20, 'unknown')

            nyu40_id, nyu40_label = category_mappings_row.nyu40id, category_mappings_row.nyu40class

            object_line.extend([raw_label, nyu_id, nyu40_id, nyu_label, nyu40_label])

            object_line += list(center)
            object_line += list(size)
            object_line.append(heading)
            object_line.append('_')

            object_line += color_3

            object_out_list.append(object_line)

            object_id += 1
        
        object_file_name = Path(self.output_directory) / self.scan_name /  f'{self.scan_name}_object_result.csv'

        object_out_df = pd.DataFrame(object_out_list, columns=OBJECT_HEADER)
        object_out_df.set_index('object_id', inplace=True)
        object_out_df.to_csv(object_file_name)

    def resample_region(self, region_points_filter: torch.Tensor, region_id: int):
        filtered_triangle_indices = self.point_triangle_indices[region_points_filter].unique()

        region_points_resampled, _ = sample_semantic_pointcloud_from_uv_mesh(
            self.mesh_objects,
            filtered_triangle_indices,
            n=self.num_region_samples,
            seed=self.seed
        )

        region_objects_df = self.semantic_config.loc[self.semantic_config['region_id']==region_id]
        region_point_object_ids = -torch.ones(self.num_region_samples)

        for i, object_row in region_objects_df.iterrows():
            object_point_filter = self.find_object_points(region_points_resampled[:, 6:9], object_row)
            region_point_object_ids[object_point_filter] = i
        
        return region_points_resampled, region_point_object_ids

    def create_region_csv(self):
                
        region_points_lengths = []

        region_list = []

        for original_region_id, remapped_id in tqdm(self.region_remapping_dict.items()):

            region_label = self.region_categories.loc[(self.region_categories['Scene Name'] == self.scan_name) &\
                            (self.region_categories['Region #'] == original_region_id)].iloc[0, -1].strip()

            # Some region labels contain ties based on the authors' heuristics. I considered these unknown.
            region_label = 'Unknown room' if region_label.startswith('Tie:') else region_label

            (
                region_center, 
                region_size, 
                region_points,
                region_points_filter
                ) = self.find_points_and_calculate_bbox_region(
                    remapped_id, self.point_region_ids, self.points)

            region_info = []
            region_info.append(remapped_id)
            region_info.append(region_label)
            region_info += list(region_center)
            region_info += list(region_size)
            region_info.append(0)

            region_list.append(region_info)

            region_points_lengths.append(len(region_points))
            
        
        region_out_df = pd.DataFrame(region_list, columns=REGION_HEADER)
        region_out_df.set_index('region_id', inplace=True)
        region_file_name = Path(self.output_directory) / self.scan_name /  f'{self.scan_name}_region_result.csv'
        region_out_df.to_csv(region_file_name)

        region_points_lengths = np.array(region_points_lengths)

        print(region_points_lengths.min(), region_points_lengths.max())


    def create_hm3d(self):

        self.create_object_csv()
        self.create_region_csv()

        vertex = torch.cat([
            self.points[:, :6],
            self.point_object_ids[:, None],
            self.point_region_ids[:, None]
        ], dim=1)

        vertex, region_indices_out, object_indices_out = sort_pointcloud(vertex)

        torch.save(region_indices_out, self.output_path / f'{self.scan_name}_region_split.npy')
        torch.save(object_indices_out, self.output_path / f'{self.scan_name}_object_split.npy')
        write_ply_file(vertex[:, :6], self.output_path / f'{self.scan_name}_pc_result.ply')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--color_mesh_directory', 
        default='/media/navigation/easystore1/Dataset/hm3d/hm3d-train-glb-v0.2')
    parser.add_argument('--semantic_mesh_directory', 
        default='/media/navigation/easystore1/Dataset/hm3d/hm3d-train-semantic-annots-v0.2')
    parser.add_argument('--category_mappings_path', 
        default='category_mappings/hm3dsem_full_mappings.csv')
    parser.add_argument('--raw_category_mappings_path', 
        default='category_mappings/hm3dsem_category_mappings.tsv')
    parser.add_argument('--region_categories_path', 
        default='orig_hm3d_statistics/Per_Scene_Region_Weighted_Votes.csv')
    parser.add_argument('--output_directory',
        default='/home/navigation/Dataset/VLA_Dataset')
    parser.add_argument('--num_points', type=int, default=5000000)
    parser.add_argument('--sampling_density', type=float, default=2e-4)
    parser.add_argument('--num_region_points', type=int, default=500000)
    parser.add_argument('--filter_objs_less_than', type=int, default=10)
    # parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()
    

    meshes = os.listdir(args.semantic_mesh_directory)
    skipped_scans = ['00023-zepmXAdrpjR', '00476-NtnvZSMK3en', '00546-nS8T59Aw3sf', # Mismatching semantic and color meshes
        '00172-bB6nKqfsb1z', '00643-ggNAcMh8JPT']
    meshes = [mesh for mesh in meshes if mesh not in skipped_scans]
    # meshes = ['00033-oPj9qMxrDEa']

    # counter = mp.Value('i', 0)

    def process_mesh(scan_name):
        # with counter.get_lock():
        #     counter.value += 1
        # print(f'Processing {scan_name} ({counter.value}/{len(meshes)})')
        HM3DPreprocessor(
            scan_name,
            args.color_mesh_directory,
            args.semantic_mesh_directory,
            args.category_mappings_path,
            args.raw_category_mappings_path,
            args.region_categories_path,
            args.output_directory,
            args.num_points,
            args.sampling_density,
            args.num_region_points,
            args.filter_objs_less_than
            # args.device
        ).create_hm3d()

    # with mp.Pool(mp.cpu_count()) as p:
        # p.map(process_mesh, meshes)
    pbar = tqdm(meshes)
    for i, mesh in enumerate(pbar):
        # print(f'Processing {scan_name} ({counter.value}/{len(meshes)})')
        pbar.set_description(f'Processing {mesh} ')
        process_mesh(mesh)