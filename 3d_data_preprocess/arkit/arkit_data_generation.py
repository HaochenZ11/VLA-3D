from time import perf_counter
from scipy.spatial import KDTree
import os
import os.path as osp
import numpy as np
import csv
import json
import math
import open3d as o3d
import matplotlib
import torch
from plyfile import PlyData
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.freespace_generation_new import generate_free_space
from utils.dominant_colors_new_lab import judge_color, generate_color_anchors
from utils.bbox_utils import calculate_bbox_hull
from utils.pointcloud_utils import get_regions, get_objects, save_pointcloud, sort_pointcloud, write_ply_file
from utils.headers import OBJECT_HEADER, REGION_HEADER

import warnings
warnings.simplefilter("error")

class ARKitPreprocessor:
    def __init__(
        self,
        input_folder: str, # the path containing scan files
        mapping_file: str, # the path containing the matterport class mapping file
        output_folder: str, # the path to store the processed results
        floor_height: float, # the floor height for generating free space
        color_standard: str, # colors standars to use for domain color calculation (css21, css3, html4, css2)
        generate_freespace=False,
        skipped_scenes = [],
        device='cuda'
    ):
        self.input_folder = input_folder
        self.mapping_file = mapping_file
        self.output_folder = output_folder
        self.floor_height = floor_height
        self.anchor_colors_array, self.anchor_colors_array_hsv, self.anchor_colors_name = generate_color_anchors(color_standard)
        self.tree = KDTree(self.anchor_colors_array_hsv)
        self.skipped_scenes = skipped_scenes
        self.device = 'cuda' if (device=='cuda' and torch.cuda.is_available()) else 'cpu'

        self.generate_freespace = generate_freespace

    def inside_test(self, points, cube3d):
        """
        cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
        points = array of points with shape (N, 3).

        Returns the indices of the points array which are inside the cube3d
        """
        b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

        dir1 = (t1-b1)
        size1 = torch.norm(dir1)
        dir1 = dir1 / size1

        dir2 = (b2-b1)
        size2 = torch.norm(dir2)
        dir2 = dir2 / size2

        dir3 = (b4-b1)
        size3 = torch.norm(dir3)
        dir3 = dir3 / size3

        cube3d_center = (b1 + t3)/2.0

        dir_vec = points - cube3d_center

        res1 = torch.abs(dir_vec @ dir1) <= size1/2
        res2 = torch.abs(dir_vec @ dir2) <= size2/2
        res3 = torch.abs(dir_vec @ dir3) <= size3/2

        return res1 & res2 & res3

    def get_bbox(self, center, size, R): # function for calculating the corner points of a bbox based on its center, size and a 3x3 rotation matrix

        h = float(size[2])
        w = float(size[1])
        l = float(size[0])
        # heading_angle = -heading_angle - np.pi / 2

        # center[2] = center[2] + h / 2
        # R = rotz(1*heading_angle)
        l = l/2
        w = w/2
        h = h/2
        x_corners = [-l,l,l,-l,-l,l,l,-l]
        y_corners = [w,w,-w,-w,w,w,-w,-w]
        z_corners = [-h,-h,-h,-h,h,h,h,h]
        corners_3d = np.dot(np.transpose(R), np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] += center[0]
        corners_3d[1,:] += center[1]
        corners_3d[2,:] += center[2]
        return torch.from_numpy(corners_3d).T.to(self.device)


    def crop_pc(self, xyz, bbox_center, bbox_length, bbox_rotation): # based on bbox information, crop the points inside it
        cube3d = self.get_bbox(bbox_center, bbox_length, bbox_rotation)
        index = self.inside_test(xyz, cube3d)

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for _ in range(len(bbox_lines))]  # red


        bbox = o3d.geometry.LineSet()
        bbox.lines  = o3d.utility.Vector2iVector(bbox_lines)
        bbox.colors = o3d.utility.Vector3dVector(colors)
        bbox.points = o3d.utility.Vector3dVector(cube3d.cpu().numpy())
        return index, bbox
            

    def create_arkit(self):
        scan_names = [scan for scan in os.listdir(self.input_folder) if scan not in self.skipped_scenes]
        cat_mapping = {}
        with open(self.mapping_file, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for object in reader:
                cat_mapping[object[0]] = [object[1], object[2], object[3], object[4]]
        for scan_name in tqdm(scan_names):
            json_data_name = osp.join(self.input_folder, scan_name, scan_name + "_3dod_annotation.json")
            print(scan_name)
            with open(json_data_name) as f:
                json_data = json.load(f)
            
            output_scan_folder = os.path.join(self.output_folder, scan_name)
            if not os.path.exists(output_scan_folder):
                os.makedirs(output_scan_folder)
            os.chdir(output_scan_folder)
            region_label = scan_name
            pc_path = osp.join(self.input_folder, scan_name, scan_name + "_3dod_mesh.ply")
            pc = PlyData.read(pc_path)
            r = np.asarray(pc.elements[0].data['red'])
            g = np.asarray(pc.elements[0].data['green'])
            b = np.asarray(pc.elements[0].data['blue'])
            a = np.asarray(pc.elements[0].data['alpha'])
            x = np.asarray(pc.elements[0].data['x'])
            y = np.asarray(pc.elements[0].data['y'])
            z = np.asarray(pc.elements[0].data['z'])
            xyz = torch.from_numpy(np.vstack((x,y,z)).transpose()).to(self.device)
            region_id = 0
            # region_ids = np.repeat(np.array([[region_id]]), len(xyz), axis = 0)
            unlabeled_obj_filter = torch.ones(xyz.shape[0], dtype=bool)
            rgba = torch.from_numpy(np.vstack((r,g,b,a)).transpose()).to(self.device)
            obj_pcs, obj_rgbs, obj_ids = [], [], []

            region_center = (torch.max(xyz, dim=0)[0] + torch.min(xyz, dim=0)[0])/2
            region_size = torch.max(xyz, dim=0)[0] - torch.min(xyz, dim=0)[0]

            object_file_name = scan_name + '_object_result.csv'
            with open(object_file_name, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(OBJECT_HEADER)
                object_id = -1
                for obj in json_data['data']:
                    raw_label = obj['label']
                    object_id += 1
                    bbox = obj['segments']['obbAligned']
                    bbox_center = np.array(bbox['centroid'])
                    bbox_length = np.array(bbox['axesLengths'])
                    bbox_rotation = np.array(bbox['normalizedAxes']).reshape(3,3)
                    if scan_name == '45663206':
                        if object_id == 17:
                            bbox_length[2] += 0.05
                            bbox_center[2] -= 0.025
                    elif scan_name == '47430213':
                        if object_id == 17:
                            bbox_length[2] += 0.05
                            bbox_center[2] -= 0.025
                    index, bbox_to_draw = self.crop_pc(xyz, bbox_center, bbox_length, bbox_rotation)
                    # draw.append(bbox_to_draw)
                    object_pc = xyz[index, :]
                    # obj_vertex = np.vstack((obj_vertex, object_pc))
                    center, size, heading = calculate_bbox_hull(object_pc.to(torch.float64).cpu().numpy())
                    
                    front_heading = ['_']

                    label_id = cat_mapping[raw_label][0]
                    label40_id = cat_mapping[raw_label][1]
                    label = cat_mapping[raw_label][2]
                    label40 = cat_mapping[raw_label][3]

                    object_colors = rgba[index, 0:3]/255
                    # obj_colors = np.vstack((obj_colors, object_colors))
                    color_3 = judge_color(object_colors.cpu().numpy(), self.tree, self.anchor_colors_array, self.anchor_colors_name)

                    unlabeled_obj_filter[index] = False
                    obj_ids.append(torch.ones(object_pc.shape[0], device=self.device) * object_id)
                    obj_pcs.append(object_pc)
                    obj_rgbs.append(rgba[index, 0:3])

                    info = []
                    info.append(object_id)
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

            region_file_name = scan_name + '_region_result.csv'
            with open(region_file_name, 'w', newline='') as f_region:
                region_writer = csv.writer(f_region, delimiter=',')
                region_writer.writerow(REGION_HEADER)
                # region_id is defined before
                region_heading = 0
                region_info = []
                region_info.append(region_id)
                region_info.append(region_label)
                region_info += region_center.cpu().tolist()
                region_info += region_size.cpu().tolist()
                region_info.append(region_heading)
                region_writer.writerow(region_info)

            unlabeled_obj_pc = xyz[unlabeled_obj_filter]
            unlabeled_obj_color = rgba[unlabeled_obj_filter, :3]
            obj_ids.append(torch.ones(unlabeled_obj_pc.shape[0], device=self.device) * -1)
            obj_pcs.append(unlabeled_obj_pc)
            obj_rgbs.append(unlabeled_obj_color)

            obj_ids = torch.concatenate(obj_ids)
            obj_pcs = torch.vstack(obj_pcs)
            obj_rgbs = torch.vstack(obj_rgbs)
            region_ids = torch.zeros_like(obj_ids)


            # region_folder = osp.join(output_scan_folder, 'regions')
            # if not os.path.exists(region_folder):
            #     os.makedirs(region_folder)
            # get_regions(xyz.astype(float), rgba[:, 0:3].astype(np.int32), region_ids.astype(np.int32), obj_ids.astype(np.int32), region_folder, scan_name)

            # object_folder = osp.join(output_scan_folder, 'objects')
            # if not os.path.exists(object_folder):
            #     os.makedirs(object_folder)
            
            # get_objects(xyz.astype(float), rgba[:, 0:3].astype(np.int32), region_ids.astype(np.int32), obj_ids.astype(np.int32), object_folder, scan_name)

            # pcd = o3d.t.geometry.PointCloud()
            # pcd.point.positions=xyz.astype(float)
            # pcd.point.colors=rgba[:, 0:3].astype(np.uint8)
            # pcd.point.obj_id=obj_ids.astype(np.int32)
            # pcd.point.region_id=region_ids.astype(np.int32)
            # print(pcd)
            # ply_file_name = scan_name + '_pc_result.ply'
            # o3d.t.io.write_point_cloud(ply_file_name, pcd)

            vertex = torch.cat([
                obj_pcs,
                obj_rgbs,
                obj_ids[:, None],
                region_ids[:, None]
            ], dim=1)

            vertex, region_indices_out, object_indices_out = sort_pointcloud(vertex)
            save_pointcloud(vertex, region_indices_out, object_indices_out, '', scan_name)

            if self.generate_freespace:
                floor_center, floor_size, floor_heading = calculate_bbox_hull(xyz)

                floor_center[2] -= (floor_size[2]/2-self.floor_height/2)
                floor_size[2] = self.floor_height

                floor_sizes = [list(floor_center)+list(floor_size)+[floor_heading, 0]]
                generate_free_space(scan_name, xyz.astype(float), floor_sizes, region_ids.astype(np.int32), self.floor_height)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', default='/media/navigation/easystore/Original_dataset/ARKitScenes/data/raw',
                        help="Input file of the mesh")
    parser.add_argument('--mapping_folder', default='./arkit_cat_mapping.csv',
                        help="Input folder of the category mapping")
    parser.add_argument('--output_folder', default='/home/navigation/Dataset/VLA_Dataset/ARKitScenes',
                        help="Output PLY file to save")
    parser.add_argument('--floor_height', default=0.35,
                        help="floor heigh for generating free space")
    parser.add_argument('--color_standard', default='css3',
                        help="color standard, chosen from css2, css21, css3, html4")
    parser.add_argument('--generate_freespace', action='store_true', help='Generate free spaces')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()

    skipped_scenes = [
        '42897846',
        '42897863',
        '42897868',
        '42897871',
        '47204424',
        '41069021' # missing .pt for some reason
    ]

    print('====================Start processing arkit training set====================')
    ARKitPreprocessor(
        os.path.join(args.input_folder, 'Training'), 
        args.mapping_folder, 
        args.output_folder, 
        args.floor_height, 
        args.color_standard, 
        args.generate_freespace,
        device=args.device
        ).create_arkit()
    print('====================End processing arkit training set====================')

    # print('====================Start processing arkit validation set====================')
    # ARKitPreprocessor(
    #     os.path.join(args.input_folder, 'Validation'), 
    #     args.mapping_folder, 
    #     args.output_folder, 
    #     args.floor_height, 
    #     args.color_standard, 
    #     args.generate_freespace
    #     ).create_arkit()
    # print('====================End processing arkit validation set====================')