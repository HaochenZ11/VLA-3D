import os
from os import path as osp
import re
import csv
from plyfile import PlyData, PlyElement
import numpy as np
import math
import open3d as o3d
import matplotlib
from tqdm import tqdm
import argparse
from collections import defaultdict
import sys
import torch
from scipy.spatial import KDTree
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.freespace_generation_new import generate_free_space
from utils.dominant_colors_new_lab import judge_color, generate_color_anchors
from scipy.spatial import ConvexHull
from utils.bbox_utils import calculate_bbox, calculate_bbox_hull
from utils.headers import OBJECT_HEADER, REGION_HEADER
from utils.pointcloud_utils import write_ply_file, sort_pointcloud, get_regions, get_objects

class MatterportPreprocessor:
    def __init__(
        self,
        top_scan_dir: str, # the path containing scan files
        save_dir:str, # the path to store the processed results
        mapping_file: str, # the path containing the matterport class mapping file
        floor_height: float, # the floor height for generating free space
        color_standard: str, # colors standars to use for domain color calculation (css21, css3, html4, css2)

        generate_freespace=False # whether generate free space or not
        ):
        self.region_mapping = {'a': "bathroom",'b': "bedroom",'c': "closet",'d': "dining room",'e': "entryway/foyer/lobby",'f': "familyroom",'g': "garage",'h': "hallway",'i': "library",'j': "laundryroom/mudroom",'k': "kitchen",'l': "living room", \
                                'm': "meetingroom/conferenceroom",'n': "lounge",'o': "office",'p': "porch/terrace/deck/driveway",'r': "rec/game",'s': "stairs",'t': "toilet",'u': "utilityroom/toolroom",'v': "tv", \
                                'w': "workout/gym/exercise",'x': "outdoor areas",'y': "balcony",'z': "other room",'B': "bar",'C': "classroom",'D': "dining booth",'S': "spa/sauna",'Z': "junk",'-': "no label"}
        self.top_scan_dir = top_scan_dir
        self.save_dir = save_dir
        self.mapping_file = mapping_file
        self.scan_names = os.listdir(top_scan_dir)
        self.floor_height = floor_height
        self.anchor_colors_array, self.anchor_colors_array_hsv, self.anchor_colors_name = generate_color_anchors(color_standard)
        self.tree = KDTree(self.anchor_colors_array_hsv)
        self.generate_freespace = generate_freespace

    def find_points_and_calculate_bbox_and_top3_color3(self, plyData, vertices, target_obj_id): # crop the points of the target object and calulate bbox and domain colors 

        object_ids = plyData['face'].data['object_id']
        index, = np.where(object_ids ==  target_obj_id)

        object_point_index = np.unique(np.array(plyData['face'].data['vertex_indices'][index].tolist()).flatten())
        object_points = vertices[object_point_index]
        center, size, heading = calculate_bbox_hull(object_points[:, :3])

        colors = object_points[:, 3:]/255
        color_3 = judge_color(colors, self.tree, self.anchor_colors_array, self.anchor_colors_name)
        
        return center, size, heading, color_3, object_point_index


    def create_matterport(self):
        for scan_name in tqdm(self.scan_names):
            os.chdir(osp.join(self.save_dir, scan_name))
            scan_data_file = osp.join(self.top_scan_dir, scan_name, scan_name + "_semantic.ply")
            data = PlyData.read(scan_data_file)
            print(f"Read scan {scan_name}")

            raw_label_mapping_list = []
            nyu_mapping_list = []
            nyu40_mapping_list = []
            nyu_id_mapping_list = []
            nyu40_id_mapping_list = []
            with open(self.mapping_file, 'r') as f: # create a mapping from raw label to nyu class, nyu40 class, nyu id and nyu40 id
                csv_reader = csv.reader(f, delimiter='\t')
                next(csv_reader)
                for row in csv_reader:
                    raw = row[2].split(r' /')[0]
                    raw = raw.split(r' \\')[0]
                    raw = raw.split(r' (')[0]
                    raw = raw.split(r'/')[0]
                    raw = raw.split(r'\\')[0]
                    raw = raw.split(r'(')[0]
                    raw_label_mapping_list.append(raw)
                    nyu_mapping_list.append(row[7])
                    nyu40_mapping_list.append(row[8])
                    nyu_id_mapping_list.append(row[4])
                    nyu40_id_mapping_list.append(row[5])

            def mapping(label_idx):
                raw_label = raw_label_mapping_list[int(label_idx) % len(nyu_mapping_list)]
                nyu_label = nyu_mapping_list[int(label_idx) % len(nyu_mapping_list)]
                if not nyu_label:
                    nyu_label = 'unknown'
                nyu40_label = nyu40_mapping_list[int(label_idx) % len(nyu40_mapping_list)]
                nyu_id = nyu_id_mapping_list[int(label_idx) % len(nyu_id_mapping_list)]
                if not nyu_id:
                    nyu_id = '20'
                nyu40_id = nyu40_id_mapping_list[int(label_idx) % len(nyu40_id_mapping_list)]

                return raw_label, nyu_label, nyu40_label, nyu_id, nyu40_id



            file_name = osp.join(self.top_scan_dir, scan_name, scan_name + '.house')
            region_lines = []
            category_lines = []
            object_lines = []
            with open(file_name, 'r') as f: # read .house file for region and object segmentation
                info = f.readline()
                cmd, version = re.split('  | ', info)
                if (cmd != 'ASCII'):
                    print('Unable to read ascii file: ', file_name)
                    return 0
                info = f.readline()
                line = re.split('  | ', info)
                if (version == '1.0'):
                    cmd = line[0]
                    name_buffer = line[1]
                    label_buffer = line[2]
                    nimages = line[3]
                    npanoramas = line[4]
                    nvertices = line[5]
                    nsurfaces = line[6]
                    nregions = line[7]
                    nlevels = line[8]

                    nsegments = 0
                    nobjects = 0
                    ncategories = 0
                    nportals = 0
                else:
                    cmd = line[0]
                    name_buffer = line[1]
                    label_buffer = line[2]
                    nimages = line[3]
                    npanoramas = line[4]
                    nvertices = line[5]
                    nsurfaces = line[6]
                    nsegments = line[7]
                    nobjects = line[8]
                    ncategories = line[9]
                    nregions = line[10]
                    nportals = line[11]
                    nlevels = line[12]

                #read levels
                level_sizes = []
                for i in range(int(nlevels)):
                    level_data = f.readline()
                    line = re.split('  | ', level_data)
                    level_sizes.append(line)

                # save region segmentation in our format
                region_file_name = scan_name + '_region_result.csv'
                with open(region_file_name, 'w', newline='') as f_region:
                    region_writer = csv.writer(f_region, delimiter=',')
                    region_writer.writerow(REGION_HEADER)

                    #read regions
                    region_level_mapping = []
                    for i in tqdm(range(int(nregions))):
                        region_data = f.readline()
                        line = re.split('  | ', region_data)
                        cmd = line[0]
                        region_index = line[1]
                        rengion_level_index = line[2]
                        region_level_mapping.append(int(rengion_level_index))

                        region_label = line[5]
                        region_x = line[6]
                        region_y = line[7]
                        region_z = line[8]
                        region_xlow = line[9]
                        region_ylow = line[10]
                        region_zlow = line[11]
                        region_xhigh = line[12]
                        region_yhigh = line[13]
                        region_zhigh = line[14]

                        if cmd != 'R':
                            print('Error reading region ', i)
                            return 0

                        region_center = [float((float(region_xhigh)+float(region_xlow))/2), float((float(region_yhigh)+float(region_ylow))/2), float((float(region_zhigh)+float(region_zlow))/2)]
                        region_size = [float(region_xhigh)-float(region_xlow), float(region_yhigh)-float(region_ylow), float(region_zhigh)-float(region_zlow)]
                        region_heading = float(0.0)
                        
                        region_info = []
                        region_info.append(region_index)
                        region_info.append(self.region_mapping[region_label])
                        region_info += list(region_center)
                        region_info += list(region_size)
                        region_info.append(region_heading)
                        region_writer.writerow(region_info)

                #read portals
                for i in range(int(nportals)):
                    portal_data = f.readline()

                #read surfaces
                for i in range(int(nsurfaces)):
                    surface_data = f.readline()

                #read vertices
                for i in range(int(nvertices)):
                    vertice_data = f.readline()

                #read panoramas
                for i in range(int(npanoramas)):
                    panorama_data = f.readline()

                #read images
                for i in range(int(nimages)):
                    image_data = f.readline()

                #read categories
                for i in range(int(ncategories)):
                    category_data = f.readline()
                    line = re.split('  | ', category_data)
                    cmd = line[0]
                    # category_index = line[1]
                    category_mapping_index = line[2]
                    # category_mapping_name = line[3]
                    if cmd != 'C':
                        print('Error reading category ', i)
                        return 0

                    category_lines.append(category_mapping_index)

                vertices = np.array(data['vertex'].data.tolist())
                obj_ids = np.zeros(vertices.shape[0], dtype=np.int32)
                region_ids = np.zeros_like(obj_ids)
                #read objects
                floor_sizes = []
                for i in tqdm(range(int(nobjects))):
                    object_line = []
                    object_data = f.readline()
                    line = re.split('  | ', object_data)
                    cmd = line[0]
                    if cmd != 'O':
                        print('Error reading object ', i)
                        return 0
                    object_index = line[1]
                    region_index = line[2]
                    category_index = line[3]
                    center, size, heading, color_3, point_object_indices = self.find_points_and_calculate_bbox_and_top3_color3(data, vertices, int(object_index))
                    obj_ids[point_object_indices] = int(object_index)
                    region_ids[point_object_indices] = int(region_index)
                    
                    object_line.append(object_index)
                    object_line.append(region_index)

                    raw_label, nyu_label, nyu40_label, nyu_id, nyu40_id = mapping(int(category_index))
                    if nyu40_label == 'floor':
                        floor_sizes.append([center[0], center[1], float(level_sizes[region_level_mapping[int(region_index)]][6]), size[0], size[1], 0.02, heading, int(region_index)])
                    object_line.append(raw_label)
                    object_line.append(nyu_id)
                    object_line.append(nyu40_id)
                    object_line.append(nyu_label)
                    object_line.append(nyu40_label)

                    object_line += center
                    object_line += size
                    object_line.append(heading)
                    object_line.append('_')

                    object_line += color_3

                    object_lines.append(object_line)

            # save object segmentation in our format
            with open(scan_name + '_object_result.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(OBJECT_HEADER)
                for i in object_lines:
                    writer.writerow(i)

            # vertices = np.vstack(vertices)
            # region_folder = osp.join(self.save_dir, scan_name, 'regions')
            # if not os.path.exists(region_folder):
            #     os.makedirs(region_folder)
            # get_regions(
            #     vertices[:, 0:3].astype(np.float32), 
            #     vertices[:, 3:6].astype(np.uint8), 
            #     np.vstack(region_ids).reshape(-1, 1).astype(np.int32), 
            #     np.vstack(obj_ids).reshape(-1, 1).astype(np.int32), region_folder, scan_name)

            # object_folder = osp.join(self.save_dir, scan_name, 'objects')
            # if not os.path.exists(object_folder):
            #     os.makedirs(object_folder)
            # get_objects(
            #     vertices[:, 0:3].astype(np.float32), 
            #     vertices[:, 3:6].astype(np.uint8), 
            #     np.vstack(region_ids).reshape(-1, 1).astype(np.int32), 
            #     np.vstack(obj_ids).reshape(-1, 1).astype(np.int32), object_folder, scan_name)

            vertex = torch.cat([
                torch.from_numpy(vertices),
                torch.from_numpy(obj_ids)[:, None],
                torch.from_numpy(region_ids)[:, None]
            ], dim=1)

            vertex, region_indices_out, object_indices_out = sort_pointcloud(vertex)

            torch.save(region_indices_out, f'{scan_name}_region_split.pt')
            torch.save(object_indices_out, f'{scan_name}_object_split.pt')
            write_ply_file(vertex[:, :6], f'{scan_name}_pc_result.ply')

            # generate free space
            if self.generate_freespace:
                point_positions = vertex[:, :3].to_numpy()
                point_region_ids = vertex[:, 7].astype(torch.uint8).to_numpy()

                generate_free_space(scan_name, point_positions, floor_sizes, point_region_ids, self.floor_height)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_scan_dir', default="/media/navigation/easystore/Original_dataset/matterport/v1/tasks/mp3d_habitat/mp3d")
    parser.add_argument('--save_dir', default="/home/navigation/Dataset/VLA_Dataset")
    parser.add_argument('--mapping_file', default='/media/navigation/easystore/Original_dataset/matterport/data/category_mapping.tsv')
    parser.add_argument('--floor_height', default=0.1)
    parser.add_argument('--color_standard', default='css3')
    parser.add_argument('--generate_freespace', action='store_true')

    args = parser.parse_args()

    print('====================Start processing matterport====================')
    MatterportPreprocessor(args.top_scan_dir, args.save_dir, args.mapping_file, args.floor_height, args.color_standard, args.generate_freespace).create_matterport()
    print('====================End processing matterport====================')
