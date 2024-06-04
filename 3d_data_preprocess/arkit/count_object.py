from collections import Counter
from scipy.spatial import KDTree
import os
import os.path as osp
import numpy as np
import csv
import json
import math
import open3d as o3d
import matplotlib
from plyfile import PlyData
from numba import jit
from tqdm import tqdm
import torch
from typing import List
from collections import defaultdict
from utils.dominant_colors import judge_color, anchor_colors_array
from utils.freespace_generation import generate_free_space
import sys
sys.path.append("..")


def inside_test(points, cube3d):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).

    Returns the indices of the points array which are inside the cube3d
    """
    b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

    dir1 = (t1-b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2-b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b4-b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t3)/2.0

    dir_vec = points - cube3d_center

    res1 = np.where((np.absolute(dir_vec @ dir1) * 2) <= size1)[0]
    res2 = np.where((np.absolute(dir_vec @ dir2) * 2) <= size2)[0]
    res3 = np.where((np.absolute(dir_vec @ dir3) * 2) <= size3)[0]

    return list(set(res1) & set(res2) & set(res3))


def get_bbox(center, size, R):

    h = float(size[2])
    w = float(size[1])
    l = float(size[0])
    # heading_angle = -heading_angle - np.pi / 2

    # center[2] = center[2] + h / 2
    # R = rotz(1*heading_angle)
    l = l/2
    w = w/2
    h = h/2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [-h, -h, -h, -h, h, h, h, h]
    corners_3d = np.dot(np.transpose(R), np.vstack(
        [x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)


def crop_pc(xyz, bbox_center, bbox_length, bbox_rotation):
    cube3d = get_bbox(bbox_center, bbox_length, bbox_rotation)
    index = inside_test(xyz, cube3d)

    bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [
        5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for _ in range(len(bbox_lines))]  # red

    bbox = o3d.geometry.LineSet()
    bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
    bbox.colors = o3d.utility.Vector3dVector(colors)
    bbox.points = o3d.utility.Vector3dVector(cube3d)
    return index, bbox


def calculate_bbox(points):
    points_2d = points[:, 0:2]
    center = np.mean(points_2d, axis=0)
    delta = points_2d - center
    objC11 = np.mean(delta[:, 0]*delta[:, 0])
    objC12 = np.mean(delta[:, 0]*delta[:, 1])
    objC22 = np.mean(delta[:, 1]*delta[:, 1])
    obj_matrix = np.array([[objC11, objC12], [objC12, objC22]])
    W, V = np.linalg.eig(obj_matrix)
    heading = math.atan2(V[1, 0], V[0, 0])
    l = math.cos(heading) * points_2d[:, 0] + \
        math.sin(heading) * points_2d[:, 1]
    w = -math.sin(heading) * points_2d[:, 0] + \
        math.cos(heading) * points_2d[:, 1]
    h = points[:, 2]
    minL = np.min(l)
    maxL = np.max(l)
    minW = np.min(w)
    maxW = np.max(w)
    minH = np.min(h)
    maxH = np.max(h)

    midL = (minL + maxL)/2
    midW = (minW + maxW)/2
    midX = math.cos(heading) * midL - math.sin(heading) * midW
    midY = math.sin(heading) * midL + math.cos(heading) * midW
    midH = (minH + maxH)/2

    lenX = maxL - minL
    lenY = maxW - minW
    lenZ = maxH - minH

    center = [midX, midY, midH]
    size = [lenX, lenY, lenZ]

    return center, size, heading


def get_regions(xyz, colors, region_ids, object_ids, input_folder, scan_name):
    '''
    Split scene into region and store data by region
    :return:
    '''

    for region in tqdm(range(-1, np.max(region_ids)+1)):
        region_ply_file = os.path.join(
            input_folder, scan_name + '_region_' + str(region) + '_pc_result.ply')

        index, = np.where(region_ids.flatten() == int(region))
        if len(index) > 0:
            xyz_region = xyz[index, :]
            colors_region = colors[index, :]

            region_pcd = o3d.t.geometry.PointCloud()
            region_pcd.point.positions = xyz_region
            region_pcd.point.colors = colors_region
            o3d.t.io.write_point_cloud(region_ply_file, region_pcd)


def get_objects(xyz, colors, region_ids, object_ids, input_folder, scan_name):
    '''
    Split scene into region and store data by region
    :return:
    '''

    for object in tqdm(range(np.max(object_ids)+1)):
        index, = np.where(object_ids.flatten() == int(object))

        if len(index) > 0:
            region = int(np.mean(region_ids[index, :]))
            region_ply_file = os.path.join(input_folder, scan_name + '_region_' + str(
                region) + '_object_' + str(object) + '_pc_result.ply')

            xyz_object = xyz[index, :]
            colors_object = colors[index, :]

            region_pcd = o3d.t.geometry.PointCloud()
            region_pcd.point.positions = xyz_object
            region_pcd.point.colors = colors_object
            o3d.t.io.write_point_cloud(region_ply_file, region_pcd)


def count_arkit(folder_name, scan_names, output_folder, floor_height, generate_freespace=False):
    count = 0
    for scan_name in tqdm(scan_names):
        json_data_name = osp.join(
            folder_name, scan_name, scan_name + "_3dod_annotation.json")
        with open(json_data_name) as f:
            json_data = json.load(f)

        for obj in json_data['data']:
            count += 1

    print(count)


if __name__ == '__main__':
    folder_name = r'/media/navigation/easystore/Original_dataset/ARKitScenes/data/raw/Training'
    output_folder = r'/media/navigation/easystore/VLA_Dataset_more'
    scan_names = os.listdir(folder_name)
    floor_height = 0.15
    count_arkit(folder_name, scan_names, output_folder, floor_height, False)
