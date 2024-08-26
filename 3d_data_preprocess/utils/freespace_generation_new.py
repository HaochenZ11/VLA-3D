import open3d as o3d
import numpy as np
from tqdm import tqdm
import csv
from numba import jit
from utils.transformations import rotz, rot_2d
from scipy.ndimage import binary_erosion
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from utils.headers import OBJECT_HEADER

occupied_length = 0.49
sample_step = 0.05
robot_height = 0.8
area_size = 50

delta = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]]) # searching direction


def are_points_in_rectangle(point: np.ndarray, vertices: np.ndarray):
    direction_vecs = vertices[[1, 2, 3, 0], :] - vertices
    norms = np.vstack([direction_vecs[:, 1], -direction_vecs[:, 0]]).T # 4 x 2
    norms /= np.linalg.norm(norms, axis=-1, keepdims=True)
    biases = -np.sum(norms * vertices, axis=-1)
    t = -np.sum(norms[None, :, :] * point[:, None, :], axis=-1) - biases
    return (t>0).all(axis=-1)


def are_points_in_already_sampled(points: np.ndarray, already_sampled):
    filt = np.zeros(points.shape[0], dtype=bool)
    for rectangle in already_sampled:
        filt |= are_points_in_rectangle(points, rectangle)
    return filt

def compute_rectangle(center_x, center_y, xlen, ylen, heading):
    R  = rot_2d(1*heading)
    xs = np.array([-xlen/2, -xlen/2, xlen/2, xlen/2]) # counterclockwise, starting from bottom left quadrant
    ys = np.array([-ylen/2, ylen/2, ylen/2, -ylen/2])
    corners_2d = np.dot(R, np.vstack([xs, ys]))
    corners_2d[0,:] += center_x
    corners_2d[1,:] += center_y

    return np.transpose(corners_2d)

@jit(nopython=True)
def find_connected_free_regions(arr, i, j):
    all_list = []
    cur_add_list = []
    row, col = arr.shape

    for ii in range(row):
        for jj in range(col):
            if arr[ii, jj] != 0:
                if [ii, jj] in cur_add_list:
                    continue
                else:
                    cur_add_list.append([ii, jj])
                    out_list = [[ii, jj]]
                    search_list = [[ii, jj]]
                    while search_list:
                        cur_search = []
                        for file in search_list:
                            for direction in range(len(delta)):
                                pos = [file[0] + delta[direction, 0], file[1] + delta[direction, 1]]
                                if row - 1 < pos[0] or pos[0] < 0 or col -1 < pos[1] or pos[1] < 0:
                                    continue
                                elif pos in out_list:
                                    continue
                                elif pos in cur_add_list:
                                    continue
                                else:
                                    if arr[pos[0], pos[1]] != 0:
                                        cur_search.append(pos)
                                        out_list.append(pos)
                        search_list = cur_search
                    all_list.append(out_list)
                    cur_add_list.extend(out_list)
    
    for region in all_list:
        for point in region:
            point[0] += i*area_size
            point[1] += j*area_size
    return all_list



def sample_points(cx, cy, cz, xlength, ylength, zlength, heading, region_id, floor_height):
    xs = np.arange(-1/2*xlength, 1/2*xlength, sample_step)
    x_num = len(xs)
    ys = np.arange(-1/2*ylength, 1/2*ylength, sample_step)
    y_num = len(ys)
    xs, ys = np.meshgrid(xs, ys)
    xs, ys = xs.ravel(), ys.ravel()
    zs = (floor_height-1/2*zlength+1/2*robot_height)*np.ones([len(xs),])
    binary_array = np.ones([x_num, y_num])

    R = rotz(1*heading)
    sampled_points = np.dot(R, np.vstack([xs, ys, zs]))
    sampled_points[0,:] += cx
    sampled_points[1,:] += cy
    sampled_points[2,:] += cz
    return np.hstack((np.transpose(sampled_points), region_id*np.ones([len(xs), 1]))), binary_array


def generate_region_free_space(
        point_positions: np.ndarray,
        already_sampled,
        floor_size,
        floor_height
):

    cx, cy, cz, xlength, ylength, zlength, heading, region_id = floor_size

    
    # Create grid
    sampled_points, binary_array = sample_points(cx, cy, cz, xlength, ylength, zlength, heading, region_id, floor_height)

    # Filter out everything above robot height and below the floor
    index, = np.where(point_positions[:, 2] <= cz-1/2*zlength+floor_height+robot_height)
    vertices_array = point_positions[index, :]
    index, = np.where(vertices_array[:, 2] >= cz-1/2*zlength+floor_height)
    vertices_array = vertices_array[index, :]

    R = rot_2d(heading)
    vertices_array_rot = (vertices_array[:, :2] - np.array([cx, cy])) @ R
    vertices_array_rot = (vertices_array_rot + np.array([xlength/2, ylength/2])) / sample_step
    vertices_array_rot = np.rint(vertices_array_rot).astype(np.int32)

    out_of_floor_filter = (vertices_array_rot[:, 0] < binary_array.shape[0]) & \
        (vertices_array_rot[:, 0] >= 0) & \
        (vertices_array_rot[:, 1] < binary_array.shape[1]) & \
        (vertices_array_rot[:, 1] >= 0)
    
    vertices_array_rot = vertices_array_rot[out_of_floor_filter]

    binary_array[vertices_array_rot[:, 0], vertices_array_rot[:, 1]] = 0

    window_size = int(occupied_length / sample_step)
    binary_array = binary_erosion(binary_array, np.ones((window_size, window_size)))
    

    previous_points_filter = are_points_in_already_sampled(sampled_points[:, :2], already_sampled)
    previous_points_filter = previous_points_filter.reshape(binary_array.shape[0], binary_array.shape[1], order='F')
    binary_array[previous_points_filter] = 0

    # plt.imshow(binary_array)
    # plt.show()

    sampled_points = sampled_points.reshape(binary_array.shape[0], binary_array.shape[1], -1, order='F')

    row, col = binary_array.shape
    x_batch = int(row/area_size)+1
    y_batch = int(col/area_size)+1
    free_regions = []
    for i in range(x_batch):
        for j in range(y_batch):
            free_regions.extend(find_connected_free_regions(binary_array[i*area_size:(i+1)*area_size, j*area_size:(j+1)*area_size], i, j))
    
    return free_regions, sampled_points, heading

def generate_free_space(
    scene_path,
    scan_name,
    region_ids,
    point_positions_split,
    floor_sizes,
    floor_height
):
    free_space_vertices = []
    previous_free_space = []
    for floor_size in tqdm(floor_sizes):

        region = floor_size[-1]
        region_positions = point_positions_split[region]
        free_regions, target_points, heading = generate_region_free_space(region_positions, previous_free_space, floor_size, floor_height)


        for free_region in free_regions:
            large_rectangle = []
            for point in free_region:
                large_rectangle.append(target_points[point[0], point[1], :])
            free_space_vertices.extend(large_rectangle)
            bbox_midpoint = target_points[int((np.max(np.array(free_region)[:, 0]) + np.min(np.array(free_region)[:, 0]))/2), int((np.max(np.array(free_region)[:, 1]) + np.min(np.array(free_region)[:, 1]))/2), :]
            bbox_xlen = (np.max(np.array(free_region)[:, 0]) - np.min(np.array(free_region)[:, 0])+1)*sample_step
            bbox_ylen = (np.max(np.array(free_region)[:, 1]) - np.min(np.array(free_region)[:, 1])+1)*sample_step
            
            previous_free_space.append(compute_rectangle(bbox_midpoint[0], bbox_midpoint[1], bbox_xlen, bbox_ylen, heading))

    if len(free_space_vertices) <= 1:
        free_space_vertices = np.array(free_space_vertices).reshape(1, -1)
    else:
        free_space_vertices = np.array(free_space_vertices)
    free_space_pcd = o3d.t.geometry.PointCloud()
    free_space_pcd.point.positions=free_space_vertices[:, 0:3].astype(np.float32)
    free_space_pcd.point.region_id=free_space_vertices[:, 3:].reshape(-1, 1).astype(np.int32)
    free_space_ply_file_name = os.path.join(scene_path, scan_name + '_free_space_pc_result.ply')
    o3d.t.io.write_point_cloud(free_space_ply_file_name, free_space_pcd)

