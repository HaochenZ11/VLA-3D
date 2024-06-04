import open3d as o3d
import numpy as np
from tqdm import tqdm
import csv
from numba import jit
from utils.transformations import rotz, rot_2d
import cv2


occupied_length = 0.49
sample_step = 0.5
robot_height = 0.8
area_size = 5

delta = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]]) # searching direction

def is_point_in_rectangle(point, vertices):
    intersections = 0
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        
        # jude if lines intersect
        if (point[1] > min(y1, y2)) and (point[1] <= max(y1, y2)) and (point[0] <= max(x1, x2)):
            x_intersect = (point[1] - y1) * (x2 - x1) / (y2 - y1) + x1
            if x_intersect > point[0]:
                intersections += 1
    
    # if the intersections number is odd, the point is inside the rectangle
    return intersections % 2 == 1

def is_point_in_already_sampled(point, already_sampled):
    flag = False
    for rectangle in already_sampled:
        if is_point_in_rectangle(point, rectangle):
            flag = True
            break
    return flag

def compute_rectangle(center_x, center_y, xlen, ylen, heading):
    R  = rot_2d(1*heading)
    xs = np.array([-xlen/2, -xlen/2, xlen/2, xlen/2])
    ys = np.array([-ylen/2, ylen/2, ylen/2, -ylen/2])
    corners_2d = np.dot(R, np.vstack([xs, ys]))
    corners_2d[0,:] += center_x
    corners_2d[1,:] += center_y

    return np.transpose(corners_2d)

@jit(nopython=True)
def find_max_min(arr, i, j):
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
    xs = np.tile(xs, y_num)
    ys = np.repeat(ys, x_num, axis=None)
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
    # if region_id != -1:
    #     with open(region_file_name, 'r', newline='') as f_region:
    #         csv_reader = csv.reader(f_region, delimiter=',')
    #         for i, row in enumerate(csv_reader):
    #             if i == region_id + 1:
    #                 region_info = list(map(float, row[2:]))
    #                 break
    #     region_min, region_max = find_region_max_min(*region_info)
    # else:
    region_min = np.min(point_positions, axis=0)
    region_max = np.max(point_positions, axis=0)
    
    sampled_points, binary_array = sample_points(cx, cy, cz, xlength, ylength, zlength, heading, region_id, floor_height)
    index, = np.where(cz-1/2*zlength+floor_height+robot_height>=point_positions[:, 2])
    vertices_array = point_positions[index, :]
    index, = np.where(vertices_array[:, 2]>=cz-1/2*zlength+floor_height)
    vertices_array = vertices_array[index, :]

    out_of_region_filter = (sampled_points[:, 0] > region_max[0]) | \
        (sampled_points[:, 0] < region_min[0]) | \
        (sampled_points[:, 1] > region_max[1]) | \
        (sampled_points[:, 1] < region_min[1])
    out_of_region_filter = out_of_region_filter.reshape(binary_array.shape, order='F')
    binary_array[out_of_region_filter] = 0
    sample_x_max = (sampled_points[:, 0]+occupied_length*1/2).reshape(binary_array.shape, order='F')
    sample_x_min = (sampled_points[:, 0]-occupied_length*1/2).reshape(binary_array.shape, order='F')
    sample_y_max = (sampled_points[:, 1]+occupied_length*1/2).reshape(binary_array.shape, order='F')
    sample_y_min = (sampled_points[:, 1]-occupied_length*1/2).reshape(binary_array.shape, order='F')
    sampled_points = sampled_points.reshape(binary_array.shape[0], binary_array.shape[1], -1, order='F')

    for i in tqdm(range(len(binary_array))):
        vertex_intersection = (sample_x_max[i, :, None] >= vertices_array[None, :, 0]) & \
            (sample_x_min[i, :, None] <= vertices_array[None, :, 0]) & \
            (sample_y_min[i, :, None] <= vertices_array[None, :, 1]) & \
            (sample_y_max[i, :, None] >= vertices_array[None, :, 1])
        vertex_intersection = np.sum(vertex_intersection, axis=1) == 0
        binary_array[i, ~vertex_intersection] = 0
    
    for i in range(binary_array.shape[0]):
        for j in range(binary_array.shape[1]):
            if binary_array[i, j] != 0:
                point = sampled_points[i, j, 0:2]
                if is_point_in_already_sampled(point, already_sampled):
                    binary_array[i, j] = 0

    row, col = binary_array.shape
    x_batch = int(row/area_size)+1
    y_batch = int(col/area_size)+1
    free_regions = []
    for i in range(x_batch):
        for j in range(y_batch):
            free_regions.extend(find_max_min(binary_array[i*area_size:(i+1)*area_size, j*area_size:(j+1)*area_size], i, j))
    
    return free_regions, sampled_points, heading

def generate_free_space(
    scan_name,
    point_positions,
    floor_sizes,
    region_ids,
    floor_height
):
    free_space_vertices = []
    free_space_obj_id = []
    existed_free_space = []
    for floor_size in floor_sizes:

        region = floor_size[-1]
        index, = np.where(region_ids.flatten() == int(region))
        region_positions = point_positions[index, :]
            
        free_regions, target_points, heading = generate_region_free_space(region_positions, existed_free_space, floor_size, floor_height)

        with open(scan_name + '_object_result.csv', 'a', newline='') as f:
            csv_write = csv.writer(f, delimiter=',')
            for i, free_region in enumerate(tqdm(free_regions)):
                large_rectangle = []
                for point in free_region:
                    large_rectangle.append(target_points[point[0], point[1], :])
                free_space_vertices.extend(large_rectangle)
                free_space_obj_id.extend([-i]*len(large_rectangle))
                target_point = target_points[int((np.max(np.array(free_region)[:, 0]) + np.min(np.array(free_region)[:, 0]))/2), int((np.max(np.array(free_region)[:, 1]) + np.min(np.array(free_region)[:, 1]))/2), :]
                xlen = (np.max(np.array(free_region)[:, 0]) - np.min(np.array(free_region)[:, 0])+1)*sample_step
                ylen = (np.max(np.array(free_region)[:, 1]) - np.min(np.array(free_region)[:, 1])+1)*sample_step
                
                regions = list(np.array(large_rectangle)[:, -1].astype(int))
                region = max(set(regions),key=regions.count)
                existed_free_space.append(compute_rectangle(target_point[0], target_point[1], xlen, ylen, heading))
                csv_write.writerow([-i-1, region, 'space', '_', '_', '_', '_', target_point[0], target_point[1], target_point[2], xlen, ylen, robot_height, heading, '_', '_', '_', '_', '_', '_', '_', '_', '_', '_'])

    if len(free_space_vertices) <= 1:
        free_space_vertices = np.array(free_space_vertices).reshape(1, -1)
    else:
        free_space_vertices = np.array(free_space_vertices)
    free_space_pcd = o3d.t.geometry.PointCloud()
    free_space_pcd.point.positions=free_space_vertices[:, 0:3].astype(np.float32)
    free_space_pcd.point.colors=np.array([[255, 0, 0]]*len(free_space_vertices)).astype(np.uint8)
    free_space_pcd.point.obj_id=np.array(free_space_obj_id).reshape(-1, 1).astype(np.int32)
    free_space_pcd.point.region_id=free_space_vertices[:, 3:].reshape(-1, 1).astype(np.int32)
    free_space_ply_file_name = scan_name + '_free_space_pc_result.ply'
    o3d.t.io.write_point_cloud(free_space_ply_file_name, free_space_pcd)

