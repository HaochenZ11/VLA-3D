from plyfile import PlyData, PlyElement
import numpy.lib.recfunctions as rf
import numpy as np
import torch
import tqdm
import os
import open3d as o3d


def get_regions(xyz, colors, region_ids, object_ids, input_folder, scan_name):
    '''
    Split scene into region and store data by region
    :return:
    '''

    for region in tqdm(range(-1, np.max(region_ids)+1)):
        region_ply_file = os.path.join(input_folder, scan_name + '_region_' + str(region) + '_pc_result.ply')

        index, = np.where(region_ids.flatten() == int(region))
        if len(index) > 0:
            xyz_region = xyz[index, :]
            colors_region = colors[index, :]
            
            region_pcd = o3d.t.geometry.PointCloud()
            region_pcd.point.positions=xyz_region
            region_pcd.point.colors=colors_region
            o3d.t.io.write_point_cloud(region_ply_file, region_pcd)


def get_objects(xyz, colors, region_ids, object_ids, input_folder, scan_name):
    '''
    Split scene into region and store data by region
    :return:
    '''

    for object in tqdm(range(np.max(object_ids)+1)):
        index, = np.where(object_ids.flatten() == int(object))

        region = int(np.mean(region_ids[index, :]))
        object_ply_file = os.path.join(input_folder, scan_name + '_region_' + str(region) + '_object_' + str(object) + '_pc_result.ply')
        if len(index) > 0:
            xyz_object = xyz[index, :]
            colors_object = colors[index, :]
            
            object_pcd = o3d.t.geometry.PointCloud()
            object_pcd.point.positions=xyz_object
            object_pcd.point.colors=colors_object
            o3d.t.io.write_point_cloud(object_ply_file, object_pcd)


def sort_pointcloud(vertex: torch.Tensor):
    
    vertex = vertex[torch.argsort(vertex[:, 7])]
    region_indices, region_splits = torch.unique_consecutive(vertex[:, 7], return_counts=True)
    vertex_split = torch.split(vertex, region_splits.cpu().tolist())
    vertex_split = [vert[torch.argsort(vert[:, 6])] for vert in vertex_split]

    object_indices_and_counts = [torch.unique_consecutive(vert[:, 6], return_counts=True) for vert in vertex_split]
    object_indices = torch.cat([o[0] for o in object_indices_and_counts])
    object_splits = torch.cat([o[1] for o in object_indices_and_counts])

    region_indices_out = torch.stack([region_indices.int(), region_splits.int()], dim=1)

    object_indices_out = torch.stack([object_indices.int(), object_splits.int()], dim=1)

    vertex = torch.vstack(vertex_split)

    return vertex, region_indices_out, object_indices_out


def write_ply_file(vertex: torch.Tensor | np.ndarray, output_ply_path, obj_id=False, region_id=False):

    dtype = [
        ('x', 'f4'), 
        ('y', 'f4'), 
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
    ]

    if obj_id:
        dtype.append(('obj_id', 'i4'))
    
    if region_id:
        dtype.append(('region_id', 'i4'))

    if isinstance(vertex, torch.Tensor):
        vertex = vertex.cpu().numpy()
    
    vertex_structured = rf.unstructured_to_structured(
        vertex, dtype=dtype)
    
    vertex_element = PlyElement.describe(vertex_structured, 'vertex')

    out_ply = PlyData([vertex_element])

    out_ply.write(output_ply_path)