import open3d as o3d
import numpy as np
import numpy.lib.recfunctions as rf
import argparse
import csv
import time
import torch
from typing import List, Dict
from trimesh.remesh import subdivide_to_size
from plyfile import PlyData

SEED = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

def subdivide_mesh(
    mesh_objects: List[torch.Tensor],
    max_edge: float 
):
    (
        vertices,
        triangles,
        triangle_normals,
        triangle_uvs,
        material_ids,
        textures,
        semantic_triangle_uvs,
        semantic_material_ids,
        semantic_textures
    ) = mesh_objects

    triangle_uvs_list = [triangle_uvs, semantic_triangle_uvs]
    material_ids_list = [material_ids, semantic_material_ids]
    
    vertex_uvs_list = []
    for triangle_uvs in triangle_uvs_list:
        triangle_uvs = triangle_uvs.view(-1, 2)
        vertex_uvs = torch.zeros((vertices.shape[0], 2))
        vertex_uvs[triangles.view(-1)] = triangle_uvs.float()
        vertex_uvs_list.append(vertex_uvs)

    verts_out, triangles_out, index = subdivide_to_size(
        torch.hstack([vertices]+vertex_uvs_list).cpu().numpy(),
        triangles.cpu().numpy(),
        max_edge,
        max_iter=10,
        return_index=True)
    
    verts_out = torch.from_numpy(verts_out).to(DEVICE)
    triangles_out = torch.from_numpy(triangles_out).to(DEVICE)
    index = torch.from_numpy(index).to(DEVICE)

    vertices_out = verts_out[:, :3]
    vertex_uvs_out_list = [verts_out[:, 2*i+3:2*i+5] for i in range(len(triangle_uvs_list))]

    print(triangles_out.shape)

    triangle_uvs_out_list = [vertex_uvs_out[triangles_out.view(-1)].view(-1, 3, 2) for vertex_uvs_out in vertex_uvs_out_list]

    material_ids_out_list = [material_ids[index] for material_ids in material_ids_list]
    triangle_normals_out = triangle_normals[index]

    return (
        vertices_out,
        triangles_out,
        triangle_normals_out,
        triangle_uvs_out_list[1],
        material_ids_out_list[0],
        textures,
        triangle_uvs_out_list[1],
        material_ids_out_list[1],
        semantic_textures
    )


def sample_points_uniformly_pytorch(
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        triangle_uvs_list: List[torch.Tensor],
        semantic_triangle_props : Dict[str, torch.Tensor] = {},
        num_points = None,
        sampling_density = None,
        filtered_triangle_indices: torch.Tensor = None,
        seed=SEED,
        device=DEVICE):
    
    
    side_1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    side_2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    triangle_areas = 0.5 * torch.norm(torch.cross(side_1, side_2, dim=1), dim=1)

    if filtered_triangle_indices is not None:
        triangle_indices_mask = torch.zeros_like(triangle_areas, dtype=torch.bool)
        triangle_indices_mask[filtered_triangle_indices] = True
        triangle_areas[~triangle_indices_mask] = 0.0
    
    total_surface_area = torch.sum(triangle_areas)
    triangle_pdf = triangle_areas / total_surface_area

    torch.random.manual_seed(seed)

    # Indices of triangles corresponding to each point

    if sampling_density is not None:
        num_points = int(total_surface_area / sampling_density)
        print(f'Sampling based on density.\nTotal surface area: {total_surface_area}\nNumber of samples: {num_points}')
    
    triangle_indices = torch.multinomial(triangle_pdf, num_points, replacement=True).to(device)

    r1 = torch.sqrt(torch.rand(num_points, 1, device=device))
    r2 = torch.rand(num_points, 1, device=device)
    a = 1 - r1
    b = r1 * (1 - r2)
    c = r1 * r2

    point_triangles = triangles[triangle_indices]

    points = a * vertices[point_triangles[:, 0]] \
        + b * vertices[point_triangles[:, 1]] \
        + c * vertices[point_triangles[:, 2]]
    
    # Triangle UVs are N x 3 x 2 arrays, where N is the number of triangles

    point_uvs_list = []

    for triangle_uvs in triangle_uvs_list:
        point_uvs = a * triangle_uvs[triangle_indices, 0, :] \
            + b * triangle_uvs[triangle_indices, 1, :] \
            + c * triangle_uvs[triangle_indices, 2, :]
        point_uvs_list.append(point_uvs)
    
    point_props_list = []
    for prop, triangle_props in semantic_triangle_props.items():
        point_props_list.append(triangle_props[triangle_indices, 0])
    
    return triangle_indices, points, point_uvs_list, point_props_list


def load_meshes(path, semantic_path, semantic_ply_props=[]):
    mesh = o3d.io.read_triangle_mesh(path, True)
    mesh.compute_vertex_normals()

    vertices = torch.from_numpy(np.asarray(mesh.vertices)).to(DEVICE)
    triangles = torch.from_numpy(np.asarray(mesh.triangles)).to(DEVICE)
    triangle_normals = torch.from_numpy(np.asarray(mesh.triangle_normals)).to(DEVICE)

    triangle_uvs = torch.from_numpy(np.asarray(mesh.triangle_uvs)).to(DEVICE).view(-1, 3, 2)
    material_ids = torch.from_numpy(np.asarray(mesh.triangle_material_ids)).to(DEVICE)
    textures = torch.from_numpy(np.array([np.asarray(texture) for texture in mesh.textures])).to(DEVICE)

    mesh_objects = (
        vertices,
        triangles,
        triangle_normals,
        triangle_uvs,
        material_ids,
        textures
    )

    if str(semantic_path).endswith('.ply'):
        semantic_ply = PlyData.read(semantic_path)
        semantic_triangles = np.vstack(semantic_ply['face']['vertex_indices'])
        semantic_triangle_props = {prop: torch.from_numpy(semantic_ply['vertex'][prop][semantic_triangles.ravel()].astype(np.int32).reshape(-1, 3)).to(DEVICE) for prop in semantic_ply_props}
        semantic_mesh_objects = (
            'vertex_colors',
            semantic_triangle_props
        )
    else:
        semantic_mesh = o3d.io.read_triangle_mesh(semantic_path, True)
        semantic_triangle_uvs = torch.from_numpy(np.asarray(semantic_mesh.triangle_uvs)).to(DEVICE).view(-1, 3, 2)
        semantic_material_ids = torch.from_numpy(np.asarray(semantic_mesh.triangle_material_ids)).to(DEVICE)
        semantic_textures = torch.from_numpy(
            np.array([np.asarray(texture) for texture in semantic_mesh.textures])).to(DEVICE)
    
        semantic_mesh_objects = (
            'uv',
            semantic_triangle_uvs,
            semantic_material_ids,
            semantic_textures
        )

    return mesh_objects, semantic_mesh_objects

def sample_semantic_pointcloud_from_uv_mesh(
        mesh_objects,
        semantic_mesh_objects,
        filtered_triangle_indices = None,
        n=500000,
        sampling_density = None,
        seed=SEED,
        visualize=False):
    
    # o3d.utility.random.seed(seed)
    # o3d.visualization.draw_geometries([mesh])

    (
        vertices,
        triangles,
        triangle_normals,
        triangle_uvs,
        material_ids,
        textures
    ) = mesh_objects

    if semantic_mesh_objects[0] == 'uv':
        _, semantic_triangle_uvs, semantic_material_ids, semantic_textures = semantic_mesh_objects

        point_triangle_indices, points, point_uvs_list, _ = sample_points_uniformly_pytorch(
            vertices, 
            triangles, 
            [triangle_uvs, semantic_triangle_uvs], 
            {}, 
            n, 
            sampling_density, 
            filtered_triangle_indices, 
            seed
            )

    else:
        _, semantic_triangle_props = semantic_mesh_objects

        point_triangle_indices, points, point_uvs_list, point_props_list = sample_points_uniformly_pytorch(
            vertices, 
            triangles, 
            [triangle_uvs],
            semantic_triangle_props, 
            n, 
            sampling_density, 
            filtered_triangle_indices, 
            seed
            )

    point_material_ids = material_ids[point_triangle_indices]

    point_colors = textures[
        point_material_ids,
        torch.clip(torch.round(textures.shape[1] * point_uvs_list[0][:, 1]).int(), 0, textures.shape[1]-1),
        torch.clip(torch.round(textures.shape[2] * point_uvs_list[0][:, 0]).int(), 0, textures.shape[2]-1),
        :
        ]

    if semantic_mesh_objects[0] == 'uv':
        semantic_point_material_ids = semantic_material_ids[point_triangle_indices]
        semantic_point_colors = semantic_textures[
            semantic_point_material_ids,
            torch.round(semantic_textures.shape[1] * point_uvs_list[1][:, 1]).int(),
            torch.round(semantic_textures.shape[2] * point_uvs_list[1][:, 0]).int(),
            :
            ]

        vertex = torch.concatenate(
            [
                points, 
                point_colors, 
                semantic_point_colors
                ], 
            dim=1)
    else:
        vertex = torch.concatenate(
            [
                points, 
                point_colors, 
                torch.vstack(point_props_list).T
                ], 
            dim=1)        

    if visualize:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.cpu().numpy()))
        pcd.colors = o3d.utility.Vector3dVector(point_colors.cpu().numpy().astype(np.float64)/255)

        o3d.visualization.draw_geometries([pcd])

    return vertex, point_triangle_indices

if __name__=="__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--mesh_path', default='habitat-matterport-3dresearch/example/hm3d-example-glb-v0.2/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.glb',
                        help="Input GLB file of the colored mesh")
    parser.add_argument('--semantic_path', default='habitat-matterport-3dresearch/example/hm3d-example-semantic-annots-v0.2/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.semantic.glb',
                        help="Input GLB file of the semantic segmentation mesh")
    parser.add_argument('--num_points', default=1000000, type=int,
                        help="Number of points to sample from mesh (uniform sampling)")
    
    args = parser.parse_args()
    
    mesh_objects, semantic_mesh_objects = load_meshes(
        args.mesh_path,
        args.semantic_path,
        ['objectId']
    )

    vertex, point_triangle_indices = sample_semantic_pointcloud_from_uv_mesh(mesh_objects, semantic_mesh_objects)

    print(vertex.shape)
    
