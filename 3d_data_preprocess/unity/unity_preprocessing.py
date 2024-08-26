import open3d as o3d
import numpy as np
import argparse
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy import stats
from collections import Counter
import fbx
import math
import cv2
import os
import csv
import json
import matplotlib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.freespace_generation_new import generate_free_space
from utils.transformations import rotz
from utils.glb_to_pointcloud import sample_points_uniformly_pytorch
from utils.dominant_colors_new_lab import judge_color, generate_color_anchors
from utils.bbox_utils import calculate_bbox_hull
from utils.headers import OBJECT_HEADER, REGION_HEADER
from utils.pointcloud_utils import save_pointcloud, write_ply_file, sort_pointcloud, get_regions, get_objects
import torch
import warnings

warnings.simplefilter('error')

class UnityPreprocessor:
    def __init__(
        self,
        in_path: str, # the path containing scan files
        num_points_per_object: int, # number of points to sample per object
        out_path: str, # the path to store the processed results
        floor_height: float, # the floor height for generating free space
        color_standard: str, # color standard to use for domain color calculation (css21, css3, html4, css2)
        name_list: list,
        sampling_density = None,
        generate_freespace = False, # whether to generate free space or not
        device='cuda',
        seed=42
    ):
        
        self.in_path = in_path 
        self.num_points_per_object = num_points_per_object
        self.out_path = os.path.join(out_path, 'Unity')
        self.floor_height = floor_height
        self.anchor_colors_array, self.anchor_colors_array_hsv, self.anchor_colors_name = generate_color_anchors(color_standard)
        self.tree = KDTree(self.anchor_colors_array_hsv)

        self.generate_freespace = generate_freespace
        self.name_list = name_list
        self.device = 'cuda' if (device=='cuda' and torch.cuda.is_available()) else 'cpu'
        self.seed = seed
        self.sampling_density = sampling_density

        
    def compute_box_3d(self, center, size, heading): # function for calculating the corner points of a bbox based on its center, size and heading

        h = float(size[2])
        w = float(size[1])
        l = float(size[0])
        # heading_angle = -heading_angle - np.pi / 2

        # center[2] = center[2] + h / 2
        R = rotz(1*heading)
        l = l/2
        w = w/2
        h = h/2
        x_corners = [-l,l,l,-l,-l,l,l,-l]
        y_corners = [w,w,-w,-w,w,w,-w,-w]
        z_corners = [h,h,h,h,-h,-h,-h,-h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] += center[0]
        corners_3d[1,:] += center[1]
        corners_3d[2,:] += center[2]
        return np.transpose(corners_3d)

    def find_region(self, points, faces): # function for finding the region this current object belongs to
        tmp_points = points[:, [2, 0, 1]]
        if self.scan_name == 'arabic_room':
            tmp_points[:, 0] += -7.9 # manually shift
            tmp_points[:, 1] += 11.43 # manually shift
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(tmp_points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        tmp_points = np.asarray(mesh.sample_points_uniformly(100).points)
        tmp_region_ids = []
        for center in tmp_points:
            flag = 0
            for key in self.region_dict.keys():
                limit = self.region_dict[key]
                if (center >= limit[1, :]).all() and (center <= limit[0, :]).all():
                    tmp_region_ids.append(key)
                    flag = 1
                    break
            if flag == 0:
                tmp_region_ids.append(-1)
        return Counter(tmp_region_ids).most_common(1)[0][0]

    def calculate_bbox_and_top3_color3(self, pcd, target_obj_id): # given the points, calculate the bbox of the target object and its dominant color

        object_ids = pcd[:, 6]
        pcd_points = pcd[:, :3]
        pcd_colors = pcd[:, 3:6]
        index, = np.where(object_ids.flatten() ==  target_obj_id)
        if len(index) == 0:
            raise Exception("Sample more points for covering all objects!")

        object_points = pcd_points[index, :]
        center, size, heading = calculate_bbox_hull(object_points)

        colors = pcd_colors[index, :]/255
        color_3 = judge_color(colors, self.tree, self.anchor_colors_array, self.anchor_colors_name)
        
        bbox = []
        bbox += center
        bbox += size
        bbox.append(heading)

        return bbox, color_3

    def DisplayTextureNames(self, pProperty): # loading the textures
        lTextureName = []
        
        lLayeredTextureCount = pProperty.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxLayeredTexture.ClassId))
        if lLayeredTextureCount > 0:
            for j in range(lLayeredTextureCount):
                lLayeredTexture = pProperty.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxLayeredTexture.ClassId), j)
                lNbTextures = lLayeredTexture.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId))

                for k in range(lNbTextures):
                    if self.scan_name == 'home_building_1':
                        lTextureName.append(os.path.join(*Path(lLayeredTexture.GetRelativeFileName()).parts[1:]))
                    else:
                        lTextureName.append(os.path.join('Textures', f'{lLayeredTexture.GetName()}.png'))
        else:
            #no layered texture simply get on the property
            lNbTextures = pProperty.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId))

            if lNbTextures > 0:

                for j in range(lNbTextures):
                    lTexture = pProperty.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxTexture.ClassId),j)
                    if lTexture:
                        if self.scan_name == 'home_building_1':
                            lTextureName.append(os.path.join(*Path(lTexture.GetRelativeFileName()).parts[1:]))
                        else:
                            lTextureName.append(os.path.join('Textures', f'{lTexture.GetName()}.png'))

                
        return lTextureName

    def readuv(self, pLayerObject, vertexIndex): # read the uv mapping

        if pLayerObject == None:
            return

        # print(pLayerObject.GetMappingMode())
        # print(pLayerObject.GetMappingMode())
        if pLayerObject.GetMappingMode() in [
            fbx.FbxLayerElement.EMappingMode.eByControlPoint, 
            fbx.FbxLayerElement.EMappingMode.eByPolygonVertex
            ]:
            if pLayerObject.GetReferenceMode() == fbx.FbxLayerElement.EReferenceMode.eDirect:
                fbxuv = pLayerObject.GetDirectArray().GetAt(vertexIndex)
                u = fbxuv[0]
                v = fbxuv[1]
            elif pLayerObject.GetReferenceMode() == fbx.FbxLayerElement.EReferenceMode.eIndexToDirect:
                id = pLayerObject.GetIndexArray().GetAt(vertexIndex)
                fbxuv = pLayerObject.GetDirectArray().GetAt(id)
                u = fbxuv[0]
                v = fbxuv[1]
        else:
            raise Exception("Wrong type!")

        return u, v

    def readmaterial(self, pLayerObject, polygonIndex): # read the material

        if pLayerObject == None:
            raise Exception("Empty object!")
        
        # print(pLayerObject.GetMappingMode())
        if pLayerObject.GetMappingMode() in [
            fbx.FbxLayerElement.EMappingMode.eByPolygon,
            fbx.FbxLayerElement.EMappingMode.eAllSame]:
            material_id = pLayerObject.GetIndexArray().GetAt(polygonIndex)
        else:
            raise Exception("Wrong type!")

        return material_id

    def multT(self, node, vector): # calculating the position in gobal coordinate
        matrixGeo = fbx.FbxAMatrix()
        matrixGeo.SetIdentity()
        if (node.GetNodeAttribute()):
            lT = node.GetGeometricTranslation(fbx.FbxNode.EPivotSet(0))
            lR = node.GetGeometricRotation(fbx.FbxNode.EPivotSet(0))
            lS = node.GetGeometricScaling(fbx.FbxNode.EPivotSet(0))
            matrixGeo.SetTRS(lT, lR, lS)
        globalMatrix = node.EvaluateGlobalTransform()

        matrix = globalMatrix*matrixGeo
        result = matrix.MultT(vector)
        return result

    def read_node(self, child_node): # recursively read the nodes in fbx format
        self.node_id += 1
        if child_node.GetNodeAttribute() and (str(child_node.GetNodeAttribute().GetAttributeType()) == 'EType.eMesh' or str(child_node.GetNodeAttribute().GetAttributeType()) == '4') and child_node.GetName().split('_')[0] != 'Shape':
            mesh = child_node.GetNodeAttribute().GetLayer(0)
            tmp_uvs = mesh.GetUVs()
            tmp_materials = mesh.GetMaterials()
            vertices = child_node.GetNodeAttribute().GetControlPoints()
            points = np.zeros((len(vertices),3))
            for i, vertex in enumerate(vertices):
                vertex = self.multT(child_node, vertex)
                points[i, 0] = vertex[0]
                points[i, 1] = vertex[1]
                points[i, 2] = vertex[2]
            self.vertices_global.append(points)
            faces = child_node.GetNodeAttribute().GetPolygonVertices()
            faces = np.array(faces).reshape(-1,3)
            uvs = np.zeros((3*len(faces),2))
            node_ids = np.zeros((len(faces),1))
            self.object_names_map[self.count] = str(child_node.GetName()).strip()
            if self.scan_name == 'home_building_1':
                self.object_names_map[self.count] = self.object_names_map[self.count].split('_')[0]
            self.object_ids_global += [self.count]*len(faces)
            self.object_region_map[self.count] = self.find_region(points, faces)
            for i in range(len(faces)):
                node_ids[i, 0] = self.node_id
                assert child_node.GetNodeAttribute().GetPolygonSize(i) == 3
                for j in range(3):
                    if tmp_uvs.GetMappingMode() == fbx.FbxLayerElement.EMappingMode.eByControlPoint:
                        lControlPointIndex = child_node.GetNodeAttribute().GetPolygonVertex(i, j)
                    elif tmp_uvs.GetMappingMode() == fbx.FbxLayerElement.EMappingMode.eByPolygonVertex:
                        lControlPointIndex = child_node.GetNodeAttribute().GetTextureUVIndex(i, j)
                    u, v = self.readuv(tmp_uvs, lControlPointIndex)
                    uvs[3*i+j, 0] = u
                    uvs[3*i+j, 1] = v
                material_id = self.readmaterial(tmp_materials, i)
                lMaterial = child_node.GetMaterial(material_id)
                names = []
                if lMaterial:
                    lProperty = lMaterial.FindProperty(fbx.FbxSurfaceMaterial.sDiffuse)
                    names = self.DisplayTextureNames(lProperty)
                if names == []:
                    names = ['None']
                for name in names:
                    count_texture = self.texture_names.count(name)
                    if count_texture == 1:
                        index = self.texture_names.index(name)
                        self.texture_global.append(index)
                    elif count_texture == 0:
                        self.texture_global.append(len(self.texture_names))
                        #texture_local.add(material_id)
                        self.texture_names.append(name)
                    else:
                        raise Exception('Repeated texture')
            self.uvs_global.append(uvs)
            faces = faces + self.n_nodes
            self.faces_global.append(faces)
            self.n_nodes += len(points)
            self.count += 1
        else:
            self.count_1 += 1

        for i in range(child_node.GetChildCount()):
            self.read_node(child_node.GetChild(i))


    def convert_fbx_to_pointcloud( # sample points to convert fbx to point cloud
            self,
            path, 
            out_path):

        vertices = np.vstack(self.vertices_global)
        triangles = np.vstack(self.faces_global)
        triangle_uvs = np.vstack(self.uvs_global).reshape(-1, 3, 2)
        material_ids = np.array(self.texture_global).astype(np.int32)

        textures = [[] for _ in range(len(self.texture_names))]
        number = 0
        texture_flag = 0
        none_texture_remaining = []
        for i in range(len(self.texture_names)):
            if not self.texture_names[i] == 'None':
                print(self.texture_names[i])
                texture=cv2.imread(os.path.join(path, self.texture_names[i]))
                if self.scan_name != 'home_building_1':
                    texture = texture[::-1, :, ::-1]
                else:
                    texture = texture[:, :, ::-1]
                number += 1
                texture_flag = 1
                available_texture_index = i
            else:
                if texture_flag:
                    texture = np.zeros(cv2.imread(os.path.join(path, self.texture_names[available_texture_index])).shape)
                else:
                    none_texture_remaining.append(i)
                    continue
            textures[i] = texture[:, :, 0:3]
        
        for i in range(len(none_texture_remaining)):
            texture = np.zeros(cv2.imread(os.path.join(path, self.texture_names[available_texture_index])).shape)
            textures[none_texture_remaining[i]] = texture[:, :, 0:3]
        

        pt_triangle_indices, pcd_vertices, pcd_uvs, _ = sample_points_uniformly_pytorch(
            torch.from_numpy(vertices).to(self.device),
            torch.from_numpy(triangles).to(self.device),
            [torch.from_numpy(triangle_uvs).to(self.device)],
            num_points=int(self.num_points_per_object*self.count),
            sampling_density=self.sampling_density,
            device=self.device,
            seed=self.seed
        )

        pt_triangle_indices, pcd_vertices, pcd_uvs = pt_triangle_indices.cpu().numpy(), pcd_vertices.cpu().numpy(), pcd_uvs[0].cpu().numpy()
        print(int(self.num_points_per_object*self.count))
        num_points = len(pt_triangle_indices)

        pcd_colors = []
        object_ids = []
        region_ids = []

        for i in tqdm(range(num_points)):
            object_id = self.object_ids_global[int(pt_triangle_indices[i])]
            object_ids.append(object_id)
            region_id = self.object_region_map[object_id]
            region_ids.append(region_id)
            material = material_ids[int(pt_triangle_indices[i])]
            texture = textures[material]
            x = int((texture.shape[1]-1) * (pcd_uvs[i, 0]%1))
            y = int((texture.shape[0]-1) * (pcd_uvs[i, 1]%1))
            color = texture[y, x, :]
            pcd_colors.append(color)

        pcd_vertices[:, [0, 1, 2]] = pcd_vertices[:, [2, 0, 1]]   
        if self.scan_name == 'arabic_room':
            pcd_vertices[:, 0] += -7.9 # manually shift
            pcd_vertices[:, 1] += 11.43 # manually shift

        pcd_colors = np.array(pcd_colors)

        vertex = np.concatenate([
            pcd_vertices,
            pcd_colors,
            np.array(object_ids)[:, None],
            np.array(region_ids)[:, None]
        ], axis=1)

        vertex, region_indices_out, object_indices_out = sort_pointcloud(torch.from_numpy(vertex))

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        save_pointcloud(vertex, region_indices_out, object_indices_out, out_path, self.scan_name)

        return vertex.cpu().numpy()

    def create_unity(self):
        name_list = self.name_list
        for scan_name in name_list:
            self.scan_name = scan_name
            self.vertices_global = []
            self.faces_global = []
            self.texture_global = []
            self.texture_names = []
            self.object_ids_global = []
            self.object_names_map = {}
            self.normals_global = []
            self.uvs_global = []
            self.region_dict = {}
            self.region_names = []
            self.object_region_map = {}
            self.unity_cat_mapping = {}
            self.n_nodes = 0
            self.node_id = -1

            cate_map_file_name = os.path.join(args.in_path, 'labels', 'Categories_' + scan_name + '.csv')
            with open(cate_map_file_name, 'r', newline='') as f_map:
                csv_reader = csv.reader(f_map)
                for row in csv_reader:
                    self.unity_cat_mapping[row[0].strip()] = {'cleaned_label': row[1].strip(), 'nyuid':row[2].strip(), 'nyu40id':row[3].strip(), 'nyuclass':row[4].strip(), 'nyu40class':row[5].strip()}

            print(os.path.join(args.in_path, 'regions', scan_name + '_region.json'))
            f = open(os.path.join(args.in_path, 'regions', scan_name + '_region.json'), 'r')
            content = f.read()
            region_json = json.loads(content)
            f.close()

            for region_index, region in enumerate(region_json['objects']):
                region_name = region['name']
                region_center = np.array([region['centroid']['x'], region['centroid']['y'], region['centroid']['z']]) # have a rotation difference, need to switch axis
                region_size = np.array([region['dimensions']['length'], region['dimensions']['width'], region['dimensions']['height']]) # have a rotation difference, need to switch axis
                region_heading = region['rotations']['z']/360*2*np.pi # have a rotation difference, need to switch axis
                region_corners_3d = self.compute_box_3d(region_center, region_size, region_heading)
                region_limits = np.vstack((np.max(region_corners_3d, axis=0), np.min(region_corners_3d, axis=0)))
                self.region_names.append(region_name)
                self.region_dict[region_index] = region_limits

            manager = fbx.FbxManager.Create()
            importer = fbx.FbxImporter.Create(manager, '')
            importer.Initialize(os.path.join(args.in_path, 'meshes', scan_name, 'Infrastructure.fbx'))
            scene = fbx.FbxScene.Create(manager, 'scene')
            importer.Import(scene)
            importer.Destroy()
            root_node = scene.GetRootNode()

            self.count = 0
            self.count_1 = 0
            if root_node:
                for i in range(root_node.GetChildCount()):
                    child_node = root_node.GetChild(i)
                    self.read_node(child_node)
            print(scan_name, "- valid", self.count, "invalid", self.count_1)
            scale = np.max(np.vstack(self.vertices_global), axis=0) - np.min(np.vstack(self.vertices_global), axis=0)
            volume = scale[0]*scale[1]*scale[2]
            dis_thred = 0.01*203.74/volume
            area_thred = 0.01*203.74/volume
            pcd = self.convert_fbx_to_pointcloud(
                os.path.join(args.in_path, 'meshes', scan_name), os.path.join(self.out_path, scan_name)
            )

            out_dir = os.path.join(self.out_path, scan_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            os.chdir(out_dir)

            region_file_name = scan_name + '_region_result.csv'
            with open(region_file_name, 'w', newline='') as f_region:
                region_writer = csv.writer(f_region, delimiter=',')
                region_writer.writerow(REGION_HEADER)
                for region_id, region in enumerate(tqdm(region_json['objects'])):
                    region_label = region['name']
                    region_center = np.array([region['centroid']['x'], region['centroid']['y'], region['centroid']['z']]) # have a rotation difference, need to switch axis
                    region_size = np.array([region['dimensions']['length'], region['dimensions']['width'], region['dimensions']['height']]) # have a rotation difference, need to switch axis
                    region_heading = region['rotations']['z']/360*2*np.pi # have a rotation difference, need to switch axis
                    region_info = []

                    region_info.append(region_id)
                    region_info.append(region_label)
                    region_info += list(region_center)
                    region_info += list(region_size)
                    region_info.append(region_heading)
                    region_writer.writerow(region_info)


            object_lines = []
            floor_sizes = []
            for i in tqdm(range(self.count)):
                if (i == 78 or i == 79 or i == 80 or i == 87 or i == 88) and scan_name == 'office_3':
                    continue
                object_bbox, object_color3 = self.calculate_bbox_and_top3_color3(pcd, i)
                object_line = [i]
                object_line.append(self.object_region_map[i])
                object_line += [self.unity_cat_mapping[self.object_names_map[i]]['cleaned_label']]
                object_line += [self.unity_cat_mapping[self.object_names_map[i]]['nyuid']]
                object_line += [self.unity_cat_mapping[self.object_names_map[i]]['nyu40id']]
                object_line += [self.unity_cat_mapping[self.object_names_map[i]]['nyuclass']]
                object_line += [self.unity_cat_mapping[self.object_names_map[i]]['nyu40class']]
                if self.unity_cat_mapping[self.object_names_map[i]]['nyu40class'] == 'floor' or self.object_names_map[i] == 'Floor structure':
                    floor_sizes.append(object_bbox + [self.object_region_map[i]])

                object_line += object_bbox
                object_line.append('_')
                object_line += object_color3
                object_lines.append(object_line)

            with open(scan_name + '_object_result.csv', 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(OBJECT_HEADER)
                for i in object_lines:
                    writer.writerow(i)

            with open('object_list.txt', 'w', newline='') as f:
                for i in object_lines:
                    f.write(str(i[0])+' '+' '.join([str(a) for a in i[7:14]])+ ' ' + '\"' + str(i[2] + '\"' + '\n'))

            # print(pcd)
            # ply_file_name = scan_name + '_pc_result.ply'
            # if args.out_path != 'none':
            #     o3d.t.io.write_point_cloud(ply_file_name, pcd)
            # else:
            #     o3d.visualization.draw_geometries([pcd])

            scene.Destroy()
            manager.Destroy()

            if self.generate_freespace:
                point_positions = pcd[:, :3]
                point_region_ids = pcd[:, 7]
                generate_free_space(scan_name, point_positions, floor_sizes, point_region_ids, self.floor_height)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_path', default='/home/navigation/Dataset/unity',
                        help="Input FBX file of the mesh")
    parser.add_argument('--num_points_per_object', default=100000, type=int,
                        help="Number of points to sample from mesh (uniform sampling)")
    parser.add_argument('--out_path', default='/home/navigation/Dataset/VLA_Dataset',
                        help="Output PLY file to save")
    parser.add_argument('--floor_height', default=0.1,
                        help="floor heigh for generating free space")
    parser.add_argument('--color_standard', default='css3',
                        help="color standard, chosen from css2, css21, css3, html4")
    parser.add_argument('--sampling_density', type=float, default=None)
    parser.add_argument('--generate_freespace', action='store_true', help='Generate free spaces')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)    

    args = parser.parse_args()

    # name_list = [
    #     'arabic_room', 
    #     'chinese_room', 
    #     'home_building_1', 
    #     'home_building_2', 
    #     'home_building_3', 
    #     'hotel_room_1', 
    #     'hotel_room_2', 
    #     'hotel_room_3', 
    #     'japanese_room', 
    #     'livingroom_1', 
    #     'livingroom_2', 
    #     'livingroom_3', 
    #     'livingroom_4', 
    #     'loft', 
    #     'office_1', 
    #     'office_2', 
    #     'office_3', 
    #     'studio'
    #     ]
    name_list = ['home_building_1']
    print('====================Start processing unity====================')
    UnityPreprocessor(
        args.in_path, 
        args.num_points_per_object, 
        args.out_path, 
        args.floor_height, 
        args.color_standard, 
        name_list, 
        args.sampling_density,
        args.generate_freespace,
        args.device,
        args.seed
        ).create_unity()
    print('====================End processing unity====================')