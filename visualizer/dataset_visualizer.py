import open3d as o3d
import csv
import numpy as np
import torch
import pandas as pd
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import argparse, json, os, itertools, random, shutil
from enum import Enum
from pathlib import Path
from helpers import get_rigid_transform, get_rigid_transform_from_bboxes


class DatasetVisualizer:

    def __init__(
            self,
            scene_path: str,
            optional_mesh_path = None,
            region_name='none',
            anchor_idx=-1,
            relationship='all',
            num_closest=3,
            num_farthest=3,
            display_bbox_labels=False,
            dont_crop_region=False
    ):
        self.scene_name = Path(scene_path).name
        self.pcd_path = os.path.join(scene_path, f'{self.scene_name}_pc_result.ply')
        self.scene_graph_path = os.path.join(scene_path, f'{self.scene_name}.json')
        self.object_csv_path = os.path.join(scene_path, f'{self.scene_name}_object_result.csv')
        self.region_csv_path = os.path.join(scene_path, f'{self.scene_name}_region_result.csv')
        self.language_path = os.path.join(scene_path, f'{self.scene_name}_label_data.json')
        self.region_split_path = os.path.join(scene_path, f'{self.scene_name}_region_split.pt')
        self.optional_mesh_path = optional_mesh_path

        self.display_bbox_labels = display_bbox_labels
        self.dont_crop_region = dont_crop_region

        if optional_mesh_path is None:
            self.pcd = o3d.t.io.read_point_cloud(self.pcd_path)
            self.material_record = rendering.MaterialRecord()
            self.material_record.point_size = 6
        else:
            triangle_model = o3d.io.read_triangle_model(optional_mesh_path)
            mesh_info = triangle_model.meshes[0]
            self.pcd = mesh_info.mesh
            self.material_record = triangle_model.materials[mesh_info.material_idx]
        
        with open(self.scene_graph_path, 'r') as f:
            self.scene_graph = json.load(f)
        
        with open(self.language_path, 'r') as f:
            self.language_queries = json.load(f)
        
        self.object_df = pd.read_csv(self.object_csv_path)
        self.region_df = pd.read_csv(self.region_csv_path)

        self.regions = self.scene_graph['regions']

        self.region_pcds = self.split_regions(os.path.join(scene_path, f'{self.scene_name}_region_split.pt'), self.pcd)

        self.region_selector_names = {}

        for region_id, region in self.regions.items():

            objects_dict = {}
            for object in region['objects']:
                object['idx'] = object['object_id']
                if object['nyu_label'] not in objects_dict:
                    objects_dict[object['nyu_label']] = [object]
                else:
                    objects_dict[object['nyu_label']].append(object)
                object['dict_idx'] = len(objects_dict[object['nyu_label']]) - 1
            region['objects_dict'] = objects_dict

            skipped_objects_dict = {}

            self.region_selector_names[f'{region["region_name"]} ({region_id})'] = int(region_id)

        self.relationship_types = [
            'all',
            'above',
            'below',
            'closest',
            'farthest',
            'between',
            'beside',
            'near',
            'in',
            'on',
            'hanging_on'
        ]

        self.num_closest = num_closest
        self.num_farthest = num_farthest

        self.bbox_colors = {
            'anchor': [1, 0, 0],
            'above': [1, 1, 0],
            'below': [1, 0, 1],
            'closest': [[1, (i+1) / 4, (i+1) / 4] for i in range(3)],
            'farthest': [[(i+1) / 4, 1, (i+1) / 4] for i in range(3)],
            'between': [0, 0, 1],
            'beside': [0, 0, 0],
            'near': [0, 1, 1],
            'in': [0, 1, 0.5],
            'on': [0, 0.5, 1],
            'hanging_on': [0, 0.5, 0.5],
        }

        # self.objects = None
        # self.object_types_dict = None
        # self.object_types = None
        # self.relationships = None
        

        print(region_name)
        self.cur_region_idx = -1 if region_name == 'none' else self.region_selector_names[region_name]
        self.cur_anchor_idx = anchor_idx
        self.cur_object_type = None
        self.cur_relationship = relationship

        self.cur_bboxes = []
        self.scene_labels = []

        # Legend

        self.legend_heights = {
            'all': 16.5 + self.num_closest + self.num_farthest,
            'above': 4.5,
            'below': 4.5,
            'closest': 4.5 + self.num_closest,
            'farthest': 4.5 + self.num_farthest,
            'between': 4.5,
            'beside': 4.5,
            'near': 4.5,
            'in': 4.5,
            'on': 4.5,
            'hanging_on': 4.5,
        }

        self.legend_height = None
        self.legend_width = None

        self.set_objects()

        self.create_window()


    def split_regions(self, region_split_path, pcd):

        region_ids_and_split = torch.load(region_split_path)
        region_split = np.cumsum(region_ids_and_split[:, 1].numpy())[:-1]
        region_ids = region_ids_and_split[:, 0].numpy()

        print(region_split, region_ids)

        positions_split = np.split(pcd.point.positions.numpy(), region_split)
        colors_split = np.split(pcd.point.colors.numpy(), region_split)

        print(pcd.point.positions.numpy())
        print(pcd.point.colors.numpy())

        region_pcds = {}
        
        for i, region_id in enumerate(region_ids):
            region_pcd = o3d.t.geometry.PointCloud()
            region_pcd.point.positions = positions_split[i]
            region_pcd.point.colors = colors_split[i]
            # print(region_pcd.point.colors)
            region_pcds[region_id] = region_pcd
        
        return region_pcds


    def set_objects(self, region_idx=None):

        if region_idx is not None:
            self.cur_region_idx = region_idx
        
        if self.cur_region_idx == -1:
            return
        
        self.objects = {obj['object_id']: obj for obj in self.regions[str(self.cur_region_idx)]['objects']}

        self.object_types_dict = self.regions[str(self.cur_region_idx)]['objects_dict']

        self.object_types = list(self.object_types_dict.keys())

        self.relationships = self.regions[str(self.cur_region_idx)]['relationships']

        self.cur_object_type = self.objects[self.cur_anchor_idx]['nyu_label'] if self.cur_anchor_idx != "-1" else None

        self.statements = self.language_queries['regions'][str(self.cur_region_idx)]
        

    def create_window(self):

        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window(f'Scene: {self.scene_name}', 1400, 900)

        self.scene_size_ratio = 0.8

        self.em = self.window.theme.font_size

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)

        self.selector = gui.Horiz(
            0.5 * self.em, 
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em))

        self.legend = gui.WidgetProxy()

        self.update_legend()

        self.populate_selector()

        self.update_widget_sizes()

        self.window.add_child(self.scene_widget)

        self.window.add_child(self.selector)

        self.window.add_child(self.legend)

        self.window.set_on_layout(self.update_widget_sizes)

        gui.Application.instance.run()

    def update_legend(self):

        self.legend.set_widget(gui.VGrid(
            2,
            0,
            gui.Margins(0.25 * self.em, 0.25 * self.em, 0.25 * self.em, 0.25 * self.em)))
    
        self.legend_height = self.legend_heights[self.cur_relationship] * self.em
        self.legend_width = 9.5 * self.em

        self.legend.add_child(gui.Label('Legend'))
        self.legend.add_child(gui.Label(''))
        for key, value in self.bbox_colors.items():
            if key == 'anchor' or self.cur_relationship == 'all' or self.cur_relationship == key:
                if key == 'farthest' or key == 'closest':
                    ordinals = ['1st', '2nd', '3rd']
                    for i in range(self.num_closest if key == 'closest' else self.num_farthest):
                        label_str = f'{ordinals[i]} {key}:'
                        label_str += ' ' * (40 - len(label_str))
                        label = gui.Label(label_str)
                        self.legend.add_child(label)
                        # colorbox = gui.ColorEdit()
                        # colorbox.color_value = gui.Color(*(value[i]))
                        color_image = (np.array(value[i])*255)[None, None, :] * np.ones((int(self.em)+6, int(self.em)+6, 1))
                        colorbox = gui.ImageWidget(o3d.geometry.Image(color_image.astype(np.uint8)))
                        self.legend.add_child(colorbox)
                else:
                    label_str = f'{key}:'
                    label_str += ' ' * (40 - len(label_str))
                    label = gui.Label(label_str)
                    self.legend.add_child(label)
                    # colorbox = gui.ColorEdit()
                    # colorbox.color_value = gui.Color(*value)
                    color_image = (np.array(value)*255)[None, None, :] * np.ones((int(self.em)+6, int(self.em)+6, 1))
                    colorbox = gui.ImageWidget(o3d.geometry.Image(color_image.astype(np.uint8)))
                    self.legend.add_child(colorbox)
        
        
        self.legend.frame = gui.Rect(
            self.window.content_rect.width - self.legend_width,
            self.window.content_rect.height * self.scene_size_ratio - self.legend_height,
            self.legend_width,
            self.legend_height,            
        )

        self.window.set_needs_layout()

    def update_widget_sizes(self, *args):
        self.scene_widget.frame = gui.Rect(
            self.window.content_rect.x,
            self.window.content_rect.y,
            self.window.content_rect.width,
            self.scene_size_ratio * self.window.content_rect.height)
        self.selector.frame = gui.Rect(
            self.window.content_rect.x,
            self.scene_widget.frame.height,
            self.window.content_rect.width,
            self.window.content_rect.height - self.scene_widget.frame.height)
        self.legend.frame = gui.Rect(
            self.window.content_rect.width - self.legend_width,
            self.window.content_rect.height * self.scene_size_ratio - self.legend_height,
            self.legend_width,
            self.legend_height,            
        )


    def populate_selector(self):

        self.region_vert = gui.CollapsableVert(
            'Region', 
            0,
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em))
        self.object_type_vert = gui.CollapsableVert(
            'Object', 
            0,
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em))
        self.object_instance_vert = gui.CollapsableVert(
            'Instance', 
            0,
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em))
        self.relationship_vert = gui.CollapsableVert(
            'Relationship', 
            0,
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em))
        self.language_vert = gui.CollapsableVert(
            'Language Queries', 
            0,
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em))

        self.region_selector = gui.ListView()
        self.object_type_selector = gui.ListView()
        self.object_instance_selector = gui.ListView()
        self.relationship_selector = gui.ListView()
        self.language_selector = gui.ListView()

        self.region_selector.set_items(list(self.region_selector_names.keys()))
        self.object_type_selector.set_items([' '*20])
        self.object_instance_selector.set_items([' '*20])
        self.relationship_selector.set_items(self.relationship_types)
        self.language_selector.set_items([' '*70])

        self.region_selector.selected_index = self.cur_region_idx
        self.draw_region()
        if self.cur_object_type is not None:
            self.object_type_selector.selected_index = self.object_types.index(self.cur_object_type)
            idxs = [str(obj['idx']) for obj in self.object_types_dict[self.cur_object_type]]
            self.object_instance_selector.set_items(idxs)
            self.object_instance_selector.selected_index = self.objects[self.cur_anchor_idx]['dict_idx']
        else:
            self.object_type_selector.selected_index = -1
            self.object_instance_selector.selected_index = -1
        self.set_language_queries()
        self.draw_bboxes()
        self.relationship_selector.selected_index = self.relationship_types.index(self.cur_relationship)

        self.region_selector.set_on_selection_changed(self.select_region)
        self.object_type_selector.set_on_selection_changed(self.select_object_type)
        self.object_instance_selector.set_on_selection_changed(self.select_object_instance)
        self.relationship_selector.set_on_selection_changed(self.select_relationship)
        self.language_selector.set_on_selection_changed(self.select_language_query)

        self.region_vert.add_child(self.region_selector)
        self.object_type_vert.add_child(self.object_type_selector)
        self.object_instance_vert.add_child(self.object_instance_selector)
        self.relationship_vert.add_child(self.relationship_selector)
        self.language_vert.add_child(self.language_selector)

        self.selector.add_child(self.region_vert)
        self.selector.add_child(self.object_type_vert)
        self.selector.add_child(self.object_instance_vert)
        self.selector.add_child(self.relationship_vert)
        self.selector.add_child(self.language_vert)
    
    def select_region(self, new_val: str, is_dbl_click):
        self.cur_region_idx = self.region_selector_names[new_val]
        self.cur_anchor_idx = "-1"
        self.set_objects()
        self.object_type_selector.set_items(self.object_types)
        self.object_instance_selector.set_items([' '*20])
        self.draw_region()
        self.draw_bboxes()
        self.set_language_queries()
        
    def select_object_type(self, new_val: str, is_dbl_click):
        idxs = [' '*10 + str(obj['idx']) + ' '*10 for obj in self.object_types_dict[new_val]]
        self.object_instance_selector.set_items(idxs)
        self.object_instance_selector.selected_index = 0
        self.draw_bboxes(anchor_idx=idxs[0].strip())
        self.set_language_queries()

        
    def select_object_instance(self, new_val: str, is_dbl_click):
        self.draw_bboxes(anchor_idx=new_val.strip())
        self.set_language_queries()

    def select_relationship(self, new_val, is_dbl_click):
        self.draw_bboxes(relationship=new_val)
        self.set_language_queries()
        self.update_legend()
    
    def set_language_queries(self):
        if self.cur_region_idx == -1:
            return
        filtered_statements = []
        for statement, instances in self.statements.items():
            if statement=="region":
                continue
            for instance in instances:
                if self.cur_relationship == 'all' or self.cur_relationship == instance['relation']:
                    if self.cur_anchor_idx == "-1" \
                    or instance['relation_type'] == 'ternary' and instance['target_index'] == self.cur_anchor_idx \
                    or instance['relation_type'] == 'binary' and instance['anchor_index'] == self.cur_anchor_idx:
                        filtered_statements.append(statement)
                        break

        self.language_selector.set_items([statement + ' '*10 for statement in filtered_statements])
        self.window.set_needs_layout()


    def select_language_query(self, new_val, is_dbl_click):
        instances = self.statements[new_val.strip()]
        relation_type = instances[0]['relation_type']
        new_relation = instances[0]['relation']
        for instance in instances:
            if relation_type == 'ternary':
                anchor_idx = instance['target_index']
                target_idx = (instance['anchor1_index'], instance['anchor2_index'])
                break
            elif relation_type == 'binary':
                anchor_idx = instance['anchor_index']
                target_idx = instance['target_index']
                break
        self.draw_bboxes(anchor_idx=anchor_idx, relationship=new_relation, target_idx=target_idx)
        self.update_legend()


    def add_object_bbox(self, object_idx, color, line_width=10):

        if self.scene_widget.scene.has_geometry(str(object_idx)):
            return

        object = self.objects[object_idx]

        name = object['nyu_label']

        corners_3d = np.array(object["bbox"])

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        colors = [color for _ in range(len(bbox_lines))]

        bbox = o3d.geometry.LineSet()
        bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
        bbox.colors = o3d.utility.Vector3dVector(colors)
        bbox.points = o3d.utility.Vector3dVector(corners_3d)

        center = object["center"]

        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = line_width

        self.scene_widget.scene.add_geometry(str(object_idx), bbox, mat)
        if self.display_bbox_labels:
            self.scene_labels.append(self.scene_widget.add_3d_label(center, f'{name} {object_idx}'))

        self.cur_bboxes.append(str(object_idx))

    def draw_region(self):

        self.scene_widget.scene.clear_geometry()

        self.cur_bboxes = []

        if self.cur_region_idx != -1:

            self.scene_widget.scene.add_geometry('pcd', self.region_pcds[self.cur_region_idx], self.material_record)

            bounds = self.region_pcds[self.cur_region_idx].get_axis_aligned_bounding_box()
        else:
            self.scene_widget.scene.add_geometry('pcd', self.pcd, self.material_record)

            bounds = self.pcd.get_axis_aligned_bounding_box()

        # TODO: change initial position to something more suitable
        # w = self.window.content_rect.width
        # h = self.window.content_rect.height
        # fov = 60
        # fy, fx = 1 / (2*np.tan(fov/2)), 1 / (2*np.tan(fov/2))
        # intrinsics = o3d.camera.PinholeCameraIntrinsic(
        #     w, 
        #     h,
        #     np.array([
        #         [fx, 0, 0],
        #         [0, fy, 0],
        #         [0, 0, 1]
        #     ])
        # )
        # rotation = np.array([
        #     [ 0.5000000,  0.0000000,  0.8660254 ],
        #     [0.0000000,  1.0000000,  0.0000000],
        #     [-0.8660254,  0.0000000,  0.5000000 ],
        # ])
        # translation = (bounds.get_extent().numpy() / 3).reshape(3, 1)
        # extrinsics = np.hstack([rotation, translation])
        # extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1]])])
        # print(extrinsics.shape)
        # extrinsics = np.array([
        #     [-0.99597436, -0.08963859,  0.,          2.2625608 ],
        #     [ 0.038237,   -0.4248515,   0.9044551,  -2.433831  ],
        #     [-0.08107408,  0.9008142,   0.42656872,  0.8896539 ],
        #     [ 0.,   0.,          0.,          1.        ]
        #     ])

        if isinstance(self.pcd, o3d.t.geometry.PointCloud):
            self.scene_widget.setup_camera(60, bounds.to_legacy(), bounds.get_center().numpy())
        else:
            self.scene_widget.setup_camera(60, bounds, bounds.get_center())
        # self.scene_widget.scene.camera.look_at(
        #     bounds.get_center().numpy(),
        #     np.array([-np.sqrt(2)/2, 0., np.sqrt(2)/2]),
        #     np.array([np.sqrt(2)/2, 0., -np.sqrt(2)/2])
        #     )


    def draw_bboxes(self, anchor_idx = None, relationship = None, target_idx = None):

        if self.cur_region_idx == -1:
            return

        for bbox in self.cur_bboxes:
            self.scene_widget.scene.remove_geometry(bbox)
        
        if self.display_bbox_labels:
            for label in self.scene_labels:
                self.scene_widget.remove_3d_label(label)
            self.scene_labels = []

        if anchor_idx is not None:
            self.cur_anchor_idx = anchor_idx
        if relationship is not None:
            self.cur_relationship = relationship
            self.relationship_selector.selected_index = self.relationship_types.index(self.cur_relationship)

        if self.cur_anchor_idx == "-1":
            for i in self.objects.keys():
                self.add_object_bbox(i, self.bbox_colors['anchor'], line_width=5)
        else:
            self.add_object_bbox(self.cur_anchor_idx, self.bbox_colors['anchor'])

            for binary_relationship in ['above', 'below', 'near', 'in', 'on', 'hanging_on']:

                if self.cur_relationship == binary_relationship or self.cur_relationship == 'all':
                    above_idxs = self.relationships[binary_relationship][str(self.cur_anchor_idx)]
                    if target_idx is not None:
                        self.add_object_bbox(target_idx, self.bbox_colors[binary_relationship])
                    else:
                        for idx in above_idxs:
                            self.add_object_bbox(idx, self.bbox_colors[binary_relationship])

            if self.cur_relationship == 'closest' or self.cur_relationship == 'all':
                closest_idxs = self.relationships["closest"][str(self.cur_anchor_idx)][:self.num_closest]
                if target_idx is not None:
                    order = closest_idxs.index(target_idx)
                    self.add_object_bbox(target_idx, self.bbox_colors['closest'][order])                    
                else:
                    for i in range(self.num_closest):
                        self.add_object_bbox(closest_idxs[i], self.bbox_colors['closest'][i])

            if self.cur_relationship == 'farthest' or self.cur_relationship == 'all':
                farthest_idxs = self.relationships["farthest"][str(self.cur_anchor_idx)][:self.num_farthest]
                if target_idx is not None:
                    order = farthest_idxs.index(target_idx)
                    self.add_object_bbox(target_idx, self.bbox_colors['farthest'][order])                      
                else:
                    for i in range(self.num_farthest):
                        self.add_object_bbox(farthest_idxs[i], self.bbox_colors['farthest'][i])

            if self.cur_relationship == 'between' or self.cur_relationship == 'all':
                between_idxs = self.relationships["between"][str(self.cur_anchor_idx)]
                if target_idx is not None:
                    self.add_object_bbox(target_idx[0], self.bbox_colors['between'])
                    self.add_object_bbox(target_idx[1], self.bbox_colors['between'])
                else:
                    for idx in between_idxs:
                        self.add_object_bbox(idx[0], self.bbox_colors['between'])
                        self.add_object_bbox(idx[1], self.bbox_colors['between'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--scene_path', required=True, 
                        help="Path to scene folder containing .ply file, scene graph JSON, and language query JSON")
    parser.add_argument('--optional_mesh_path', 
                        help="Optional path to mesh file for easier visualization")
    parser.add_argument('--region', default='none',
                        help="Initial region of scene to visualize (default is none, will pick first region)")
    parser.add_argument('--anchor_idx', default='-1',
                        help="Initial index for the anchor object (default is no index chosen)")
    parser.add_argument('--relationship', default='all',
                        help="relationship to visualize (default is all)")
    parser.add_argument('--num_closest', default='3', type=int,
                        help="Number of closest objects to visualize")
    parser.add_argument('--num_farthest', default='3', type=int,
                        help="Number of farthest objects to visualize")
    parser.add_argument('--display_bbox_labels', action='store_const', const=True, default=False,
                        help="Display labels on bounding boxes directly")
    parser.add_argument('--dont_crop_region', action='store_const', const=True, default=False,
                        help="Do not crop region on selection")

    args = parser.parse_args()

    dataset_visualizer = DatasetVisualizer(
        args.scene_path,
        args.optional_mesh_path,
        args.region,
        args.anchor_idx,
        args.relationship,
        args.num_closest,
        args.num_farthest,
        args.display_bbox_labels,
        args.dont_crop_region
    )
