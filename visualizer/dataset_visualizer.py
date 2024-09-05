import open3d as o3d
import csv
import numpy as np
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
            target_idx=-1,
            relationship='all',
            display_labels_and_colors=True,
            show_freespace_annotations = False,
            colored_color_labels=True,
            dont_crop_region=False,
            hide_legend=False
    ):
        self.scene_name = Path(scene_path).name
        self.pcd_path = os.path.join(scene_path, f'{self.scene_name}_pc_result.ply')
        self.freespace_pcd_path = os.path.join(scene_path, f'{self.scene_name}_free_space_pc_result.ply')
        self.scene_graph_path = os.path.join(scene_path, f'{self.scene_name}_scene_graph.json')
        self.object_csv_path = os.path.join(scene_path, f'{self.scene_name}_object_result.csv')
        self.region_csv_path = os.path.join(scene_path, f'{self.scene_name}_region_result.csv')
        self.language_path = os.path.join(scene_path, f'{self.scene_name}_referential_statements.json')
        self.region_split_path = os.path.join(scene_path, f'{self.scene_name}_region_split.npy')
        self.optional_mesh_path = optional_mesh_path
        self.hide_legend = hide_legend

        self.display_labels_and_colors = display_labels_and_colors
        self.show_freespace_annotations = show_freespace_annotations
        self.colored_color_labels = colored_color_labels
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
        
        # Free space pointcloud
        if os.path.exists(self.freespace_pcd_path):
            self.freespace_pcd = o3d.io.read_point_cloud(self.freespace_pcd_path)
            self.freespace_color = np.array([[1., 0., 0.]])
            self.freespace_pcd.colors = o3d.utility.Vector3dVector((self.freespace_color * np.ones_like(np.array(self.freespace_pcd.points))).astype(np.float32))
            self.freespace_material_record = rendering.MaterialRecord()
            self.freespace_material_record.point_size = 10
        
        # Scene Graph
        with open(self.scene_graph_path, 'r') as f:
            self.scene_graph = json.load(f)
        
        # Referential Statements
        with open(self.language_path, 'r') as f:
            self.language_queries = json.load(f)
        
        self.object_df = pd.read_csv(self.object_csv_path)
        self.region_df = pd.read_csv(self.region_csv_path)

        self.regions = self.scene_graph['regions']

        self.region_pcds = self.split_regions(self.region_split_path, self.pcd)

        self.region_selector_names = {}

        for region_id, region in self.regions.items():

            objects_dict = {}
            for object in region['objects']:
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
            'second_closest',
            'third_closest',
            'farthest',
            'second_farthest',
            'third_farthest',
            'between',
            'near',
            'in',
            'on',
            'hanging_on'
        ]

        self.bbox_colors = {
            'target': [0, 1, 0],
            'distractor': [1, 0, 0],
            'anchor': [0, 0, 1]
        }

        self.cur_region_idx = -1 if region_name == 'none' else self.region_selector_names[region_name]
        self.cur_target_idx = target_idx
        self.cur_object_type = None
        self.cur_relationship = relationship

        self.cur_bboxes = []
        self.scene_labels = []

        self.set_objects()

        self.create_window()


    def split_regions(self, region_split_path, pcd):

        region_ids_and_split = np.load(region_split_path)
        region_split = region_ids_and_split[:-1, 1]
        region_ids = region_ids_and_split[:, 0]

        positions_split = np.split(pcd.point.positions.numpy(), region_split)
        colors_split = np.split(pcd.point.colors.numpy(), region_split)


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

        self.object_types = ['all'] + sorted(list(self.object_types_dict.keys()))

        self.relationships = self.regions[str(self.cur_region_idx)]['relationships']

        self.relationship_targets = {}
        for rel in self.relationship_types:
            if rel == "all":
                pass
            elif rel == "between":
                self.relationship_targets[rel] = self.relationships[rel]
            else:
                self.relationship_targets[rel] = {obj_id: [] for obj_id in self.relationships[rel].keys()}
                for anchor, targets in self.relationships[rel].items():
                    for target in targets:
                        self.relationship_targets[rel][target].append([anchor])
        

        self.cur_object_type = self.objects[self.cur_target_idx]['nyu_label'] if self.cur_target_idx != "-1" else None

        self.statements = self.language_queries['regions'][str(self.cur_region_idx)]
        

    def create_window(self):

        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window(f'Scene: {self.scene_name}', 1920, 1080)

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

        self.create_options()

        self.update_widget_sizes()

        self.window.add_child(self.scene_widget)

        self.window.add_child(self.selector)

        self.window.add_child(self.legend)

        self.window.set_on_layout(self.update_widget_sizes)

        gui.Application.instance.run()

    def update_legend(self):
        
        if self.hide_legend:
            return

        self.legend.set_widget(gui.VGrid(
            2,
            0,
            gui.Margins(0.25 * self.em, 0.25 * self.em, 0.25 * self.em, 0.25 * self.em)))
    
        self.legend_height = 6.5 * self.em
        self.legend_width = 9.5 * self.em

        self.legend.add_child(gui.Label('Legend'))
        self.legend.add_child(gui.Label(''))
        for key, value in self.bbox_colors.items():
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
        if not self.hide_legend:
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
            idxs = [str(obj['object_id']) for obj in self.object_types_dict[self.cur_object_type]]
            self.object_instance_selector.set_items(idxs)
            self.object_instance_selector.selected_index = self.objects[self.cur_target_idx]['dict_idx']
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
    

    def create_options(self):
        self.options_vert = gui.CollapsableVert(
            'Visualization Options', 
            0,
            gui.Margins(0.5 * self.em, 0.5 * self.em, 0.5 * self.em, 0.5 * self.em))
        
        self.show_colors_cb = gui.Checkbox("Show color labels")
        self.show_colors_cb.checked = self.display_labels_and_colors
        self.show_colors_cb.set_on_checked(self.show_color_labels)
        self.options_vert.add_child(self.show_colors_cb)

        self.colored_labels_cb = gui.Checkbox("Display label colors")
        self.colored_labels_cb.checked = self.colored_color_labels
        self.colored_labels_cb.set_on_checked(self.show_color_label_colors)
        self.colored_labels_cb.tooltip = "Color 3D labels with dominant object colors"
        self.colored_labels_cb.visible = self.display_labels_and_colors
        self.options_vert.add_child(self.colored_labels_cb)

        self.show_freespace_cb = gui.Checkbox("Show free space annotations")
        self.show_freespace_cb.checked = self.show_freespace_annotations
        self.show_freespace_cb.set_on_checked(self.draw_freespace_annotations)
        self.options_vert.add_child(self.show_freespace_cb)

        self.selector.add_child(self.options_vert)
        
    
    def select_region(self, new_val: str, is_dbl_click):
        self.cur_region_idx = self.region_selector_names[new_val]
        self.cur_target_idx = "-1"
        self.set_objects()
        self.object_type_selector.set_items(self.object_types)
        self.object_instance_selector.set_items([' '*20])
        self.draw_region()
        self.draw_bboxes()
        self.set_language_queries()
        
    def select_object_type(self, new_val: str, is_dbl_click):
        if new_val != 'all':
            idxs = [' '*10 + str(obj['object_id']) + ' '*10 for obj in self.object_types_dict[new_val]]
            self.object_instance_selector.set_items(idxs)
            self.object_instance_selector.selected_index = 0
            self.draw_bboxes(target_idx=idxs[0].strip())
        else:
            self.object_instance_selector.set_items([' '*20])
            self.draw_bboxes(target_idx="-1")
        self.set_language_queries()

        
    def select_object_instance(self, new_val: str, is_dbl_click):
        self.draw_bboxes(target_idx=new_val.strip())
        self.set_language_queries()

    def select_relationship(self, new_val, is_dbl_click):
        self.draw_bboxes(relationship=new_val)
        self.set_language_queries()
    
    def set_language_queries(self):
        if self.cur_region_idx == -1:
            return
        filtered_statements = []
        for statement, instances in self.statements.items():
            if statement=="region":
                continue
            for instance in instances:
                if self.cur_relationship == 'all' or self.cur_relationship == instance['relation']:
                    if self.cur_target_idx == "-1" or instance['target_index'] == self.cur_target_idx:
                        filtered_statements.append(statement)
                        break

        self.language_selector.set_items([statement + ' '*10 for statement in filtered_statements])
        self.window.set_needs_layout()


    def select_language_query(self, new_val, is_dbl_click):
        instances = self.statements[new_val.strip()]
        relation_type = instances[0]['relation_type']
        new_relation = instances[0]['relation']
        for instance in instances:
            target_idx = instance['target_index']
            anchor_idx = [anchor['index'] for anchor in instance['anchors'].values()]
            break
        self.draw_bboxes(target_idx=target_idx, anchor_idx=anchor_idx, relationship=new_relation)

        cur_object = self.objects[target_idx]
        self.object_type_selector.selected_index = self.object_types.index(cur_object['nyu_label'])
        idxs = [' '*10 + str(obj['object_id']) + ' '*10 for obj in self.object_types_dict[cur_object['nyu_label']]]
        self.object_instance_selector.set_items(idxs)
        self.object_instance_selector.selected_index = int(cur_object['dict_idx'])


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
        if self.display_labels_and_colors:
            self.add_color_label(object_idx)

        self.cur_bboxes.append(str(object_idx))
    
    def add_color_label(self, object_idx):
        object = self.objects[object_idx]
        name = object['nyu_label']
        center = object["center"]          

        color_labels = [label for label in object['color_labels'] if label != "N/A"]
        color_vals = [val for val in object['color_vals'] if -1 not in val]
        color_percentages = [p for p in object['color_percentages'] if p != "N/A"]
        label_str = f'{name} {object_idx}:'
        for label, percentage in zip(color_labels, color_percentages):
            label_str += f'\n{label}: {float(percentage)*100:.2f}%'
        label = self.scene_widget.add_3d_label(center, label_str)
        if self.colored_color_labels:
            label.color = gui.Color(color_vals[0][0]/255, color_vals[0][1]/255, color_vals[0][2]/255)
        else:
            label.color = gui.Color(0.0, 0.0, 0.0)
        label.scale = 1.5
        self.scene_labels.append(label)

    def show_color_labels(self, is_checked):
        for label in self.scene_labels:
            self.scene_widget.remove_3d_label(label)
        self.scene_labels = []
        if is_checked:
            self.display_labels_and_colors = True
            self.colored_labels_cb.visible = True
            for object_idx in self.cur_bboxes:
                self.add_color_label(object_idx)
        else:
            self.display_labels_and_colors = False
            self.colored_labels_cb.visible = False
        
    def show_color_label_colors(self, is_checked):
        if is_checked:
            self.colored_color_labels = True
            self.show_color_labels(True)
        else:
            self.colored_color_labels = False
            for label in self.scene_labels:
                label.color = gui.Color(0.0, 0.0, 0.0)

    def draw_region(self):

        self.scene_widget.scene.clear_geometry()

        self.cur_bboxes = []

        if self.cur_region_idx != -1:

            self.scene_widget.scene.add_geometry('pcd', self.region_pcds[self.cur_region_idx], self.material_record)

            bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

            colors = [[0, 0, 0] for _ in range(len(bbox_lines))]

            self.region_bbox = o3d.geometry.LineSet()
            self.region_bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
            self.region_bbox.colors = o3d.utility.Vector3dVector(colors)
            self.region_bbox.points = o3d.utility.Vector3dVector(self.regions[str(self.cur_region_idx)]['region_bbox'])

            mat = rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 10

            self.scene_widget.scene.add_geometry('region_bbox', self.region_bbox, mat)

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

    def draw_freespace_annotations(self, is_checked):
        if self.freespace_pcd is None:
            return
        if is_checked:
            self.show_freespace_annotations = True
            self.scene_widget.scene.add_geometry('freespace_pcd', self.freespace_pcd, self.material_record)
        else:
            self.show_freespace_annotations = False
            self.scene_widget.scene.remove_geometry('freespace_pcd')

    def draw_bboxes(
            self, 
            target_idx = None,
            anchor_idx = None, 
            relationship = None): 

        if self.cur_region_idx == -1:
            return

        for bbox in self.cur_bboxes:
            self.scene_widget.scene.remove_geometry(bbox)
        self.cur_bboxes = []
        
        if self.display_labels_and_colors:
            for label in self.scene_labels:
                self.scene_widget.remove_3d_label(label)
            self.scene_labels = []

        if target_idx is not None:
            self.cur_target_idx = target_idx
        if relationship is not None:
            self.cur_relationship = relationship
            self.relationship_selector.selected_index = self.relationship_types.index(self.cur_relationship)

        if self.cur_target_idx == "-1":
            for i in self.objects.keys():
                self.add_object_bbox(i, self.bbox_colors['target'], line_width=5)
        else:
            self.add_object_bbox(self.cur_target_idx, self.bbox_colors['target'])


            if self.cur_relationship == 'all':
                pass
            else:
                if anchor_idx is not None:
                    for idx in anchor_idx:
                        self.add_object_bbox(idx, self.bbox_colors['anchor'])
                else:
                    anchor_idxs = self.relationship_targets[self.cur_relationship][str(self.cur_target_idx)]
                    for anchor_list in anchor_idxs:
                        for idx in anchor_list:
                            self.add_object_bbox(idx, self.bbox_colors['anchor'])


            cur_object_type = self.objects[self.cur_target_idx]['nyu_label']
            for distractor in self.object_types_dict[cur_object_type]:
                if distractor['object_id'] == self.cur_target_idx:
                    continue
                self.add_object_bbox(distractor['object_id'], self.bbox_colors['distractor'])

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
                        help="Relationship to visualize (default is all)")
    parser.add_argument('--display_labels_and_colors', action='store_const', const=True, default=False,
                        help="Display labels and colors on bounding boxes directly")
    parser.add_argument('--show_freespace_annotations', action='store_const', const=True, default=False,
                        help="Display colors of labels")
    parser.add_argument('--dont_crop_region', action='store_const', const=True, default=False,
                        help="Do not crop region on selection")
    parser.add_argument('--hide_legend', action='store_const', const=True, default=False,
                        help="Hide legend")

    args = parser.parse_args()

    dataset_visualizer = DatasetVisualizer(
        args.scene_path,
        args.optional_mesh_path,
        args.region,
        args.anchor_idx,
        args.relationship,
        args.display_labels_and_colors,
        args.show_freespace_annotations,
        args.dont_crop_region,
        args.hide_legend
    )
