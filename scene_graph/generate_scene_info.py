'''
Parse csv scene data into json format for downstream language query generation

TODO: define properties file/json schema?
'''

import csv
import json
from itertools import combinations, permutations
import numpy as np
import argparse
import os
# import pandas as pd
import pandas as pd
from shapely.geometry import Polygon
from bbox_utils import *
from colors import *
from special_relation_classes import *
from tqdm import tqdm
from time import perf_counter
import multiprocessing as mp
from itertools import repeat
from copy import deepcopy


def group_by_class(objects):
    same_class_objects = {}
    for obj in objects:
        if obj["nyu_id"] not in same_class_objects:
            same_class_objects[obj["nyu_id"]] = [obj]
        else:
            same_class_objects[obj["nyu_id"]].append(obj)
    return same_class_objects


def relate_in(anchor_idx, objects, in_thres):
    # find objects inside anchor object

    anchor_obj = objects[anchor_idx]
    if int(anchor_obj["nyu_id"]) not in IN_RELATION and int(anchor_obj["nyu_id"]) not in IN_ON_RELATION:
        return []
    
    in_objs = []
    for i in range(len(objects)):
        
        if anchor_idx == i:
            continue

        max_z_tgt = max([pt[-1] for pt in objects[i]["bbox"]])
        min_z_anc = min([pt[-1] for pt in anchor_obj["bbox"]])
        min_z_tgt = min([pt[-1] for pt in objects[i]["bbox"]])
        max_z_anc = max([pt[-1] for pt in anchor_obj["bbox"]])

        if is_inside_bbox(np.array(objects[i]["center"]), np.array(anchor_obj["bbox"])) \
            and (np.array(anchor_obj["size"]) > np.array(objects[i]["size"])).all() \
            and max_z_tgt < max_z_anc and min_z_tgt > min_z_anc:
            if int(anchor_obj["nyu_id"]) not in IN_ON_RELATION:
                in_objs.append(i)
            elif max_z_tgt < max_z_anc - in_thres * (max_z_anc - min_z_anc):
                in_objs.append(i)
            # print(f"{objects[i]['raw_label']} in {anchor_obj['raw_label']}")
    
    in_obj_ids = [objects[ind]['object_id'] for ind in in_objs]
    return in_obj_ids


def relate_on(vertical_iom, on_thres, under_thres, in_thres, anchor_idx, objects):
    # find objects on another object

    anchor_obj = objects[anchor_idx]

    on_objs = []
    for i in range(len(objects)):
        if anchor_idx == i:
            continue

        max_z_tgt = max([pt[-1] for pt in objects[i]["bbox"]])
        min_z_anc = min([pt[-1] for pt in anchor_obj["bbox"]])
        min_z_tgt = min([pt[-1] for pt in objects[i]["bbox"]])
        max_z_anc = max([pt[-1] for pt in anchor_obj["bbox"]])

        if int(objects[i]['nyu_id']) in STRUCTURES_BLACKLIST:
            continue

        if min_z_tgt <= (max_z_anc + on_thres) \
            and min_z_tgt >= (min_z_anc + under_thres) \
            and calculate_iom_poly(objects[i], anchor_obj) > vertical_iom \
            and int(anchor_obj["nyu_id"]) not in VERTICAL_STRUCTURES \
            and (int(anchor_obj["nyu_id"]) not in IN_RELATION \
                 or (np.array(anchor_obj["size"]) <= np.array(objects[i]["size"])).any()) \
            and anchor_obj["size"][0] * anchor_obj["size"][1] > objects[i]["size"][0] * objects[i]["size"][1]:
            # or int(anchor_obj['nyu_id']) in ON_RELATION \
            # and is_inside_bbox(np.array(objects[i]["center"]), np.array(anchor_obj["bbox"])) \
            # and anchor_obj["size"] > objects[i]["size"] \
            # and min_z_tgt > min_z_anc + under_thres:

            # print(f"{objects[i]['nyu_label']} on {anchor_obj['nyu_label']}")
            if int(anchor_obj['nyu_id']) not in IN_ON_RELATION:
                on_objs.append(i)
            elif max_z_tgt > max_z_anc - in_thres * (max_z_anc - min_z_anc):
                on_objs.append(i)

    on_obj_ids = [objects[ind]['object_id'] for ind in on_objs]
    return on_obj_ids


def relate_hanging_on(hanging_thres_h, hanging_thres_v, anchor_idx, objects, relationships):
    anchor_obj = objects[anchor_idx]

    hanging_on_objs = []
    # norm, bias = get_bbox_face_planes(np.array(anchor_obj["bbox"]))
    for i in range(len(objects)):

        if anchor_idx == i:
            continue


        max_z_tgt = max([pt[-1] for pt in objects[i]["bbox"]])
        min_z_anc = min([pt[-1] for pt in anchor_obj["bbox"]])
        min_z_tgt = min([pt[-1] for pt in objects[i]["bbox"]])
        max_z_anc = max([pt[-1] for pt in anchor_obj["bbox"]])

        # point = np.array(objects[i]["bbox"])
        # t = np.sum(norm[None, 2:, :] * point[:4, None, :], axis=-1) - bias[None, 2:]
        # dist = np.min(np.abs(t))
        dist = get_bbox_horiz_distance(anchor_obj, objects[i])

        if abs(dist) < hanging_thres_h and min_z_tgt > (min_z_anc + hanging_thres_v) and max_z_tgt < max_z_anc \
            and int(objects[i]['nyu_id']) not in STRUCTURES_BLACKLIST:

            on_something = False

            # Make sure that the object is not on anything else
            for obj_id, on_relations in relationships['on'].items():
                if objects[i]['object_id'] in on_relations:
                    on_something = True
                    break

            # Make sure that the object is not in anything else
            for obj_id, in_relations in relationships['in'].items():
                if objects[i]['object_id'] in in_relations:
                    # print(f"{anchor_obj['raw_label']} on something")
                    on_something = True
                    break
            
            if not on_something:
                # print(f"{objects[i]['raw_label']} hanging on {anchor_obj['raw_label']}")
                hanging_on_objs.append(i)

    hanging_on_obj_ids = [objects[ind]['object_id'] for ind in hanging_on_objs]
    return hanging_on_obj_ids


def relate_above(vertical_iom, on_thres, anchor_idx, objects):
    # finds objects above anchor object
    above_objs = []

    anchor_obj = objects[anchor_idx]

    for i in range(len(objects)):
        # skip same object
        if anchor_idx == i:
            continue

        # get lowest/highest points on objects
        max_z2 = max([pt[-1] for pt in objects[i]["bbox"]])
        min_z1 = min([pt[-1] for pt in anchor_obj["bbox"]])
        min_z2 = min([pt[-1] for pt in objects[i]["bbox"]])
        max_z1 = max([pt[-1] for pt in anchor_obj["bbox"]])

        iom = calculate_iom_poly(anchor_obj, objects[i])

        # get 2D bboxes (faces) in x-y plane to ensure some level of overlap in these axes
        # bbox1, bbox2 = get_2D_bboxes([0, 1], objects[anchor_idx], objects[j])
        # compute IOU of x-y faces
        # iou = calculate_iou(bbox1, bbox2)

        # three criteria: center z-coordinate, min/max of object in z-axis, IOU of faces in x-y plane
        if max_z1 + on_thres <= min_z2 and iom > vertical_iom:
            # object above reference object
            # print(f"{objects[i]['raw_label']} above {anchor_obj['raw_label']}")
            above_objs.append(i)


    above_obj_ids = [objects[ind]["object_id"] for ind in above_objs]

    return above_obj_ids


def relate_below(vertical_iom,under_thres, anchor_idx, objects):
    # finds objects below anchor object
    below_objs = []
    anchor_obj = objects[anchor_idx]

    for i in range(len(objects)):
        # skip same object
        if anchor_idx == i:
            continue


        # get lowest/highest points on objects
        max_z_tgt = max([pt[-1] for pt in objects[i]["bbox"]])
        min_z_anc = min([pt[-1] for pt in anchor_obj["bbox"]])
        min_z_tgt = min([pt[-1] for pt in objects[i]["bbox"]])
        max_z_anc = max([pt[-1] for pt in anchor_obj["bbox"]])

        iom = calculate_iom_poly(anchor_obj, objects[i])

        # get 2D bboxes (faces) in x-y plane to ensure some level of overlap in these axes
        # bbox1, bbox2 = get_2D_bboxes([0, 1], objects[anchor_idx], objects[j])
        # compute IOU of x-y faces
        # iou = calculate_iou(bbox1, bbox2)

        # three criteria: center z-coordinate, min/max of object in z-axis, IOU of faces in x-y plane
        # if anchor_obj['nyu_label'] == 'ceiling':
            # print(f'IoM between ceiling and {objects[i]["nyu_label"]}: {iom}')

        if max_z_tgt <= min_z_anc and iom > vertical_iom and int(objects[i]["nyu_id"]) not in SUPPORTING_STRUCTURES \
            or min_z_tgt <= min_z_anc + under_thres and max_z_tgt <= max_z_anc and iom > vertical_iom and int(anchor_obj["nyu_id"]) in UNDER_RELATION:
            # object below reference object
            # print(f"{objects[i]['raw_label']} below {anchor_obj['raw_label']}")
            below_objs.append(i)

    below_obj_ids = [objects[ind]["object_id"] for ind in below_objs]

    return below_obj_ids


def relate_between(between_iom, anchor_idx, objects, overlap_thres, symmetry_thres, distance_thres, anchor_size_thres):
    between_objs = []
    obj_inds = list(range(len(objects)))
    obj_inds.remove(anchor_idx)
    obj_pairs = list(permutations(obj_inds, 2))
    obj_pairs = np.array(obj_pairs)
    if not len(obj_pairs):
        return []
    if len(obj_pairs.shape) < 2:
        obj_pairs = obj_pairs[None, :]

    centers = np.array([o["center"] for o in objects])
    bboxes = np.array([o["bbox"] for o in objects])

    center1 = np.array(objects[anchor_idx]["center"])

    center2 = centers[obj_pairs[:, 0]]
    center3 = centers[obj_pairs[:, 1]]

    bbox1 = np.array(objects[anchor_idx]["bbox"])

    bbox2 = bboxes[obj_pairs[:, 0]]
    bbox3 = bboxes[obj_pairs[:, 1]]

    r = (center3 - center2)[:, :2]
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    R = np.zeros((center2.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 0, 1] = -r[:, 1]
    R[:, 1, 1] = r[:, 0]
    R[:, 2, 2] = 1
    center1_rot = (R.transpose(0, 2, 1) @ center1[None, :, None])[..., 0]
    center2_rot = (R.transpose(0, 2, 1) @ center2[:, :, None])[..., 0]
    center3_rot = (R.transpose(0, 2, 1) @ center3[:, :, None])[..., 0]

    between_xy = (center1_rot[:, 0] > center2_rot[:, 0]) & (center1_rot[:, 0] < center3_rot[:, 0]) 


    # bboxes: N x 6 x 3 x 1
    # R: N x 1 x 3 x 3

    bbox1_rot = (R[:, None, :, :].transpose(0, 1, 3, 2) @ bbox1[None, :, :, None])[..., 0]
    bbox2_rot = (R[:, None, :, :].transpose(0, 1, 3, 2) @ bbox2[:, :, :, None])[..., 0]
    bbox3_rot = (R[:, None, :, :].transpose(0, 1, 3, 2) @ bbox3[:, :, :, None])[..., 0]

    # Check xy intersection

    pairs_filt_xy = obj_pairs[between_xy]

    bbox1_rot_between_xy = bbox1_rot[between_xy]
    bbox2_rot_between_xy = bbox2_rot[between_xy]
    bbox3_rot_between_xy = bbox3_rot[between_xy]

    # Make sure bboxes are not overlapping by checking single dimensional iom

    max_xy1 = bbox1_rot_between_xy[..., 0].max(axis=-1)
    min_xy1 = bbox1_rot_between_xy[..., 0].min(axis=-1)
    max_xy2 = bbox2_rot_between_xy[..., 0].max(axis=-1)
    min_xy2 = bbox2_rot_between_xy[..., 0].min(axis=-1)
    max_xy3 = bbox3_rot_between_xy[..., 0].max(axis=-1)
    min_xy3 = bbox3_rot_between_xy[..., 0].min(axis=-1)

    iom1_1d_xy = (max_xy1 - min_xy3) / np.minimum(max_xy1 - min_xy1, max_xy3 - min_xy3)
    iom2_1d_xy = (max_xy2 - min_xy1) / np.minimum(max_xy1 - min_xy1, max_xy2 - min_xy2)

    dist_sums_xy = -iom1_1d_xy - iom2_1d_xy

    filter_xy = (iom1_1d_xy < overlap_thres) \
        & (iom2_1d_xy < overlap_thres) \
        & (np.abs(iom1_1d_xy - iom2_1d_xy) < symmetry_thres) \
        & (-iom1_1d_xy < distance_thres) \
        & (-iom2_1d_xy < distance_thres) 

    dist_sums_xy = dist_sums_xy[filter_xy]
    pairs_filt_xy = pairs_filt_xy[filter_xy]

    bbox1_rot_between_xy = bbox1_rot_between_xy[filter_xy]
    bbox2_rot_between_xy = bbox2_rot_between_xy[filter_xy]
    bbox3_rot_between_xy = bbox3_rot_between_xy[filter_xy]

    bbox1, bbox2 = get_2D_bboxes([1, 2], bbox1_rot_between_xy, bbox2_rot_between_xy)
    iom1_xy = calculate_iom(bbox1, bbox2)
    # IOU to second anchor
    bbox1, bbox2 = get_2D_bboxes([1, 2], bbox1_rot_between_xy, bbox3_rot_between_xy)
    iom2_xy = calculate_iom(bbox1, bbox2)

    filt = (iom1_xy > between_iom) & (iom2_xy > between_iom)
    dist_sums_xy = dist_sums_xy[filt]
    pairs_filt_xy = pairs_filt_xy[filt]
    pairs_filt_xy = pairs_filt_xy[dist_sums_xy.argsort()][::2]
    dist_sums_xy.sort()
    dist_sums_xy = dist_sums_xy[::2]
    # for pair in pairs_filt:
        # print(f"{objects[anchor_idx]['raw_label']} between {objects[pair[0]]['raw_label']} and {objects[pair[0]]['raw_label']}")



    # Check z intersection

    between_z = (center1_rot[:, 2] > center2_rot[:, 2]) & (center1_rot[:, 2] < center3_rot[:, 2])

    pairs_filt_z = obj_pairs[between_z]

    bbox1_rot_between = bbox1_rot[between_z]
    bbox2_rot_between = bbox2_rot[between_z]
    bbox3_rot_between = bbox3_rot[between_z]

    max_z1 = bbox1_rot_between[..., 2].max(axis=-1)
    min_z1 = bbox1_rot_between[..., 2].min(axis=-1)
    max_z2 = bbox2_rot_between[..., 2].max(axis=-1)
    min_z2 = bbox2_rot_between[..., 2].min(axis=-1)
    max_z3 = bbox3_rot_between[..., 2].max(axis=-1)
    min_z3 = bbox3_rot_between[..., 2].min(axis=-1)

    iom1_1d_z = (max_z1 - min_z3) / np.minimum(max_z1 - min_z1, max_z3 - min_z3)
    iom2_1d_z = (max_z2 - min_z1) / np.minimum(max_z1 - min_z1, max_z2 - min_z2)

    dist_sums_z = -iom1_1d_z - iom2_1d_z

    filter_z = (iom1_1d_z < overlap_thres) & (iom2_1d_z < overlap_thres)

    dist_sums_z = dist_sums_z[filter_z]
    pairs_filt_z = pairs_filt_z[filter_z]

    bbox1_rot_between = bbox1_rot_between[filter_z]
    bbox2_rot_between = bbox2_rot_between[filter_z]
    bbox3_rot_between = bbox3_rot_between[filter_z]

    # print(bbox1_rot_between.shape)

    iom1_z = calculate_iom_poly_vectorized(bbox1_rot_between[:, [0, 3, 2, 1], :], bbox2_rot_between[:, [0, 3, 2, 1], :])
    iom2_z = calculate_iom_poly_vectorized(bbox1_rot_between[:, [0, 3, 2, 1], :], bbox3_rot_between[:, [0, 3, 2, 1], :])

    filt = (iom1_z > between_iom) & (iom2_z > between_iom)
    dist_sums_z = dist_sums_z[filt]
    pairs_filt_z = pairs_filt_z[filt]

    pairs_filt = np.concatenate([pairs_filt_xy, pairs_filt_z])
    dist_sums = np.concatenate([dist_sums_xy, dist_sums_z])
    pairs_filt = pairs_filt[dist_sums.argsort()]

    between_objs.extend([[objects[pair[0]]["object_id"], objects[pair[1]]["object_id"]] for pair in pairs_filt])


    return between_objs


# Group targets of the same class
# Closest target must be close enough to anchor
# Targets must be spaced enough apart such that humans can tell, otherwise discard relation
# 

def relate_ordered(anchor_idx: int, objects: list, same_class_objects: dict, ordered_thres: float):

    anchor_center = objects[anchor_idx]["center"]

    # copy dictionary of same class objects
    same_class_objects_copy = deepcopy(same_class_objects)

    closest_objs = []
    farthest_objs = []

    # calculate distance between all objects and anchor object in dictionary
    for nyu_id, obj_list in same_class_objects_copy.items():
        distances = []
        lengths = []
        for obj in obj_list:
            o_center = obj["center"]
            dir_vec = np.array(anchor_center)-np.array(o_center)
            dist = np.linalg.norm(dir_vec)
            dir_vec /= dist
            bbox = np.array(obj["bbox"])
            projected_bounds = bbox @ dir_vec

            lengths.append(projected_bounds.max() - projected_bounds.min())
            distances.append(float(dist))

        # sort entries in dictionary by ascending order of distance to anchor, remove anchor if found
        dist_idxs_sorted: list = np.argsort(distances).tolist()
        if obj_list[dist_idxs_sorted[0]]["object_id"] == objects[anchor_idx]["object_id"]:
            removed_idx = dist_idxs_sorted.pop(0)

        obj_list = [obj_list[idx] for idx in dist_idxs_sorted]
        distances = np.array(distances)[dist_idxs_sorted]
        lengths = np.array(lengths)[dist_idxs_sorted]

        # populate closest
        closest_obj_list = obj_list[:3]
        closest_distances = distances[:3]
        closest_lengths = lengths[:3]
        
        if len(closest_obj_list) >= 2:
            avg_length = closest_lengths.sum() / len(closest_lengths)
            dist_differences = np.abs(closest_distances[1:] - closest_distances[:-1])
            if ((dist_differences * avg_length) > ordered_thres).all():
                closest_objs.append([obj["object_id"] for obj in closest_obj_list])

        # populate farthest
        farthest_obj_list = list(reversed(obj_list))[:3]
        farthest_distances = distances[::-1][:3]
        farthest_lengths = lengths[::-1][:3]
        
        if len(farthest_obj_list) >= 2:
            avg_length = farthest_lengths.sum() / len(farthest_lengths)
            dist_differences = np.abs(farthest_distances[1:] - farthest_distances[:-1])
            if ((dist_differences * avg_length) > ordered_thres).all():
                farthest_objs.append([obj["object_id"] for obj in farthest_obj_list])

    return closest_objs, farthest_objs

# near_thres is some value between 0 and 1 for percentage of region size
# IDEA: check for nearness based on faces?
def relate_near(near_thres, anchor_idx, objects, region_bbox):
    anchor_coords = np.array(objects[anchor_idx]["bbox"])
    anchor_center = objects[anchor_idx]["center"]
    r_bbox = np.array(region_bbox)
    r_xlen = np.max(r_bbox[:, 0]) - np.min(r_bbox[:, 0])
    r_ylen = np.max(r_bbox[:, 1]) - np.min(r_bbox[:, 1])
    r_zlen = np.max(r_bbox[:, 2]) - np.min(r_bbox[:, 2])

    region_size = r_xlen * r_ylen * r_zlen
    near_objs = []

    for i in range(len(objects)):
        if i == anchor_idx:
            continue

        object_coords = np.array(objects[i]["bbox"])
        object_center = objects[i]["center"]
        center_dist = np.linalg.norm(np.array(anchor_center) - np.array(object_center))

        # check if centers close
        if (center_dist < near_thres * region_size):
            near_objs.append(i)
        else:
            dists = []
            # get distance between each pair of box coordinates
            for p1 in anchor_coords:
                d = [np.linalg.norm(p1-p2) for p2 in object_coords]
                dists += d

            #print("dists", dists)
            # object near if at least two points close to each other
            if np.any(dists < near_thres * region_size) > 1:
                near_objs.append(i)

    near_obj_ids = [objects[ind]["object_id"] for ind in near_objs]
    return near_obj_ids


def compute_spatial_relationships(args, region_struct):
    # compute spatial relationships: above/below, closest/farthest, between
    objects = region_struct["objects"] # list of dicts
    region_bbox = region_struct["region_bbox"]
    relations = [
        "above", 
        "below", 
        "closest", 
        "second_closest", 
        "third_closest", 
        "farthest",
        "second_farthest",
        "third_farthest",
        "between", 
        "beside", 
        "near", 
        "in", 
        "on", 
        "hanging_on"]
    between_iom = args.between_iom
    vertical_iom = args.vertical_iom
    near_thres = args.near_thres
    overlap_thres = args.overlap_thres
    symmetry_thres = args.symmetry_thres
    distance_thres = args.distance_thres
    anchor_size_thres = args.anchor_size_thres
    on_thres = args.on_thres
    in_thres = args.in_thres
    under_thres = args.under_thres
    hanging_thres_h = args.hanging_thres_h
    hanging_thres_v = args.hanging_thres_v
    ordered_thres = args.ordered_thres

    for r in relations:
        region_struct["relationships"].update({r:{}})

    same_class_objects = group_by_class(objects)

    # requires all objects: closest, farthest
    for i in range(len(objects)):
        object_id = objects[i]["object_id"]
        closest_objs, farthest_objs = relate_ordered(i, objects, same_class_objects, ordered_thres)

        # schema 1: anchor: [[closest, 2nd_closest, 3rd_closest], [closest, 2nd_closest, ...]]
        # region_struct["relationships"]["closest"].update({object_id: closest_objs})
        # region_struct["relationships"]["farthest"].update({object_id: farthest_objs})

        # schema 2: split, anchor: [closest, ...], anchor: [second_closest, ...]
        first_closest = [obj[0] for obj in closest_objs]
        second_closest = [obj[1] for obj in closest_objs if len(obj)>1]
        third_closest = [obj[2] for obj in closest_objs if len(obj)>2]

        region_struct["relationships"]["closest"].update({object_id: first_closest})
        region_struct["relationships"]["second_closest"].update({object_id: second_closest})
        region_struct["relationships"]["third_closest"].update({object_id: third_closest})

        first_farthest = [obj[0] for obj in farthest_objs]
        second_farthest = [obj[1] for obj in farthest_objs if len(obj)>1]
        third_farthest = [obj[2] for obj in farthest_objs if len(obj)>2]

        region_struct["relationships"]["farthest"].update({object_id: first_farthest})
        region_struct["relationships"]["second_farthest"].update({object_id: second_farthest})
        region_struct["relationships"]["third_farthest"].update({object_id: third_farthest})
        # store sorted list of closest and farthest objects

        # pairwise (binary) relations: above, below, in
        above_objs = relate_above(vertical_iom, on_thres, i, objects)
        below_objs = relate_below(vertical_iom, under_thres, i, objects)
        near_objs = relate_near(near_thres, i, objects, region_bbox)
        in_objs = relate_in(i, objects, in_thres)
        on_objs = relate_on(vertical_iom, on_thres, under_thres, in_thres, i, objects)

        # triple object (ternary) relations: between
        # init_time = perf_counter()
        between_objs = relate_between(between_iom, i, objects, overlap_thres, symmetry_thres, distance_thres, anchor_size_thres)
        # end_time = perf_counter() - init_time
        # between_time += end_time

        # store relationships per object at region-level
        #region_struct["relationships"]["beside"].append(beside_objs)
        region_struct["relationships"]["above"].update({object_id:above_objs})
        region_struct["relationships"]["below"].update({object_id:below_objs})
        region_struct["relationships"]["between"].update({object_id:between_objs})
        region_struct["relationships"]["near"].update({object_id:near_objs})
        region_struct["relationships"]["in"].update({object_id:in_objs})
        region_struct["relationships"]["on"].update({object_id:on_objs})
    
    # print(between_time)
    
    for i in range(len(objects)):
        object_id = objects[i]["object_id"]
        hanging_on_objs = relate_hanging_on(hanging_thres_h, hanging_thres_v, i, objects, region_struct["relationships"])
        region_struct["relationships"]["hanging_on"].update({object_id:hanging_on_objs})
    
    return relations


def csv_to_json(scene_tuple, i, total_num_scenes):

    dataset_name, scene_name = scene_tuple
    input_folder = args.input_path
    scene_path = os.path.join(input_folder, dataset_name, scene_name)
    
    if not os.path.isdir(scene_path):
        return
    region_file = os.path.join(scene_path, scene_name + '_region_result.csv')
    object_file = os.path.join(scene_path, scene_name + '_object_result.csv')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    affordance_file = os.path.join(current_dir, 'NYUv2_ChatGPT_Affordances.csv')

    affordance_df = pd.read_csv(affordance_file, index_col=0)
    affordance_dict = affordance_df['affordance'].to_dict()
    affordance_dict = {key: [v.strip() for v in val.split(',')] for key, val in affordance_dict.items()}

    scene_data = {"scene_name":scene_name, "regions": {}}
    # output_folder = args.output_folder
    # objects to remove based on nyu label
    remove_objs = ["void", "otherstructure", "otherprop", "otherfurniture", "nyu40class", "unknown"]

    # open csv with scene information
    with open(region_file, encoding='utf-8') as csv_file:
        csvReader = csv.DictReader(csv_file)

        # for each region
        for row in csvReader:
            region = row["region_label"]
            region_id = row["region_id"]

            region_struct = {
                "region_id": region_id,
                "region_name": region,
                "region_bbox": get_bbox_coords_heading('region', row),
                "objects": [],
                "relationships": {}
            }

            # store with region_id as key
            scene_data["regions"].update({region_id: region_struct})

    with open(object_file, encoding='utf-8') as csv_file:
        csvReader = csv.DictReader(csv_file)

        for row in csvReader:
            if row["nyu_label"] in remove_objs or row["region_id"] == "-1":
                continue
            region_id = row["region_id"]
            # define object dict
            obj = {
                "object_id": row["object_id"],
                "raw_label": row["raw_label"],
                "nyu_id": row["nyu_id"],
                "nyu40_id": row["nyu40_id"],
                "nyu_label": row["nyu_label"],
                "nyu40_label": row["nyu40_label"],
                "color_vals": get_color_vals(row),
                "color_labels": get_color_labels(row),
                "color_percentages": get_color_percentages(row),
                "bbox": get_bbox_coords_heading('object', row),
                "center": [float(row["object_bbox_cx"]), float(row["object_bbox_cy"]), float(row["object_bbox_cz"])],
                "volume": float(get_obj_volume(row)),
                "size": list(get_obj_size(row)),
                "affordances": affordance_dict[int(row["nyu_id"])]
            }
            # separate based on region
            scene_data["regions"][region_id]["objects"].append(obj)

    # compute and store spatial relationships for each region
    for r in scene_data["regions"].keys():
        region_data = scene_data["regions"][r]
        compute_spatial_relationships(args, region_data)
        scene_data["regions"][r] = region_data
        #print(region_data)

    output_path = os.path.join(scene_path, scene_name + '_scene_graph.json')

    # write one file per region in Matterport scene
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(scene_data, out, indent=4)
    
    print(f'Processed {scene_name} ({i}/{total_num_scenes})')


def process_data():
    input_folder = args.input_path
    # iterate over scene folders in dataset
    dataset_list = [dataset for dataset in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, dataset))]
    scene_list = [(dataset, scene) for dataset in dataset_list for scene in os.listdir(os.path.join(input_folder, dataset))]
    start_time = perf_counter()
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(csv_to_json, zip(scene_list, range(len(scene_list)), repeat(int(len(scene_list)))))
    print(f'Time taken: {perf_counter() - start_time}')


# TODO: put into config file
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates json scene file with relationships")
    parser.add_argument("--input_path", required=True, help="input folder path to dataset")
    #parser.add_argument("--output_folder", required=True, help="output folder path to dump json files")
    parser.add_argument("--between_iom", default=0.5, help="minimum intersection over minimum (IOM) threshold between bboxes for between relationship")
    parser.add_argument("--vertical_iom", default=0.5, help="minimum intersection over minimum (IOM) threshold between bboxes for above/below relationship")
    parser.add_argument("--near_thres", default=0.01, help="distance threshold for near relationship")
    parser.add_argument("--overlap_thres", default=0.3, help="allowable overlap threshold for calculating between relationship")
    parser.add_argument("--symmetry_thres", default=0.5, help="allowable symmetry threshold for calculating between relationship")
    parser.add_argument("--distance_thres", default=1, help="allowable distance threshold for calculating between relationship")
    parser.add_argument("--anchor_size_thres", default=1.5, help="allowable anchor size threshold for calculating between relationship")
    parser.add_argument("--on_thres", default=0.01, help="allowable height threshold for calculating on relationship")
    parser.add_argument("--in_thres", default=0.1, help="allowable height threshold for calculating in relationship for certain classes")
    parser.add_argument("--under_thres", default=0.01, help="allowable overlap threshold for calculating between relationship")
    parser.add_argument("--hanging_thres_h", default=0.01, help="allowable overlap threshold for calculating between relationship")
    parser.add_argument("--hanging_thres_v", default=0.5, help="allowable overlap threshold for calculating between relationship")
    parser.add_argument("--ordered_thres", default=0.2, help="allowable overlap threshold for calculating between relationship")
    args = parser.parse_args()

    process_data()