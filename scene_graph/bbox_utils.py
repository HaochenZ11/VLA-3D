import numpy as np
from shapely.geometry import Polygon
from shapely import distance
from scipy.spatial import ConvexHull
from numba import jit
from polygon_intersection import intersect_polygons, polygon_area

BBOX_FACES = np.array([
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [1, 2, 6, 5],
    [0, 4, 7, 3],
    [0, 1, 5, 4],
    [2, 3, 7, 6]
])

# IOU code referenced from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def calculate_iou(bbox1, bbox2):
    # determine the coordinates of intersection rectangle
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    # compute the area of intersection of the two boxes
    intersect = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if intersect == 0:
        return 0

    # get area of each box
    bbox1_area = abs((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    bbox2_area = abs((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))

    # compute the area union of the two boxes
    union = bbox1_area + bbox2_area - intersect

    # divide intersection by union
    iou = intersect / float(union)

    return iou

def calculate_iom(bbox1, bbox2):
    # determine the coordinates of intersection rectangle
    xA = np.maximum(bbox1[:, 0], bbox2[:, 0])
    yA = np.maximum(bbox1[:, 1], bbox2[:, 1])
    xB = np.minimum(bbox1[:, 2], bbox2[:, 2])
    yB = np.minimum(bbox1[:, 3], bbox2[:, 3])

    # compute the area of intersection of the two boxes
    intersect = np.abs(np.maximum(xB - xA, np.zeros(xB.shape)) * np.maximum(yB - yA, np.zeros(xB.shape)))

    # if intersect == 0:
        # return 0

    # get area of each box
    bbox1_area = np.abs((bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]))
    bbox2_area = np.abs((bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]))

    # get minimum area between boxes
    minimum = np.minimum(bbox1_area, bbox2_area)

    # divide intersection by union
    iom = intersect / minimum

    return iom


def calculate_iom_poly(obj1, obj2):
    '''
    Calculate intersection over minimum area between two oriented 2D bounding boxes.
    '''
    bbox1 = np.array(obj1["bbox"])[[0, 3, 2, 1, 0], :2]
    bbox2 = np.array(obj2["bbox"])[[0, 3, 2, 1, 0], :2]

    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    if min(poly1.area, poly2.area) == 0.0:
        return 0
    iom = poly1.intersection(poly2).area / min(poly1.area, poly2.area)

    return iom

@jit(nopython=True)
def calculate_iom_poly_vectorized(bboxes1, bboxes2):
    N = len(bboxes2)
    ioms = np.zeros(N, dtype=np.float64)
    for i in range(N):
        bbox1 = bboxes1[i, :, :2]
        bbox2 = bboxes2[i, :, :2]
        # print(bbox1.shape, bbox2.shape)
        intersection = polygon_area(intersect_polygons(bbox1, bbox2))
        bbox1_area = polygon_area(bbox1)
        bbox2_area = polygon_area(bbox2)
        # print(intersection)
        if intersection > 0:
            ioms[i] = intersection / min(bbox1_area, bbox2_area)
            # print(ioms[i])
    return ioms


def calculate_iou_poly(bbox1, bbox2):
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    if poly1.union(poly2).area == 0.0:
        return 0
    iou = poly1.intersection(poly2).area / poly1.union(poly2).area

    return iou


# gets 2D bbox based on two specified axes
def get_2D_bboxes(axes, bbox1_coords: np.ndarray, bbox2_coords: np.ndarray):
    bbox1_2d = bbox1_coords[..., axes]
    bbox2_2d = bbox2_coords[..., axes]

    bbox1 = np.concatenate([
        bbox1_2d[..., 0].min(axis=-1, keepdims=True),
        bbox1_2d[..., 1].min(axis=-1, keepdims=True),
        bbox1_2d[..., 0].max(axis=-1, keepdims=True),
        bbox1_2d[..., 1].max(axis=-1, keepdims=True),
        ], axis=-1)

    bbox2 = np.concatenate([
        bbox2_2d[..., 0].min(axis=-1, keepdims=True),
        bbox2_2d[..., 1].min(axis=-1, keepdims=True),
        bbox2_2d[..., 0].max(axis=-1, keepdims=True),
        bbox2_2d[..., 1].max(axis=-1, keepdims=True),
        ], axis=-1)

    # # get coordinates of box without 3rd axis
    # for i in range(len(bbox1_coords)):
    #     bbox1_2d.append([bbox1_coords[i][axes[0]], bbox1_coords[i][axes[1]]])
    #     bbox2_2d.append([bbox2_coords[i][axes[0]], bbox2_coords[i][axes[1]]])

    # remove duplicates
    # bbox1_2d = [list(i) for i in set(map(tuple, bbox1_2d))]
    # bbox2_2d = [list(i) for i in set(map(tuple, bbox2_2d))]

    # convert to (x1, y1, x2, y2) format
    # bbox1 = [min([x[0] for x in bbox1_2d]), min([x[1] for x in bbox1_2d]), max([x[0] for x in bbox1_2d]), max([x[1] for x in bbox1_2d])]
    # bbox2 = [min([x[0] for x in bbox2_2d]), min([x[1] for x in bbox2_2d]), max([x[0] for x in bbox2_2d]), max([x[1] for x in bbox2_2d])]

    return bbox1, bbox2

def get_obj_size(row):
    xlen = float(row["object_bbox_xlength"])
    ylen = float(row["object_bbox_ylength"])
    zlen = float(row["object_bbox_zlength"])

    return xlen, ylen, zlen

def get_obj_volume(row):
    # compute size of object based on volume of bounding box
    xlen, ylen, zlen = get_obj_size(row)

    return xlen*ylen*zlen


def get_bbox_coords(prefix, row):
    lengths = []
    center = []
    # get centers and lengths for each axis
    for ax in ['x', 'y', 'z']:
        ax_c_key = prefix + '_bbox_c' + ax
        ax_c = float(row[ax_c_key])
        ax_l_key = prefix + '_bbox_' + ax + 'length'
        ax_l = float(row[ax_l_key])
        lengths.append(ax_l)
        center.append(ax_c)

    # get rotation matrix
    R = []
    for i in range(4):
        for j in range(4):
            r_key = prefix + '_bbox_rot' + str(i+1) + str(j+1)
            R.append(float(row[r_key]))

    R = np.array(R).reshape((4,4))

    # calculate corner points
    l = lengths[0] / 2
    w = lengths[1] / 2
    h = lengths[2] / 2
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    addition = [1, 1, 1, 1, 1, 1, 1, 1]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners, addition]))[:3, :]
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]

    coords = []
    for i in range(corners_3d.shape[-1]):
        coords.append(list(corners_3d[:, i]))

    return coords


def get_bbox_coords_unrot(prefix, row):
    # get list of bounding box coordinates for object given object center and length

    ax_minmax = []
    for ax in ['x', 'y', 'z']:
        ax_c_key = prefix + '_bbox_c' + ax
        ax_c = float(row[ax_c_key])
        ax_l_key = prefix + '_bbox_' + ax + 'length'
        ax_l = float(row[ax_l_key])

        ax_min = ax_c - ax_l / 2
        ax_max = ax_c + ax_l / 2
        ax_minmax.append([ax_min, ax_max])

    size_x, size_y, size_z = 2, 2, 2
    g = ((x, y, z) for x in range(size_x) for y in range(size_y) for z in range(size_z))

    coords = []
    for inds in g:
        coords.append([ax_minmax[inds[0]], ax_minmax[inds[1]], ax_minmax[inds[2]]])

    return coords


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def get_bbox_coords_heading(prefix, row):
    lengths = []
    center = []
    heading = float(row[prefix + '_bbox_heading'])
    # get centers and lengths for each axis
    for ax in ['x', 'y', 'z']:
        ax_c_key = prefix + '_bbox_c' + ax
        ax_c = float(row[ax_c_key])
        ax_l_key = prefix + '_bbox_' + ax + 'length'
        ax_l = float(row[ax_l_key])
        lengths.append(ax_l)
        center.append(ax_c)

    h = float(lengths[2])
    w = float(lengths[1])
    l = float(lengths[0])

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

    coords = []
    for i in range(corners_3d.shape[-1]):
        coords.append(list(corners_3d[:, i]))

    return coords

def get_bbox_face_planes(bbox: np.ndarray):

    vec1 = bbox[BBOX_FACES[:, 1]] - bbox[BBOX_FACES[:, 0]]
    vec2 = bbox[BBOX_FACES[:, 3]] - bbox[BBOX_FACES[:, 0]]
    norm = np.cross(vec1, vec2)
    norm /= np.linalg.norm(norm)
    bias = -np.sum(norm * bbox[BBOX_FACES[:, 0]], axis=-1)

    return norm, bias

def is_inside_bbox(point: np.ndarray, bbox: np.ndarray):
    # point + t * normal_vec = x
    # normal_vec.dot(x) + d = 0
    # t = -n.dot(x) - d > 0 out, < 0 in
    norm, bias = get_bbox_face_planes(bbox)
    t = -np.sum(norm * point[None, :], axis=-1) - bias
    return (t>0).all()

def is_inside_xy_bbox(point: np.ndarray, bbox: np.ndarray):
    norm, bias = get_bbox_face_planes(bbox)
    t = -np.sum(norm * point[None, :], axis=-1) - bias
    return (t[2:]>0).all()

def get_bbox_horiz_distance(obj1, obj2):
    bbox1 = np.array(obj1["bbox"])[[0, 3, 2, 1, 0], :2]
    bbox2 = np.array(obj2["bbox"])[[0, 3, 2, 1, 0], :2]

    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    return distance(poly1, poly2)