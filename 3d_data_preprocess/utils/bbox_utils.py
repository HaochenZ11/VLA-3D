import numpy as np
import math
from numba import jit
from scipy.spatial import ConvexHull
import torch

def calculate_axis_aligned_bbox(points: torch.Tensor):
    '''
    Calculates the axis-aligned bbox of an object.
    '''
    min_xyz, _ = torch.min(points, dim=0)
    max_xyz, _ = torch.max(points, dim=0)
    center = (min_xyz + max_xyz) / 2
    size = max_xyz - min_xyz
    return center.cpu().numpy(), size.cpu().numpy()


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
    l = math.cos(heading) * points_2d[:, 0] + math.sin(heading) * points_2d[:, 1]
    w = -math.sin(heading) * points_2d[:, 0] + math.cos(heading) * points_2d[:, 1]
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

@jit(nopython=True)
def get_minimal_rectangle_from_hull(hull: np.ndarray):
    min_rect_size = np.inf
    for i in range(len(hull)):
        pt1, pt2 = hull[i], hull[(i+1)%len(hull)]
        t = (pt2 - pt1)
        t /= np.linalg.norm(t)
        R = np.array([[t[0], -t[1]],
                        [t[1],  t[0]]], dtype=hull.dtype)
        hull_rot = hull @ R
        x_max_ind, x_min_ind = hull_rot[:, 0].argmax(), hull_rot[:, 0].argmin()
        y_max_ind, y_min_ind = hull_rot[:, 1].argmax(), i
        rect_size = (hull_rot[x_max_ind, 0] - hull_rot[x_min_ind, 0]) * (hull_rot[y_max_ind, 1] - hull_rot[y_min_ind, 1])

        if rect_size < min_rect_size:
            min_rect_size = rect_size
            t_min = t
            center = [(hull_rot[x_max_ind, 0] + hull_rot[x_min_ind, 0]) / 2, (hull_rot[y_max_ind, 1] + hull_rot[y_min_ind, 1]) / 2]
            size = [hull_rot[x_max_ind, 0] - hull_rot[x_min_ind, 0], hull_rot[y_max_ind, 1] - hull_rot[y_min_ind, 1]]

    heading = np.arctan2(t_min[1], t_min[0])
    R = np.array([[t_min[0], -t_min[1]],
                    [t_min[1],  t_min[0]]], dtype=hull.dtype)
    center = R @ np.array(center, dtype=hull.dtype)
    return list(center), size, heading

def calculate_bbox_hull(points: np.ndarray):
    points_2d = points[:, :2]
    hull = points_2d[ConvexHull(points_2d).vertices]
    center_2d, size_2d, heading = get_minimal_rectangle_from_hull(hull)
    center = center_2d + [(points[:, 2].max() + points[:, 2].min()) / 2]
    size = size_2d + [points[:, 2].max() - points[:, 2].min()]
    return center, size, heading

if __name__=="__main__":
    hull = np.array([
        [0., 0.],
        [1., 0.],
        [2., 1.],
        [2., 2.],
        [1., 2.],
        [0., 1.]
    ])
    center, size, heading = get_minimal_rectangle_from_hull(hull)
    print(center, size, heading)