import numpy as np
import open3d as o3d
from typing import Tuple

def get_rigid_transform(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    Get the 4x4 rigid transformation matrix relating transforming P to Q

    Parameters
    ----------
    P: np.ndarray
       3xN array of points.
    Q: np.ndarray
       3xN array of points.

    Returns
    -------
    T: np.ndarray
       4x4 rigid transformation matrix.
    '''
    p_bar = np.mean(P, axis=1, keepdims=True)
    q_bar = np.mean(Q, axis=1, keepdims=True)
    P_centered = P - p_bar
    Q_centered = Q - q_bar
    U, _, Vh = np.linalg.svd(P_centered@Q_centered.T)
    R = Vh.T@U.T
    d = q_bar - R@p_bar
    T = np.hstack([R, d])
    T = np.vstack([T, [[0, 0, 0, 1]]])
    print(T.dtype)
    return T

def get_rigid_transform_from_bboxes(box1: o3d.geometry.OrientedBoundingBox, box2: o3d.geometry.OrientedBoundingBox):
    P = np.asarray(box1.get_box_points()).T
    Q = np.asarray(box2.get_box_points()).T
    return get_rigid_transform(P, Q)
