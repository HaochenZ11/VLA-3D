import numpy as np
from numba import jit, float64
from numba.typed.typedlist import List
from numba.experimental import jitclass

@jit(nopython=True)
def cut_polygon(polygon: np.ndarray, line: np.ndarray):
    # Check for any intersections
    d_line = line[1] - line[0]
    n_line = np.array([d_line[1], -d_line[0]], dtype=np.float64)
    n_line /= np.linalg.norm(n_line)
    bias = -np.sum(n_line * line[0])
    t = -np.sum(n_line[None, :] * polygon, axis=-1) - bias
    
    # If no intersections, return either full polygon or none of the polygon
    if (t>=0).all():
        return polygon
    elif (t<=0).all():
        return None

    # Otherwise, compute intersections
    intersection_idx = -np.ones(2, dtype=np.int32)
    intersection_pts = np.zeros((2, 2), dtype=np.float64)
    A = np.zeros((2, 2), dtype=np.float64)
    A[1, :] = -d_line
    j = 0
    for i in range(len(polygon)):
        d_poly = polygon[(i+1)%len(polygon)] - polygon[i]
        A[0, :] = d_poly
        if np.isnan(A).any() or np.isinf(A).any() or np.linalg.matrix_rank(A) < 2:
            continue
        b = line[0] - polygon[i]
        x = np.linalg.solve(A.T, b)
        if 0 < x[0] < 1:
            intersection_idx[j] = i
            intersection_pts[j, :] = polygon[i] + d_poly * x[0]
            j += 1

    poly_out = np.zeros((np.count_nonzero(t>0)+2, 2), dtype=np.float64)
    j = 0
    k = 0
    for i in range(len(polygon)):
        if t[i] > 0:
            poly_out[k] = polygon[i]
            k += 1
        if i == intersection_idx[j]:
            poly_out[k] = intersection_pts[j]
            j += 1
            k += 1
    
    return poly_out

@jit(nopython=True)
def intersect_polygons(polygon1: np.ndarray, polygon2: np.ndarray):
    intersection = polygon2
    line = np.zeros((2, 2), dtype=np.float64)
    for i in range(len(polygon1)):
        line[0] = polygon1[i]
        line[1] = polygon1[(i+1)%len(polygon1)]
        intersection = cut_polygon(intersection, line)
        if intersection is None:
            return None
    return intersection

@jit(nopython=True)
def polygon_area(polygon: np.ndarray):
    area = 0.
    if polygon is None:
        return 0.
    for i in range(len(polygon)):
        area += 0.5 * (polygon[i, 1] + polygon[(i+1)%len(polygon), 1]) * (polygon[(i+1)%len(polygon), 0] - polygon[i, 0])
    return abs(area)


if __name__ == "__main__":
    poly1 = np.array([
        [0., 0.],
        [2., 0.],
        [2., 2.],
        [0., 2.]
    ])
    poly2 = np.array([
        [3., 0.],
        [3., 3.],
        [0., 3.],
    ])
    out = intersect_polygons(poly1, poly2)
    print(out)
    print(polygon_area(out))