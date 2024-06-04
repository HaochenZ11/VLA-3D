import numpy as np

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def rot_2d(t):
    """Rotation in 2D."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s,  c],])