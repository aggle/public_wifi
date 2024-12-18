"""
Useful tools
"""
import numpy as np

def calc_stamp_center(stamp):
    """
    Get the central pixel of a stamp
    """
    shape = np.array(stamp.shape)
    center = np.floor((shape/2)).astype(int)
    return center
