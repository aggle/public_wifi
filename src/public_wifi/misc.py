"""
Useful tools
"""
import numpy as np
import pandas as pd
from photutils.centroids import centroid_2dg as centroid_func
from scipy import ndimage

def get_stamp_center(stamp : int | np.ndarray | pd.Series) -> np.ndarray:
    """
    Get the central pixel of a stamp or cube
    Output
    ------
    center : np.ndarray
      the center in (x, y)/(col, row) format
    """
    if isinstance(stamp, int):
        shape = np.tile(stamp, 2)
    elif isinstance(stamp, pd.Series):
        shape = np.stack(stamp.values).shape[-2:]
    else:
        shape = stamp.shape[-2:]
    center = np.floor(np.array(shape)/2).astype(int)
    return center

def compute_psf_center(stamp):
    """
    Compute the PSF center in a 3x3 box around the nominal center
    return center in (x, y) convention
    """
    pad=2
    center = get_stamp_center(stamp)
    ll = center - pad # lower left corner
    rows, cols = (center[1]-pad,center[1]+pad+1), (center[0]-pad,center[1]+pad+1)
    fit_stamp = stamp[rows[0]:rows[1], cols[0]:cols[1]]
    psf_center = centroid_func(fit_stamp) + ll
    return psf_center

def center_stamp(stamp):
    center = get_stamp_center(stamp)
    psf_center = compute_psf_center(stamp)
    shift = -(psf_center-center)[::-1]
    shifted_img = ndimage.shift(stamp, shift, mode='reflect')
    return shifted_img


def scale_stamp(stamp):
    return (stamp - np.nanmin(stamp))/np.ptp(stamp)


def row_get_psf_stamp_position(row, stamp_size=0):
    """Use the catalog position to get the PSF center in the stamp"""
    xy = np.array(row[['x', 'y']] % 1) - 0.5
    if stamp_size != 0:
        stamp_center = get_stamp_center(stamp_size)
        xy += stamp_center
    return xy

def get_pix_separation_from_center(stamp_size):
    """Get a map of the separation of each pixel from the center"""
    center = get_stamp_center(stamp_size)
    grid = (np.mgrid[:stamp_size, :stamp_size] - center[:, None, None])
    sep_map = np.linalg.norm(grid, axis=0)
    return sep_map
