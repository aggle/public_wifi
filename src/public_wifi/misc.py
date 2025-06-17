"""
Useful tools
"""
import numpy as np
import pandas as pd
from photutils.centroids import centroid_2dg as centroid_func
from scipy import ndimage

def get_stamp_shape(stamp : int | np.ndarray | pd.Series) -> np.ndarray:
    """
    Get the 2-d shape of a stamp given an int, array, or series

    Parameters
    ----------
    stamp : int | np.ndarray | pd.Series
      Either the length of one side, a 2-D image, or a Series of 2-D images
    Output
    ------
    shape : tuple[int]
      the (row, col) shape
    """
    if isinstance(stamp, int):
        shape = np.tile(stamp, 2)
    elif isinstance(stamp, pd.Series):
        shape = np.stack(stamp.values).shape[-2:]
    elif (len(stamp) == 2) and (np.ndim(stamp) == 1):
        shape = np.array(stamp)
    else:
        shape = stamp.shape[-2:]
    return shape

def get_stamp_center(stamp : int | np.ndarray | pd.Series) -> np.ndarray:
    """
    Get the central pixel of a stamp or cube

    Parameters
    ----------
    stamp : int | np.ndarray | pd.Series
      Either the length of one side, a 2-D image, or a Series of 2-D images
    Output
    ------
    center : np.ndarray
      the center in (x, y)/(col, row) format
    """
    shape = get_stamp_shape(stamp)
    center = np.floor(np.array(shape)/2).astype(int)
    return center

def compute_psf_center(stamp, pad=2):
    """
    Compute the PSF center in a 3x3 box around the nominal center
    return center in (x, y) convention
    """
    center = get_stamp_center(stamp)
    ll = center - pad # lower left corner
    rows, cols = (center[1]-pad, center[1]+pad+1), (center[0]-pad, center[1]+pad+1)
    fit_stamp = stamp[rows[0]:rows[1], cols[0]:cols[1]]
    psf_center = centroid_func(fit_stamp) + ll
    return psf_center

def shift_stamp_to_center(stamp, pad=3):
    center = get_stamp_center(stamp)
    # assume the center is already in the correct pixel and we want only a
    # subpixel shift cut out a small region around the center to avoid affects
    # from possible nearby companions
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

def center_to_ll_coords(stamp_size, pix):
    """
    Convert center-origin coordinates to ll-origin coordinates
    returns pix in (x, y) convention
    """
    center = get_stamp_center(stamp_size)
    ll_coord = center + np.array(pix)
    return ll_coord

def ll_to_center_coords(stamp_size, pix):
    """Convert center-origin coordinates to ll-origin coordinates"""
    center = get_stamp_center(stamp_size)
    center_coord = np.array(pix) - center
    return center_coord

def get_annuli(
        stamp_size : int,
        width : int = 2,
        rolling : bool = False,
) -> tuple[float, tuple[tuple[np.ndarray]]] :
    """
    Return a list of indices that define successive annuli of the stamp

    Parameters
    ----------
    stamp_size : int
      The one-sided size of the stamp
    width : int = 2
      width in pixels of the annuli
    rolling : bool = False
      if True, the annuli overlap by half the width

    Output
    ------
    annuli : dict
      Key is the whole-integer radius. Value is the list of coordinates

    """
    radii = get_pix_separation_from_center(stamp_size)
    overlap = width/2 if rolling else 0
    max_width = int(np.floor(stamp_size/2)) - width
    annuli = []
    for i in np.arange(0, max_width+overlap+1, width - overlap):
         pix = np.where((radii >= i) & (radii < i+width))
         sep = np.mean(radii[pix])
         annuli.append((sep, pix))
    return annuli

