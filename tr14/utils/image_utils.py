"""
This file contains utilities for image manipulation: cutting stamps, rotations and reflections, computing indices, etc.
"""

import numpy as np

from astropy.io import fits
from . import shared_utils


"""
One handy tool is being able to cut a stamp out of an image. This function cuts out square stamps given the stamp center (can be floating point) and desired shape.
"""
def get_stamp(image, xy, stamp_shape, return_img_ind=False):
    """
    Cut a stamp from the given image, centered at xy with shape stamp_shape.
    If the stamp shape N is odd, the center c will be at c = (N-1)/2.
    If N is even, then c is given by N/2.
    If the stamp would extend past the borders of the image, those parts of the
    stamp are padded with nan's.

    Parameters
    ----------
    image : np.array
      2-D source image
    xy : np.array
      the xy (col, row) center of the stamp
    stamp_shape : int or tuple of ints
      the dimensions of the stamp
    return_img_ind : bool [False] (not yet implemented)
      if True, then return arrays that index the stamp into the original image

    Returns
    -------
    stamp : np.array
      a copied stamp from the image, centered at xy with shape stamp_shape
    img_ind : np.array (optional)
      if return_img_ind is True, return a matrix of the image indices.
      Can be useful for plotting
    """
    if isinstance(stamp_shape, np.int):
        stamp_shape = np.tile(stamp_shape, 2)
    xy = np.array(xy)
    center = xy[::-1] # now in row, col order
    # set up the stamp so that negative is below center and positive is above center
    stamp_range = np.outer(np.array(stamp_shape)/2,
                           np.array((-1, 1))).T
    # on the following line np.floor is necessary to index the proper pixels;
    # astype(np.int) just makes it compatible with an index
    index_range = np.floor(np.transpose(center + stamp_range)).astype(np.int)
    # crop the range, in case index_range goes beyond the image shape
    cropped_range = np.clip(index_range, [0, 0], image.shape)
    # generate indices into the image to pull out the stamp
    image_ind = np.mgrid[cropped_range[0,0]:cropped_range[0,1],
                         cropped_range[1,0]:cropped_range[1,1]]
    # fancy index math to align the valid regions of the image with the stamp
    cropped_shape = image_ind.shape[-2:]
    cropped_ind = np.indices(cropped_shape)
    # this keeps track of the stamp cropping - 0 if no cropping performed
    expand_ind = index_range - cropped_range
    # if necessary, shift the coordinates to adjust for cropping 
    for ax, shift in enumerate(expand_ind[:, 0]):
        cropped_ind[ax] -= shift
    # initialize the stamp as full of nans
    stamp = np.ones(stamp_shape)*np.nan
    # index only the relevant stamp pixels into only the relevant image indices
    stamp[cropped_ind[0], cropped_ind[1]] = image[image_ind[0], image_ind[1]]
    if return_img_ind is True:
        # get the image indices of the stamp image, even beyond the image bounds
        full_img_ind = np.ogrid[index_range[0,0]:index_range[0,1],
                                index_range[1,0]:index_range[1,1]]
        full_img_ind = [np.squeeze(i) for i in full_img_ind]
        return stamp, full_img_ind
    return stamp


"""
OK, this is good, but it's a little clunky to use. What I need is a wrapper
that takes in a row of the KS2 FIND_NIMFO catalog and handles the parsing
and file-finding for you to get the stamp
OK, I put the wrapper in ks2_utils
ks2_utils.get_stamp_from_ks2(row, stamp_size=11, return_img_ind=False)
"""

"""
This is a handy utility that converts an array of indices into one that
pyplot.pcolor will accept
"""
def make_pcolor_index(indices):
    """
    pyplot.pcolor likes to have the x and y arrays for an image have 1 more
    element than the image, i.e. if the image is  MxN, then the row indices
    for y should have M+1 elements, and the col indices (x) should have N+1
    elements. This assumes that all the indexes are pixel integers.
    It then applies a half-pixel shift so that the integer values
    refer to the pixel centers.

    Parameters
    ----------
    indices: list-like, dim 2xN 
      a tuple, list, or array of the indices, as (row, col) (i.e. y, x). They
      do not need to have the same number of elements

    Returns
    -------
    extended_indices : tuple (y, x)
      the indices with one more element tacked onto the end
    """
    # adjust x and y for use with pcolor
    y = list(indices[0]) + [indices[0][-1] + 1]
    x = list(indices[1]) + [indices[1][-1] + 1]
    return (np.array(y)+0.5,
            np.array(x)+0.5)

