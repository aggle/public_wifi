"""
This file contains utilities for image manipulation: cutting stamps, rotations and reflections, computing indices, etc.
"""

import numpy as np

"""
One handy tool is being able to cut a stamp out of an image. This function cuts out square stamps given the stamp center (can be floating point) and desired shape.
"""
def get_stamp(image, xy, stamp_shape):
    """
    Cut a stamp from the given image, centered at xy with shape stamp_shape.
    If the stamp shape N is odd, the center c will be at c = (N-1)/2.
    If N is even, then c is given by N/2.
    If the stamp would extend past the borders of the image, those parts of the
    stamp are padded with nan's.

    Parametersaaaaa
    ----------
    image : np.array
      2-D source image
    xy : np.array
      the xy (col, row) center of the stamp
    stamp_shape : int or tuple of ints
      the dimensions of the stamp

    Returns
    -------
    stamp : np.array
      a copied stamp from the image, centered at xy with shape stamp_shape
    """
    if isinstance(stamp_shape, np.int):
        stamp_shape = np.tile(stamp_shape, 2)
    center = xy[::-1] # now in row, col order
    # set up the stamp so that negative is below center and positive is above center
    stamp_range = np.outer(np.array(stamp_shape)/2,
                           np.array((-1, 1))).T
    stamp_range = np.floor(stamp_range) # for proper indexing
    index_range = np.transpose(center + stamp_range).astype(np.int)
    # crop the range, in case index_range goes beyond the image shape
    cropped_range = np.clip(index_range, [0, 0], image.shape)
    # generate indices into the image to pull out the stamp
    image_ind = np.mgrid[cropped_range[0,0]:cropped_range[0,1],
                         cropped_range[1,0]:cropped_range[1,1]]
    # fancy index math to align the valid regions of the image with the stamp
    cropped_shape = image_ind.shape[-2:]
    cropped_ind = np.indices(cropped_shape)
    # this keeps track of the stamp cropping
    expand_ind = index_range - cropped_range
    # if necessary, shift the coordinates to adjust for cropping 
    for ax, shift in enumerate(expand_ind[:, 0]):
        cropped_ind[ax] -= shift
    # initialize the stamp as full of nans
    stamp = np.ones(stamp_shape)*np.nan
    # index only the relevant stamp pixels into only the relevant image indices
    stamp[cropped_ind[0], cropped_ind[1]] = image[image_ind[0], image_ind[1]]
    return stamp
