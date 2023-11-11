"""
This file contains utilities for image manipulation: cutting stamps, rotations and reflections,
computing indices, etc.
"""

from functools import reduce
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.io import fits

from . import shared_utils
from . import table_utils


###############
# Save Figure #
###############
# n.b. this has been moved to plot_utils
def savefig(fig, name, save=False, fig_args={}):
    """
    Wrapper for fig.savefig that handles enabling/disabling and printing

    Parameters
    ----------
    fig : mpl.Figure
    name : str or pathlib.Path
      full path for file
    save : bool [False]
      True: save figure. False: only print information
    fig_args : dict {}
      (optional) args to pass to fig.savefig()

    Output
    ------
    No output; saves to disk
    """
    print(name)
    if save != False:
        fig.savefig(name, **fig_args)
        print("Saved!")

"""
One handy tool is being able to cut a stamp out of an image. This function cuts
out square stamps given the stamp center (can be floating point) and desired
shape.
"""
def get_stamp_ind(xy_cent, stamp_shape):
    """
    Return the indices corresponding to the stamp. Does not contain any
    information about the image size; i.e. indices may not be valid near
    the edges of the image.

    Parameters
    ----------
    xy_cent : 2x1 array-like of ints of the xy (col, row) center
    stamp_shape: 2x1 tuple or int

    Output
    ------
    stamp_ind : np.array
      2xN array of (row, col) coordinates

    """
    if isinstance(stamp_shape, int):
        stamp_shape = np.tile(stamp_shape, 2)
    xy_cent = np.array(xy_cent)
    center = xy_cent[::-1] # now in row, col order
    # set up the stamp so that negative is below center and positive is above center
    stamp_range = np.outer(np.array(stamp_shape)/2,
                           np.array((-1, 1))).T
    # on the following line np.floor is necessary to index the proper pixels;
    # astype(int) just makes it compatible with an index
    index_range = np.floor(np.transpose(center + stamp_range)).astype(int)
    stamp_ind = np.mgrid[index_range[0,0]:index_range[0,1],
                         index_range[1,0]:index_range[1,1]]
    return stamp_ind


def get_master_stamp_ind(star_id, stamp_shape, df, extra_row=True):
    """
    Same as get_stamp_ind, but using the master reference frame instead of the
    exposure frame.

    Parameters
    ----------
    star_id : identifier for the star (e.g. S000000)
    stamp_shape : 2x1 tuple or int
    df : pd.DataFrame
      supply the stars dataframe.
    extra_row : bool [True]
      If True, add an extra index on the end. This is needed for plotting with
      plt.pcolor(x, y, img), where x and y need 1 more row and column than img

    Output
    ------
    stamp_ind : np.array
      2xN array of (row, col) coordinates

    """
    if isinstance(stamp_shape, int):
        stamp_shape = np.tile(stamp_shape, 2)

    id_col = shared_utils.find_column(df.columns, 'star_id')
    u_col =  shared_utils.find_column(df.columns, 'u_mast')
    v_col =  shared_utils.find_column(df.columns, 'v_mast')

    xy_cent = df.set_index(id_col).loc[star_id, [u_col, v_col]]
    xy_cent = np.floor(np.array(xy_cent))
    center = xy_cent[::-1] # now in row, col order
    # set up the stamp so that negative is below center and positive is above center
    stamp_range = np.outer(np.array(stamp_shape)/2,
                           np.array((-1, 1))).T
    # on the following line np.floor is necessary to index the proper pixels;
    # astype(int) just makes it compatible with an index
    index_range = np.floor(np.transpose(center + stamp_range)).astype(int)
    shift = 0
    if extra_row == True:
        shift = 1
    stamp_ind = np.mgrid[index_range[0,0]:index_range[0,1]+shift,
                         index_range[1,0]:index_range[1,1]+shift]
    return stamp_ind


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
    xy : list-like (list, tuple, np.array, etc)
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
    if isinstance(stamp_shape, int):
        stamp_shape = np.tile(stamp_shape, 2)
    xy = np.array(xy)
    center = xy[::-1] # now in row, col order
    # set up the stamp so that negative is below center and positive is above center
    stamp_range = np.outer(np.array(stamp_shape)/2.,
                           np.array((-1, 1))).T
    # on the following line np.floor is necessary to index the proper pixels;
    # astype(int) just makes it compatible with an index
    index_range = np.floor(np.transpose(center + stamp_range)).astype(int)
    # crop the range - if index_range goes beyond the image shape, then
    # we only want to index part of the image, and only part of the stamp
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
This is a wrapper for get_stamp, which is a little clunky to use by itself.
This takes in a row of the point sources catalog and pulls out a stamp for that
point source, in that file, of the specified size
"""
def get_stamps_from_ps_table(df, kwargs={}):
    """
    pass in a point source table and get the stamp for the object of each row

    Parameters
    ----------
    df : pd.DataFrame
      a subset of the point sources table (or the whole thing)
    kwargs : dict [{}]
      keyword args to pass to get_stamp_from_ps_row

    Output
    ------
    stamps : pd.DataFrame
      dataframe of stamps indexed by the point source ID

    """
    stamps = df.set_index('ps_id', drop=False).apply(lambda x:
                                                     get_stamp_from_ps_row(x,
                                                                           **kwargs),
                                                     axis=1)
    return stamps

def get_stamp_from_ps_row(row, stamp_size=11, return_img_ind=False, hdr='SCI'):
    """
    Given a table, or a row of the point sources table, this gets a stamp of
    the specified size of the given point source
    TODO: accept multiple rows

    Parameters
    ----------
    row : pd.DataFrame row
      a row containing the position and file information for the source
    stamp_size : int or tuple [11]
      (row, col) size of the stamp [(int, int) if only int given]
    return_img_ind : bool (False)
      if True, return the row and col indices of the stamp in the image
    hdr : str or int ['SCI']
      which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']

    Returns
    -------
    stamp_size-sized stamp
    """
    # get the file name where the point source is located and pull the exposure
    flt_file = table_utils.get_file_name_from_exp_id(row['ps_exp_id'])
    img = fits.getdata(shared_utils.get_data_file(flt_file), hdr)
    # location of the point source in the image
    xy = row[['ps_x_exp', 'ps_y_exp']].values
    # finally, get the stamp (and indices, if requested)
    return_vals = get_stamp(img, xy, stamp_size, return_img_ind)
    return return_vals


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



#######################
### Array Reshaping ###
#######################
def flatten_image_axes(array):
    """
    returns the array with the final two axes - assumed to be the image pixels - flattened
    """
    shape = array.shape
    imshape = shape[-2:]
    newshape = [i for i in list(shape[:-2])]

    newshape += [reduce(lambda x,y: x*y, shape[-2:])]
    return array.reshape(newshape)


def flatten_leading_axes(array, axis=-1):
    """
    For an array of flattened images of with N axes where the first N-1 axes
    index parameters (or whatever else), return an array with the first N-1 axes flattened
    so the final result is 2-D, with the last axis being the pixel axis
    Args:
        array: an array of at least 2 dimensions
        axis [-1]: flattens shape up to this axis (e.g. -1 to flatten up to 
          the last axis, -2 to preserve last two axes, etc.)
    """
    # test axis value is valid
    if np.abs(axis) >= array.ndim:
        print("Preserving all axes shapes")
        return array
    oldshape = array.shape
    newshape = [reduce(lambda x,y: x*y, oldshape[:axis])] + list(oldshape[axis:])
    return np.reshape(array, newshape)


def make_image_from_flat(flat, indices=None, shape=None, squeeze=True):
    """
    put the flattened region back into an image. if no indices or shape are specified, assumes that
    the region of N pixels is a square with Nx = Ny = sqrt(N). Only operates on the last axis.
    Input:
        flat: [Ni,[Nj,[Nk...]]] x Npix array (any shape as long as the last dim is the pixels)
        indices: [None] Npix array of flattened pixel coordinates 
                 corresponding to the region
        shape: [None] image shape
        squeeze [True]: gets rid of extra axes 
    Returns:
        img: an image (or array of) with dims `shape` and with nan's in 
            whatever indices are not explicitly set
    """
    # sometimes you get just a number, e.g. if the image is a NaN
    if np.ndim(flat) == 0:
        return flat
    oldshape = flat.shape[:]
    if shape is None:
        # assume that you have been given the full square imae
        Npix = oldshape[-1]
        Nside = int(np.sqrt(Npix))
        indices = np.array(range(Npix))
        shape = (Nside, Nside)
        return flat.reshape(oldshape[:-1]+shape)

    img = np.ravel(np.zeros(shape))*np.nan
    # this is super memory inefficient
    # handle the case of region being a 2D array by extending the img axes
    if flat.ndim > 1:
        # assume last dimension is the pixel
        flat = np.reshape(flat, (reduce(lambda x,y: x*y, oldshape[:-1]), oldshape[-1]))
        img = np.tile(img, (flat.shape[0], 1))
    else:
        img = img[None,:]
    # fill in the image
    img[:,indices] = flat
    # reshape and get rid of extra axes, if any
    img = img.reshape(list(oldshape[:-1])+list(shape))
    if squeeze == True:
        img = np.squeeze(img)
    return img

def make_stamp_mask(shape, mask_dim, invert=False, return_ind=False, nan=False):
    """
    Make a square mask on the middle of the stamp. 1's on the outside, 0's in the middle

    Parameters
    ----------
    shape : tuple of stamp shape
    mask_dim : int
      size of the mask (one side of the square)
    invert : bool [False]
      inver the max
    return_ind : bool [False]
      if True, return the indices of the pixels you want to keep
      instead of the mask itself
    nan : bool [False]
      if True, use np.nan instead of 0

    Output
    ------
    NxN binary mask with 1 on the outside and 0 (or nan) on the inside

    """
    shape = np.array(shape)
    center = np.floor(shape/2.).astype(int)
    mask_rad = np.floor(mask_dim/2).astype(int)
    mask = np.ones(shape)
    mask[center[0]-mask_rad:center[0]+mask_rad+1,
         center[1]-mask_rad:center[0]+mask_rad+1] = 0
    if invert == True:
        mask = np.abs(mask - 1)
    if nan == True:
        mask[mask==0] = np.nan
    if return_ind == True:
        mask = np.where(mask == 1)
    return mask


# image manipulation
