"""
This file contains utilities for image manipulation: cutting stamps, rotations and reflections, computing indices, etc.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from astropy.io import fits

from . import shared_utils, table_utils


###############
# Save Figure #
###############
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
    if isinstance(stamp_shape, np.int):
        stamp_shape = np.tile(stamp_shape, 2)
    xy_cent = np.array(xy_cent)
    center = xy_cent[::-1] # now in row, col order
    # set up the stamp so that negative is below center and positive is above center
    stamp_range = np.outer(np.array(stamp_shape)/2,
                           np.array((-1, 1))).T
    # on the following line np.floor is necessary to index the proper pixels;
    # astype(np.int) just makes it compatible with an index
    index_range = np.floor(np.transpose(center + stamp_range)).astype(np.int)
    stamp_ind = np.mgrid[index_range[0,0]:index_range[0,1],
                         index_range[1,0]:index_range[1,1]]
    return stamp_ind


def get_master_stamp_ind(star_id, stamp_shape, df=None, extra_row=True):
    """
    Same as get_stamp_ind, but using the master reference frame instead of the
    exposure frame.

    Parameters
    ----------
    star_id : identifier for the star (e.g. S000000)
    stamp_shape : 2x1 tuple or int
    df : pd.DataFrame [None]
      optional : supply the dataframe. If None, uses the `stars` table
    extra_row : bool [True]
      If True, add an extra index on the end. This is needed for plotting with
      plt.pcolor(x, y, img), where x and y need 1 more row and column than img

    Output
    ------
    stamp_ind : np.array
      2xN array of (row, col) coordinates

    """


    if df is None:
        df = table_utils.load_table("stars")

    if isinstance(stamp_shape, np.int):
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
    # astype(np.int) just makes it compatible with an index
    index_range = np.floor(np.transpose(center + stamp_range)).astype(np.int)
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
    if isinstance(stamp_shape, np.int):
        stamp_shape = np.tile(stamp_shape, 2)
    xy = np.array(xy)
    center = xy[::-1] # now in row, col order
    # set up the stamp so that negative is below center and positive is above center
    stamp_range = np.outer(np.array(stamp_shape)/2.,
                           np.array((-1, 1))).T
    # on the following line np.floor is necessary to index the proper pixels;
    # astype(np.int) just makes it compatible with an index
    index_range = np.floor(np.transpose(center + stamp_range)).astype(np.int)
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


"""
I made a cubescroller that works in jupyterlab!
"""
def update_image1(img_ind, stamps, df, fig, ax):
    """not used, for some reason"""
    fig, ax = plt.subplots(1, 1, **fig_args)
    row_ind = list(stamps.keys())[img_ind]
    row = df.loc[row_ind]
    title = (f"{row['NMAST']} + {row['exp_id']}\nMag: {row['magu']}"
             + "\nSNR: {row['z2']/row['sz2']:0.2f}")
    img = stamps[row_ind]
    imax = ax.imshow(img)
    fig.colorbar(imax)
    ax.set_title(title)

def cube_scroller(df,
                  stamp_args={},
                  fig_args={},
                  imshow_args={},
                  norm_func=mpl.colors.Normalize,
                  norm_args=()):
    """
    Accept a dataframe and show the stamps in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
      point sources dataframe OR stamps dataframe, with just the ones you want
      to show
    stamp_args : dict
      dict of arguments to pass to ks2_utils.get_stamp_from_ks2(row, **stamp_args)
    fig_args : dict
      dict of arguments to pass to plt.subplots(**fig_args)
    imshow_args : dict
      dict of arguments to pass to ax.imshow(img, **imshow_args)
    norm_func : mpl.colors.Normalize function / child function
      this is separated from imshow_args now because if you pass an *instance*
      then it doesn't get reset; you need to pass the function
    norm_args : list [()]
      arguments to pass to the normalization function (e.g. for PowerNorm(0.5))

    Output
    ------
    interactive_plot : ipywidgets.interact
      an widget that will run in the output of a notebook cell

    Example:
    plot_stamps_scroll(df.query("mag >= 10 and mag < 15"), fig_args=dict(figsize=(6,6)))
    """
    # default arguments
    stamp_args.setdefault("stamp_size", 11)
    #imshow_args.setdefault("norm", mpl.colors.Normalize())


    # store data to be plotted in these lists
    stamps = []
    titles = []
    stamp_indices = []


    # collect data
    # if it's a series, assume it's an array of stamps
    if isinstance(df, pd.Series):
        stamps = np.stack(df.values)
        titles = df.index
        stamp_indices.append([np.arange(stamps.shape[-2]),
                              np.arange(stamps.shape[-1])])

    elif "stamp_array" in df.columns:
        # if the dataframe has the column "stamp_array", it's a stamps table
        for i, row in df.iterrows():
            stamps.append(row['stamp_array'])
            titles.append(f"{row['stamp_star_id']}/{row['stamp_ps_id']}")
            ind = get_stamp_ind(row[['stamp_x_cent','stamp_y_cent']],
                                row['stamp_array'].shape[0])
            stamp_indices.append(ind)
    else:
        # otherwise, assume it's a point sources table
        for i, row in df.iterrows():
            s, ind =  get_stamp_from_ps_row(row, **stamp_args, return_img_ind=True)
            ind[0] = np.tile(ind[0], (ind[0].size, 1)).T
            ind[1] = np.tile(ind[1], (ind[1].size, 1))
            stamps.append(s)
            titles.append(f"{row['ps_star_id']}/{row['ps_id']}\nMag: {row['ps_mag']:0.2f}")
            stamp_indices.append(ind)


    return _cube_scroller(stamps, titles, stamp_indices,
                          fig_args=fig_args,
                          imshow_args=imshow_args,
                          norm_func=norm_func,
                          norm_args=norm_args)


def _cube_scroller(stamps, titles, indices,
                   fig_args={},
                   imshow_args={},
                   norm_func=mpl.colors.Normalize,
                   norm_args=()):

    # plotting functions start here
    def update_image(img_ind): # img_ind goes from 0 to N for N stamps
        fig, ax = plt.subplots(1, 1, **fig_args)
        img = stamps[img_ind]
        if indices == None:
            x = np.arange(0, img.shape[1]+1)
            y = np.arange(0, img.shape[0]+1)
        else:
            yx = indices[img_ind]
            # get the x and y that work with plt.pcolor
            x = np.concatenate((yx[1][0,:], [yx[1][0,-1]+1]))
            y = np.concatenate((yx[0][:,0], [yx[0][-1,0]+1]))
        title = titles[img_ind]
        #imax = ax.imshow(img, **imshow_args, norm=norm_func())
        imax = ax.pcolor(x, y, img, **imshow_args, norm=norm_func())
        ax.set_aspect("equal")
        fig.colorbar(imax, shrink=0.75)
        ax.set_title(title)
        #plt.show(fig)

    slider = widgets.IntSlider(min=0, max=len(stamps)-1, step=1, value=0,
                               description='stamp index')

    interactive_plot = interactive(update_image, img_ind=slider)#, fig=fixed(fig), ax=fixed(ax))
    output = interactive_plot.children[-1]
    if 'figsize' in fig_args.keys():
        width = f"{fig_args['figsize'][0]}in"
        height = f"{fig_args['figsize'][1]}in"
    else:
        width = '350px'
        height = '350px'
    output.layout.width = width
    output.layout.height = height
    return interactive_plot
