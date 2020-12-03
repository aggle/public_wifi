import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import seaborn as sns

from astropy.io import fits

from . import shared_utils
from . import table_utils
from . import image_utils

filt1_label = table_utils.get_filter_name_from_filter_id('F1')
filt2_label = table_utils.get_filter_name_from_filter_id('F2')
filt_label = {'F1': filt1_label,
              'F2': filt2_label}


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


############
# Plot CMD #
############
def plot_cmd(df, ax=None, ax_args={}):
    """
    Given a stars table, plot the CMD

    Parameters
    ----------
    df : pd.DataFrame
      stars catalog
    ax : matplotlib axis object [None]
      axis object to draw on
    ax_args : dict [{}]
      arguments to pass to the axis
    Output
    ------
    fig : matplotlib Figure object
    """
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel(f"{filt_label['F1']}-{filt_label['F2']}")
    ax.set_ylabel(f"{filt_label['F1']}");

    # set default values
    ax_args.setdefault('marker', '.')
    ax_args.setdefault('s', 50)
    ax_args.setdefault('lw', 0)
    ax_args.setdefault('ec', 'none')
    ax_args.setdefault('alpha', 1)
    ax.scatter(df['star_mag_F1']-df['star_mag_F2'],
               df['star_mag_F1'],
               **ax_args,
               )
    # axis inversion
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim), min(ylim))

    ax.legend()

    fig = ax.get_figure()
    return fig

"""
I made a cubescroller that works in jupyterlab!
"""
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


def _cube_scroller(stamps,
                   titles=None,
                   indices=None,
                   fig_args={},
                   imshow_args={},
                   norm_func=mpl.colors.Normalize,
                   norm_args=()):

    if titles is None:
        titles = [f'Stamp {i}' for i in range(len(stamps))]
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
