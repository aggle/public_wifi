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
from . import db_manager


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

    if ax_args.get('label') != None:
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
        imax = ax.pcolor(x, y, img, **imshow_args, norm=norm_func(*norm_args))
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


def plot_detector_with_grid(ax=None, dbm=None):
    """
    Plot the grid (as defined in the table grid_sector_definitions) overlaid
    with the point source detections, all with pretty colors

    Parameters
    ----------
    ax : matplotlib axis (None)
      axis to draw on
    dbm : database manager instance (None)
      the database to use for plotting

    Output
    ------
    fig : the parent figure

    """
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
    else:
        fig = ax.get_figure()

    if dbm == None:
        dbm = db_manager.DBManager() # use the default file

    plot_args = {'ls':'-', 'lw':1, 'c':'k'}
    for i, row in dbm.grid_defs_tab.iterrows():
        # bottom
        ax.plot(row[['x_llim', 'x_ulim']], row[['y_llim','y_llim']], **plot_args)
        # top
        ax.plot(row[['x_llim', 'x_ulim']], row[['y_ulim','y_ulim']], **plot_args)
        # left
        ax.plot(row[['x_llim', 'x_llim']], row[['y_llim','y_ulim']], **plot_args)
        # right
        ax.plot(row[['x_ulim', 'x_ulim']], row[['y_llim','y_ulim']], **plot_args)
        # label
        text_x = row[['x_llim','x_ulim']].mean()
        text_y = row[['y_llim','y_ulim']].mean()
        ax.text(text_x, text_y, f"{row['sector_id']}",
                color='k', fontsize='xx-large', fontweight='demibold',
                horizontalalignment='center', verticalalignment='center')
    xmin, ymin = dbm.grid_defs_tab[['x_llim', 'y_llim']].min()
    xmax, ymax = dbm.grid_defs_tab[['x_ulim', 'y_ulim']].max()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # draw point sources
    ps_sec_tab = dbm.ps_tab.merge(dbm.lookup_dict['lookup_point_source_sectors'], on='ps_id')
    sector_colors = mpl.cm.rainbow_r(dbm.grid_defs_tab.index/dbm.grid_defs_tab.index.size)
    for i, group in enumerate(ps_sec_tab.groupby('sector_id').groups.items()):
        ps_sector = ps_sec_tab.loc[group[1]]
        sector_color = sector_colors[i]
        ax.scatter(ps_sector['ps_x_exp'], ps_sector['ps_y_exp'], marker='.',
                   s=50, c=[sector_color])
    ax.set_aspect('equal')
    return fig

def plot_correlation_matrix(subtr_man, which='mse', ax=None):
    """
    Plot the correlation matrix. Takes asubtraction manager object with the correlation
    matrices already computed

    Parameters
    ----------
    subtr_man : a SubtrManager instance
    which : str ['mse']
      which matrix to plot. options: mse, pcc, ssim
    ax : matplotlib axis [None]
      axis object to plot on

    Output
    ------
    fig : parent figure object

    """
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()



    return fig
 

def plot_sky_context(star_id, stars_table, img_size=61,
                     show_neighbors=False, neighbors_table=None,
                     ax=None):
    """
    Given a star identifier, show the scene on-sky

    Parameters
    ----------
    star_id : star identifier for the target star
    stars_table : stars table
    img_size : int [61] size of the field to show, pix
    neighbors : bool [False] if True, label neighbors
    neighbor_tab : pd.DataFrame [None] star table used for looking up neighbors. if None, use stars_table
    ax : matplotlib axis [None] optional axis to draw on
    Output
    ------
    fig : plt.Figure instance
      if ax is given, returns a reference to the parent figure
    """

    if ax == None:
      fig, ax = plt.subplots(1, 1)
    else:
      fig = ax.get_figure()

    ax.set_title(f"{star_id} context on sky")
    ax.set_xlabel("RA [pix]")
    ax.set_ylabel("Dec [pix]")

    mast_ind = image_utils.get_master_stamp_ind(star_id,
                                                stamp_shape=img_size,
                                                df=stars_table,
                                                extra_row=True)
    ylim = mast_ind[0][[0, -1]][:, 0]
    xlim = mast_ind[1][:, [0, -1]][0]
    img = fits.getdata(shared_utils.load_config_path("composite_img_file"))
    img = img[mast_ind[0, :-1, :-1], mast_ind[1, :-1, :-1]]
    ax.pcolor(mast_ind[1]+0.5,
              mast_ind[0]+0.5,
              img,
              norm=mpl.colors.LogNorm())

    db_star_pos = stars_table.query("star_id == @star_id")[['u_mast', 'v_mast']].squeeze()
    ax.scatter(*db_star_pos, marker='*', s=100, c='y', linewidths=1, edgecolors='k')

    # neighbors
    if show_neighbors is True:
        nbr_query = "u_mast >= @xlim[0] and u_mast <= @xlim[1] "\
            "and v_mast >= @ylim[0] and v_mast <= @ylim[1] "\
            "and star_id != @star_id"
        if not isinstance(neighbors_table, pd.DataFrame):
            neighbors_table = stars_table
        nbrs = neighbors_table.query(nbr_query)
        for i, nbr  in nbrs.iterrows():
            nbr_pos = nbr[['u_mast', 'v_mast']].squeeze()
            ax.scatter(*nbr_pos, marker='o', s=100, c='none', linewidths=1, edgecolors='w')
            ax.scatter(*nbr_pos, marker='x', s=50, c='k', linewidths=1)

    stamp_dim = 11
    rectangle = plt.Rectangle(db_star_pos-stamp_dim/2, stamp_dim, stamp_dim, 
                              fc='none',ec="white", transform=ax.transData)
    ax.add_patch(rectangle)
    ax.set_aspect('equal')

    return fig

def show_detector_scene(ps_table, target_ps_ids=None, alt_ps_table=None, ax=None):
    """
    Show the scene on the detector

    Parameters
    ----------
    ps_table : the point source table to draw from
    targ_ps_ids : point source ids [None]
      if given, plant a marker on the average position of this/these ps IDs
    alt_ps_table : [None]
      alternate ps table; plot all point sources that appear only in this one
    ax : plt.axis [None]
      optional: provide the axis to draw on

    Output
    ------
    fig : plt.Figure instance
      if ax is given, returns a reference to the parent figure
    """
    if ax == None:
      fig, ax = plt.subplots(1, 1)
    else:
      fig = ax.get_figure()

    ax.scatter(ps_table['ps_x_exp'], ps_table['ps_y_exp'], marker='.', 
               s=ps_table['ps_phot'], 
               c=ps_table['ps_mag'], cmap=mpl.cm.magma_r)


    target_locs = ps_table.set_index('ps_id').loc[target_ps_ids, ['ps_x_exp', 'ps_y_exp']]
    ax.scatter(target_locs['ps_x_exp'].mean(),
               target_locs['ps_y_exp'].mean(),
               marker='*', s=400, c='none', edgecolors='y', linewidths=2,
               label='target star')

   #all_sec_ids = [i for i in alt_ps_table['ps_id'] if i not in ps_table['ps_id']]
   #all_star_pos = dbm_raw.join_all_tables().query("ps_id in @all_sec_ids").groupby('star_id')[['ps_x_exp', 'ps_y_exp']].mean()
   #ax.scatter(all_star_pos['ps_x_exp'], all_star_pos['ps_y_exp'], marker='x', s=20, c='k', label='cut detections')

    ax.set_aspect('equal')

    ax.legend()

    return fig
