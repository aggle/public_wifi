"""
Holder for plots that you make repeatedly
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

from . import table_utils
from . import shared_utils
from . import plot_utils
from . import db_manager




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
 
