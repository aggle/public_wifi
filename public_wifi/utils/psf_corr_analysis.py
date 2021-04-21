"""
This module contains functions for *investigating* and plotting PSF correlation.
It is *not* a container for utilities for calulating correlations
"""
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from . import shared_utils
from . import db_manager
from . import subtr_utils
from . import plot_utils
from . import image_utils

figpath = Path("./figs/psf_corr/")
db_file = shared_utils.db_clean_file


def load_full_corr_mat():
    """
    Load the *full* correlation matrix from file. Warning: Large!
    returns full_corr_mat dataframe indexed by corr method and stamp id
    """
    return pd.read_hdf(shared_utils.full_corr_mat_file, mode='r')



################
# Useful plots #
################
def hist_stamps_per_star(dbm, ax=None):
    """
    histogram the number of stamps per star

    Parameters
    ----------
    dbm : db_manager.DBManager instance

    Output
    ------
    fig : plt.figure() object

    """
    matching_stars = dbm.find_matching_id(dbm.stamps_tab['stamp_id'], 'star').reset_index()
    gb_star = pd.merge(dbm.stamps_tab, matching_stars).groupby('star_id')
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    nstamps = gb_star.size()
    ax.hist(nstamps, bins=np.arange(nstamps.min(), nstamps.max()+1),
            log=True);
    ax.set_ylabel("# stars")
    ax.set_xlabel("# stamps per star")
    return fig


def hist_mag_per_filter(dbm, ax=None):
    """
    Histogram the number of stars in each magnitude bin for each filter

    Parameters
    ----------
    dbm : db_manager.DBManager instance

    Output
    ------
    fig : plt.figure() instance

    """
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    hist_args = {'histtype':'step', 'bins':np.arange(-13.5, -2)}
    ax.hist(dbm.stars_tab['star_mag_F1'],
            color='C0', 
            label=dbm.lookup_dict['lookup_filters'].query('filter_id == "F1"')['filter_name'], 
            **hist_args)
    ax.hist(dbm.stars_tab['star_mag_F2'],
            color='C1', 
            label=dbm.lookup_dict['lookup_filters'].query('filter_id == "F2"')['filter_name'],
            **hist_args)
    ax.legend(loc='upper left')
    ax.set_xlabel("instr. mag")
    ax.set_ylabel("# stars")
    return fig


def show_correlation_matrix(corr_mat, ax_title='', ax=None):
    """
    Plot a correlation matrix

    Parameters
    ----------
    corr_mat : pd.DataFrame correlation matrix
    corr_func_name : string name of correlation function (e.g. 'mse')
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

    ax.set_title(ax_title)
    imax = ax.imshow(corr_mat)
    fig.colorbar(imax, ax=ax)

    return fig

