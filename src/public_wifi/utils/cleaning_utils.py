"""
These functions are used to clean the catalogs:
- perform magnitude cuts
- perform color cuts (i.e. membership)
- identify visual binaries
- check stamps for hot pixels
- check stamps for NaNs

Probably I should have a table that has a column of True/False for each cut, e.g.
| star id | mag_cut | binary_cut | membership_cut |
|-------------------------------------------------|
| S000001 | True    |   False    |     False      | <- cut for being too faint
| S000002 | False   |   True     |     False      | <- cut for being a visual binary
| S000003 | False   |   False    |     True       | <- cut for being a field object

"""

from pathlib import Path
import pandas as pd
import re
import numpy as np
import warnings

from astropy.io import fits

from . import shared_utils
from . import header_utils
from . import table_utils
from . import distance_utils

tables = {
    'stars': None,
    'point_sources': None,
    'stamps': None,
    'companion_status': None
}

def clean_visual_binaries(star_table, thresh=7):
    """
    Find stars that are closer to each other (in units of pixels) than some
    threshold, and return their IDs

    Parameters
    ----------
    star_table: pd.DataFrame
      catalog of stars and their positions
    thresh : int [7]
      distance threshold. Stars with separations <= this value are flagged

    Output
    ------
    visual_binaries : pd.Series
      list of star_ids for stars that are visual binaries

    """
    dist_mat = distance_utils.calc_distance_matrix(star_table, set_nan=True)
    visual_binaries = dist_mat[dist_mat.min() <= thresh].index
    return visual_binaries


def generate_cleaned_tables(sep_thresh=7,
                            mF127M_thresh = -6,
                            color_thresh = 0.3,
                            nan_stamps=True,
                            db_file=shared_utils.db_clean_file):
    """
    Wrap all the cleaning steps in this function. For each cut, trim the
    Enter np.nan for any argument to skip that cut

    Parameters
    ----------
    sep_thresh : int [7]
      stars closer to each other in pixels than this distance are labeled
      binaries and rejected
    mF127M_thresh : float [-6]
      keep stars brighter than this in the F127M filter
    nan_stamps : bool [True]
      if True, drop any stamps with NaN's in them

    Output
    ------
    cleaned_tables : dict
      dictionary of cleaned tables

    """
    clean_tables = {}
    # first, load the full catalog and a copy for cleaning
    for k in tables.keys():
        tables[k] = table_io.load_table(k, db_file=db_file)
        clean_tables[k] = table_io.load_table(k, db_file=db_file)

    # remove visual contaminants
    if np.isnan(sep_thresh):
        print("Skipping visual binaries cut")
    else:
        visual_binaries = clean_visual_binaries(tables['stars'], sep_thresh)
        qstr = 'comp_star_id in @visual_binaries'
        for t in [tables, clean_tables]:
            t['companion_status'].loc[t['companion_status'].query(qstr).index,
                                  'companion_status'] = 1
        # apply visual binary cut
        clean_tables['stars'] = tables['stars'].query('star_id not in @visual_binaries')

    # magnitude cut
    if np.isnan(mF127M_thresh):
        print("Skipping magnitude cut")
    else:
        clean_tables['stars'] = clean_tables['stars'].query('star_mag_F1 <= @mF127M_thresh')

    # color cut
    if np.isnan(color_thresh):
        print("Skipping color cut")
    else:
        color_func = lambda x: (x['star_mag_F1'] - x['star_mag_F2']) <= color_thresh
        #color = clean_tables['stars'].set_index('star_id').apply(color_func, axis=1)
        #color_cut_stars = color[color <= color_thresh].index
        #clean_tables['stars'] = clean_tables['stars'].query('star_id in @color_cut_stars')
        color_cut = clean_tables['stars'].apply(color_func, axis=1)
        clean_tables['stars'] = clean_tables['stars'][color_cut]

    # NaN stamps
    if nan_stamps is not True:
        print("Skipping NaN stamp cut")
    else:
        not_nan_ind = clean_tables['stamps']['stamp_array'].apply(lambda x: ~np.any(np.isnan(x)))
        keep_stars = clean_tables['stamps'].loc[not_nan_ind, 'stamp_star_id'].unique()
        clean_tables['stars'] = clean_tables['stars'].query('star_id in @keep_stars')

    # lastly, sync the point source and stamp tables with the cleaned stars table
    clean_tables['point_sources'], clean_tables['stamps'] = \
        table_utils.create_database_subset(clean_tables['stars']['star_id'],
                                           [clean_tables['point_sources'],
                                            clean_tables['stamps']])
    return clean_tables
