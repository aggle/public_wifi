"""
Methods for calculating and analyzing separations between point sources and stars
"""

import numpy as np
import pandas as pd

from . import shared_utils, table_utils

"""
It is useful to quickly calculate the distances of objects from each other,
in a particular exposure. The following methods do that:
calc_exp_dist_from_obj efficiently gets the distances from a particular object, and
generate_exposure_distance_matrix uses this to get the distances of all objects
from each other
"""
def calc_exp_dist_from_obj(df, obj_id, set_nan=True):
    """
    Given a dataframe of a single exposure and a particular object,
    calculate the pixelwise distance of every other point source in that
    exposure from the specified object.

    Parameters
    ----------
    df : pd.DataFrame
      pandas dataframe for a single exposure with only one entry per
      astrophysical object
    obj_id : str
      identifier for the astrophysical object
    set_nan : bool [True]
      if True, set the distance of the object from itself to nan instead of 0
    """
    star_col = shared_utils.find_star_id_col(df.columns)
    obj_row = df.query('ps_star_id == @obj_id').squeeze()
    obj_pos = obj_row[['ps_x_exp', 'ps_y_exp']]
    diff = df[['ps_x_exp','ps_y_exp']] - obj_row[['ps_x_exp','ps_y_exp']]
    # now set the index to be the obj_id
    # this is a little convoluted but it uses pandas' automatic
    # index matching to make sure the distances line up with the right
    # objects
    diff['ps_star_id'] = df['ps_star_id']
    diff.set_index('ps_star_id', inplace=True)
    dist = diff.apply(np.linalg.norm, axis=1)

    if set_nan == True:
        dist.loc[obj_id] = np.nan
    return dist

def generate_exposure_distance_matrix(df, same_nan=True):
    """
    Given a KS2 dataframe, compute a symmetric matrix of pixelwise distances.
    The dataframe should only contain point sources from the same exposure.
    this goes really slowly as the size of the dataframe increases; I'm not sure why

    Parameters
    ----------
    df : pd.DataFrame
      dataframe that at least contains the columns [NMAST, xraw1, and yraw1]
    same_nan : bool (False)
      if True, set the diagonal to nan instead of 0

    Returns
    -------
    dist_mat : pd.DataFrame
      N sources x N sources dataframe, where the index and columns are
      the star identifiers, containing the pixelwise distance between sources

    """

    # use apply with fancy indexing to avoid duplicate calculations
    dist_mat = df.apply(lambda x: calc_exp_dist_from_obj(df.loc[x.name:],
                                                         x['ps_star_id'],
                                                         True),
                        axis=1)
    # use the star_ids for the indexing
    dist_mat.index = dist_mat.columns
    # if the distance is 0, set to nan
    dist_mat[dist_mat == 0] = np.nan

    return dist_mat


"""
These functions do the same, but using the master catalog instead of the point
source catalog. This means that some stars may not be in the same exposure.
"""
def calc_dist_from_obj(df, star_id, set_nan=True):
    """
    Given a stars table and a star_id, calculate the pixelwise distance of every
    other point source in that exposure from the specified object.

    Parameters
    ----------
    df : pd.DataFrame
      pandas dataframe for a single exposure with only one entry per
      astrophysical object
    star_id : str
      identifier for the astrophysical object
    set_nan : bool [True]
      if True, set the distance of the object from itself to nan instead of 0
    """
    id_col = shared_utils.find_column(df.columns, 'star_id')
    u_col =  shared_utils.find_column(df.columns, 'u_mast')
    v_col =  shared_utils.find_column(df.columns, 'v_mast')

    uv = df.set_index(id_col).loc[star_id][[u_col, v_col]]
    diff = df.set_index(id_col)[[u_col, v_col]] - uv
    dist = diff.apply(np.linalg.norm, axis=1)

    if set_nan == True:
        dist.loc[star_id] = np.nan

    return dist


def calc_distance_matrix(df, set_nan=True):
    """
    Calculate the pairwise distance matrix for the dataframe, in the master frame.
    This works for the point source table if you slice it to be a single exposure

    Parameters
    ----------
    df : dataframe with columns containing the star_id, u_mast, and v_mast values
    set_nan : bool [True]
      if True, set the diagonal to NaN instead of 0
    Output
    ------
    dist_mat : pd.DataFrame
      dataframe of pairwise distances in the master frame of reference
    """
    # get the relevant columns
    id_col = shared_utils.find_column(df.columns, 'star_id')
    u_col =  shared_utils.find_column(df.columns, 'u_mast')
    v_col =  shared_utils.find_column(df.columns, 'v_mast')

    # cut the dataframe down to only the values you need
    tmp = df.set_index(id_col)[[u_col, v_col]]
    dist_mat = tmp.apply(lambda x: pd.Series(np.linalg.norm(x - tmp, axis=1)),
                     axis=1)
    if set_nan == True:
        # set the diagonal to nan
        diag = np.diag_indices_from(dist_mat)
        dist_mat.values[diag[0], diag[1]] = np.nan
    dist_mat.columns = dist_mat.index
    return dist_mat


"""
I have found it helpful to collect all the neighbors of an a star -- in a
particular exposure -- within some radius. This helps with plotting to see
if the nearby point sources are real or spurious.
"""
def get_neighbors(star_id, radius, stars_table=None):
    """
    Given a star, use the master catalog to get the neighbors in a radius

    Parameters
    ----------
    star_id: str
      stars:star_id identifier for the primary
    radius: int
      search radius in pixels
    stars_table: pd.DataFrame [None]
      table to search for neighbors. Must have columns ending in star_id,
      u_mast, and v_mast. If None, reads "stars" table from database file

    Output
    ------
    neighbor_ids: list
      list of star_id values for the neighbors

    """
    if stars_table is None:
        stars_table = table_utils.load_table("stars")
    dist = calc_dist_from_obj(stars_table, star_id, set_nan=True)
    neighbor_ids = dist[dist <= radius].index
    return neighbor_ids



def get_exposure_neighbors(catalog, obj_id, exp_id, radius):
    """
    Get all the point sources within a certain radius of an object in a
    particular exposure. This tests the selected obj_id's xraw1 and yraw1
    against those of the rest of the objects in that exposure.
    Returns a dataframe of the neighbors.

    Parameters
    ----------
    catalog : pd.DataFrame
      a catalog of point sources
    obj_id : str
      the stellar object identifier. Only one should be present in any exposure
    exp_id : str
      identifier for the exposure
    radius : int
      distance in pixels from the object to keep

    Returns
    -------
    neighbor_df : pd.DataFrame
      catalog entries for the neighbors
    """
    # untested
    # first, make sure the catalog is a single exposure
    exp_df = catalog.query("exp_id == @exp_id")
    dist = calc_exp_dist_from_obj(exp_df, obj_id)
    neighbors = dist[dist <= radius].index
    neighbor_df = exp_df[exp_df['NMAST'].isin(neighbors)]
    return neighbor_df


