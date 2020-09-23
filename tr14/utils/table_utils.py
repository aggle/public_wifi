"""
This module contains the API for interacting with data tables
"""

import csv
from pathlib import Path
import h5py
import pandas as pd
import re
import numpy as np
import warnings

from astropy.io import fits

from . import shared_utils, header_utils


def write_table(table, key, kwargs={}, db_file=shared_utils.db_file, verbose=True):
    """
    Add a table to the DB file `shared_utils.db_file`. Really just a simple
    wrapper so I don't have to keep typing shared_utils.db_file every time
    I need to write something

    Parameters
    ----------
    table : pd.DataFrame
      pandas DataFrame (or Series) containing the data you want to store
    key : str
      key for the table in the HSF file
    kwargs : dict [{}]
      any keyword arguments to pass to pd.DataFrame.to_hdf for DataFrames and
      Series, or to h5py.File for non-pandas objects
    db_file : str or Path [shared_utils.db_file]
      path to the database file
    verbose : bool [True]
      print some output
    Output
    -------
    Nothing; writes to file
    """
    # this throws a performance warning when you store python objects with
    # mixed or complex types in an HDF file, but I want to ignore those
    with warnings.catch_warnings() as w:
        kwargs['mode'] = kwargs.get("mode", 'a')
        if hasattr(table, 'to_hdf'):
            table.to_hdf(db_file, key=key, **kwargs)
            if verbose == True:
                print(f"Table {key} written to {str(db_file)}")
        else:
            print("Error: cannot currently store non-pandas types")
            """
            with h5py.File(db_file, **kwargs) as store:
                try:
                    store[key] = table
                except KeyError:
                    grp = store.create_group(key)
                store.close()
            """
    

def load_table(key, db_file=shared_utils.db_file):
    """
    Load a table.

    Parameters
    ----------
    key : string or list
    db_file : path to the database file that holds the table

    Output
    ------
    df : pd.DataFrame or dict of dataframes
      the table (or, if `key` is a list, then a dict of dataframes)
    """
    # df = None
    if isinstance(key, list):
        df = {}
        for k in key:
            # drop the leading /, if relevant
            if k[0] == '/': k = k[1:]
            try:
                df[k] = pd.read_hdf(db_file, k)
            except KeyError:
                print(f"Error: Key `{key}` not found in {str(db_file)}")
                df[k] = None
    else:
        try:
            df = pd.read_hdf(db_file, key)
        except KeyError:
            print(f"Error: Key `{key}` not found in {str(db_file)}")
            df = None

    return df


def list_available_tables(return_list=False, db_file=shared_utils.db_file):
    """
    Print a list of tables and their descriptions. Alternately, return a list of the tables.
    TODO print the subtables, too
    Parameters
    ----------
    return_list: bool [False]
      if True, instead of printing out the table names, return a list.

    Output
    ------
    table_names : list [optional]
      a list of available keys
    """
    with pd.HDFStore(db_file, mode='r') as store:
        table_names = sorted(store.keys())
        if return_list == False:
            print(f"Available tables in {db_file}:")
            print('\n'.join(table_names))
        store.close()
    if return_list == True:
        return table_names



def get_file_from_file_id(file_id):
    """
    The header files don't store the whole filename, so this function fills in
    the rest of the name as well as the path.

    Parameters
    ----------
    file_id : str
      the file identifier (everything except _flt.fits)

    Returns
    -------
    filename : pathlib.Path
      the full absolute path to the fits file
    """
    suffix = "_flt.fits"
    filename = shared_tils.data_path.absolute() / (file_id + suffix)
    return filename.absolute()


"""
Helpers for getting file and filter names.
Since these queries are run often, this simplifies the process.
"""
file_mapper = load_table("lookup_files")
filter_mapper = load_table("lookup_filters")

def get_file_name_from_exp_id(exp_id):
    """
    Given a KS2 file identifier, get the name of the FLT file

    Parameters
    ----------
    exp_id : str
      The identifier assigned to each exposure file. Typically E followed by a number.
    file_mapper : pd.DataFrame (default: result of table_utils.get_file_mapper())
      The dataframe that tracks the file IDs and the corresponding file names

    Returns
    -------
    file_name : the HST identifier for the file
    """
    exp_id = exp_id.upper()
    querystr = f"file_id == '{exp_id}'"
    file_name = file_mapper.query(querystr)['file_name'].values[0]
    return file_name


def get_filter_name_from_filter_id(filter_id):
    """
    Given a filter identifier, get the name of the HST filter

    Parameters
    ----------
    filter_id : str
      The identifier assigned to each filter by the database. Typically F followed by a number, like F1 or F2.
    filter_mapper : pd.DataFrame (default: result of ks2_utils.get_filter_mapper())
      The dataframe that tracks the filter IDs and the corresponding filter names

    Returns
    -------
    filter_name : the HST name of the filter
    """
    filter_id = filter_id.upper()
    querystr = f"filter_id == '{filter_id}'"
    filter_name = filter_mapper.query(querystr)['filter_name'].values[0]
    return filter_name



"""
Helpers for getting exposures/images and stamps
"""
def get_img_from_file_id(file_id, hdr='SCI'):
    """
    Pull out an image from the fits file, given the exposure identifier.

    Parameters
    ----------
    file_id : str
      the identifier for the file/exposure
    hdr : str or int ['SCI']
      which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']

    Returns:
    img : numpy.array
      2-D image from the fits file
    """
    flt_name = get_file_name_from_file_id(file_id)
    flt_path = shared_utils.get_data_file(flt_name)
    img = fits.getdata(flt_path, hdr)
    return img


# def get_stamp_from_ps_row(row, stamp_size=11, return_img_ind=False, hdr='SCI'):
#     """
#     Given a row of the FIND_NIMFO dataframe, this gets a stamp of the specified
#     size of the given point source
#     TODO: accept multiple rows

#     Parameters
#     ----------
#     row : pd.DataFrame row
#       a row containing the position and file information for the source
#     stamp_size : int or tuple [11]
#       (row, col) size of the stamp [(int, int) if only int given]
#     return_img_ind : bool (False)
#       if True, return the row and col indices of the stamp in the image
#     hdr : str or int ('SCI')
#       each HDU in the flt Duelist has a different kind of image - specify
#       which one you want here

#     Returns
#     -------
#     stamp_size-sized stamp
#     """
#     img = get_img_from_file_id(row['ps_exp_id'], hdr=hdr)
#     # location of the point source in the image
#     xy = row[['ps_x_exp','ps_y_exp']].values
#     # finally, get the stamp (and indices, if requested)
#     return_vals = image_utils.get_stamp(img, xy, stamp_size, return_img_ind)
#     return return_vals

"""
If the stamps are written to file, you can just find them using the ps_id or stamp_id
"""
def get_stamp_from_id(ident, stamp_df=None):
    """
    Given a stamp or point source identifier, pull the corresponding stamp from
    the database file /stamp_arrays/ group

    Parameters
    ----------
    ident: str
      stamp or point source identifier ([S/T]000000)

    Output
    ------
    stamp_array: np.array
      array corresponding to the requested stamp

    """
    # force the identifier to the stamp format, i.e. first letter "T"
    if ident[0] == 'S':
        ident.replace("S","T")

    if stamp_df == None:
        stamp_df = load_table('stamps')
    stamp_array = stamp_df.set_index('stamp_id').loc[ident, 'stamp_array']
    return stamp_array



def get_stamp_coords_from_center(x, y, stamp_size):
    """
    Docstring goes here

    Parameters
    ----------
    x : int
      x/col index
    y : int
      y/row index
    stamp_size : int or tuple [11]

    Output
    ------
    stamp_ind : np.array
      2xN array of (row, col) coordinates

    """
    pass


"""
Shortcut for loading the FITS headers
"""
def load_header(extname='pri'):
    """
    Helper function to load the right header file

    Parameters
    ----------
    extname : str [pri]
      shorthand name for the extension whose dataframe you want
      options are: pri, sci, err, dq, samp, and time
    Returns
    -------
    df : pd.DataFrame
      a dataframe with the right header selected
    """
    # special case
    if extname == 'all':
        # return all the headers
        print("Returning all headers.")
        dfs = {e.lower(): load_table(f"hdr_{e.lower()}")
               for e in header_utils.all_headers}
        return dfs
    # otherwise, check that the input is OK
    try:
        assert(extname.upper() in header_utils.all_headers)
    except AssertionError:
        print(f"{extname} not one of {all_headers}, please try again.")
        return None
    df = load_table(f"hdr_{extname.lower()}")
    return df


######################
# Table manipulation #
######################

def query_table(table, query, return_column=None):
    """
    Pass a query to the given table.
    Parameters
    ----------
    table : pd.DataFrame
      the table to query
    query : string
      a string with proper query syntax
    return_column : string (optional, default is None)
      the column whose value you want to return. If left as default (None),
      returns the full (queried) dataframe

    Returns
    -------
    return_value : a series (or dataframe) of all the values that match the query
    """
    return_value = table.query(query)
    if return_column is not None:
        return_column = return_column.lower()
        return_value = return_value[return_column]
    return return_value


def match_value_in_table(table, query_col, match_val, return_col=None):
    """
    Retrieve a value from a table. Assumes the equality operator ('==')
    Remember to cast any field names into lower case
    Parameters
    ----------
    table : pd.DataFrame
      the table to query
    query_col: string
      the name of the column to query
    match_val: type(query_column)
      the value to match in the query column
    return_col : string (optional, default is None)
      the column whose value you want to return. If left as default (None),
      returns the full matching dataframe/row

    Returns
    -------
    return_value : a series (or dataframe) of all the values that match the
      query. If only one item matches the query, returns the item itself (i.e.
      not in a dataframe/series.)

    """
    # all columns are in lowercase
    #query_column = query_column.lower()
    return_val = table.query(f"{query_col} == @match_val")
    if return_col is not None:
        #return_col = return_column.lower()
        return_val = return_val[return_col]
    return return_val.squeeze()



def set_value():
    """
    Set table value
    """
    pass

def index_into_table(table, index_col, values):
    """
    Shortcut for pulling out a table subset of `values` matching entries in the
    `index_col` of `table`.
    Basically a wrapper for table.set_index(index_col).loc[values]

    Parameters
    ----------
    table : pd.DataFrame
      a table you want a subset of
    index_col : str
      the table column you want to filter
    values : the values in index_col you want to match

    Output
    ------
    table_subset : pd.DataFrame
      subset of the input table containing only rows whose entry in the
      index_col are in `values`

    """
    table_subset = table.set_index(index_col).loc[list(values)]
    table_subset = table_subset.reset_index()
    return table_subset


def create_database_subset(list_of_stars, table_names=None):
    """
    Load a subset of the database: pick a subset of stars to work on, and make
    sure you have a set of tables that are self-consistent

    Parameters
    ----------
    list_of_stars: list (or iterable)
      a list of star_ids to use for generating a new database
    table_names : list [None]
      list of tables to include in the subset parsing. must have the star_id
      stored in a column. If None is ['stars', 'point_sources'] is used

    Output
    ------
    table_dict : dict
      a dictionary of tables - one entry for each table in the database file.
      The table key is the same as the file key

    """
    if table_names is None:
        table_names = ['stars', 'point_sources']
    table_dict = {}
    for t in table_names:
        table = load_table(t)
        star_col = shared_utils.find_star_id_col(table.columns)
        table_dict[t] = index_into_table(table, star_col, list_of_stars)
        del table
    return table_dict


def get_working_catalog_subset():
    """
    This is the canonical working catalog subset. It is defined as all stars
    between u=[600, 1000] and v=[800, 1200] in the master frame of reference.
    There are 228 unique stars and 9807 point sources, about 10% of the total
    catalog. This function accepts no arguments because it shouldn't change

    Parameters
    ----------
    none

    Output
    ------
    subset_tables : dict
      dictionary of tables with the stars, point_sources, and stamps subsets
    """
    # load the master catalog
    stars_table = load_table("stars")
    # select subset
    u_range = (600, 1000)
    v_range = (800, 1200)
    # generate the query string
    qstr = (f"u_mast >= @u_range[0] and u_mast <= @u_range[1]"
            "and v_mast >= @v_range[0] and v_mast <= @v_range[1]")
    subset_tables = create_database_subset(stars_table.query(qstr)['star_id'],
                                           ['stars','point_sources','stamps'])
    return subset_tables


