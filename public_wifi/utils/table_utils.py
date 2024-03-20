"""
This module contains the API for interacting with data tables
"""

from pathlib import Path
import h5py
import pandas as pd
import re
import numpy as np
import warnings
import configparser

from astropy.wcs import WCS
from astropy.io import fits

from . import table_io
from . import shared_utils
from . import header_utils


# this dictionary helpfully maps the first letter of an identifier
# to its typical table name
ident_map = {'S': 'star_id', 'P': 'ps_id', 'T': 'stamp_id'}


# Read from the table definitions file
table_definition_file = (Path(__file__).parent.absolute() / "../table_definitions.cfg").resolve()
config = configparser.ConfigParser()

def load_table_definition(
        table_name : str = '',
        config_file : str | Path = table_definition_file
) -> dict:
    """
    Load a path from the config file. If no table name (or bad table name)
    given, prints a list of allowed values. Otherwise, returns a dict of
    {col_name : col_type}.

    Parameters
    ----------
    table_name : str = ''
      the name for the table (see docs)
    config_file : str or Path ("../table_definitions.cfg")
      Path to a config file where the table columns and dtypes are defined

    Output
    -------
    col_dict : dict
        a dictionary of column names and dtypes

    """
    # reread the file whenever the function is called so you don't have to
    # reload the entire module if the config file gets updated
    if table_name == '':
        available_tables = '\n\t'.join(config.sections())
        print(f"No table name provided.")
        print(f"Available tables are: \n\t{available_tables}")
        return {}

    config.read(config_file)
    table_name = table_name.upper()
    try:
        assert(table_name in config)
        # table_cols = list(config[table_name].items())
    except AssertionError:
        available_tables = '\n\t'.join(config.sections())
        print(f"Error, no table named {table_name.upper()}")
        print(f"Available tables are: \n\t{available_tables}")
        return {}

    entries = {}
    for entry in config[table_name].items():
        index = int(re.search(r"\d+", entry[0]).group())
        key = re.search(r"\D+", entry[0]).group()
        val = entry[1]
        if index in entries.keys():
            entries[index].update({key:val})
        else:
            entries[index] = {key: val}
    table = pd.DataFrame.from_dict(entries, orient='index')
    # table_dict = dict([(i[0][1], i[1][1]) for i in zip(table_cols[::2], table_cols[1::2])])
    table_dict = table.set_index("col")["dtype"].to_dict()
    return table_dict

def initialize_table(table_name, nrows):
    """
    Create an empty table based on the definitions found in table_definitions.cfg

    Parameters
    ----------
    table_name : str
      A valid table name
    nrows : the length of the table

    Output
    ------
    table : pd.DataFrame
      the specified table

    """
    table_cols = load_table_definition(table_name)
    table = pd.DataFrame(data=None,
                         columns=table_cols.keys(),
                         index=range(nrows))
    for col in table_cols.keys():
        table[col] = table[col].astype(table_cols[col])
    return table


def list_available_tables(db_file, return_list=False):
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
    with h5py.File(db_file, mode='r') as f:
        table_names = sorted(f.keys())
    f.close()
    if return_list == True:
        return table_names
    else:
        print(f"Available tables in {db_file}:")
        print('\n'.join(table_names))



def get_file_name_from_file_id(file_id, path):
    """
    The header files don't store the whole filename, so this function fills in
    the rest of the name as well as the path.

    Parameters
    ----------
    file_id : str
      the file identifier (everything except _flt.fits)
    path : str or Path
      path to the folder where the FITS files are kept

    Returns
    -------
    filename : pathlib.Path
      the full absolute path to the fits file
    """
    suffix = "_flt.fits"
    filename = Path(path).absolute() / (file_id + suffix)
    try:
        assert filename.exists()
    except AssertionError:
        print(f"Error: file_id {file_id} resolves to {filename}, which is not found")
        return None
    return filename.absolute()


"""
Helpers for getting file and filter names.
Since these queries are run often, this simplifies the process.
"""

def get_file_name_from_exp_id(exp_id, file_mapper, root=False):
    """
    Given a KS2 file identifier, get the name of the FLT file

    Parameters
    ----------
    exp_id : str
      The identifier assigned to each exposure file. Typically E followed by a number.
    file_mapper : pd.DataFrame (default: result of table_utils.get_file_mapper())
      The dataframe that tracks the file IDs and the corresponding file names
    root : bool [False]
      if True, return the root name (i.e. filename without the _flt.fits extension)

    Returns
    -------
    file_name : the HST identifier for the file
    """
    exp_id = exp_id.upper()
    querystr = f"exp_id == '{exp_id}'"
    file_name = file_mapper.query(querystr)['file'].values[0]
    if root == True:
        file_name = file_name.split("_flt.fits")[0]
    return file_name


def get_filter_name_from_filter_id(filter_id, filter_mapper):
    """
    Given a filter identifier, get the name of the HST filter

    Parameters
    ----------
    filter_id : str
      The identifier assigned to each filter by the database.
      Typically F followed by a number, like F1 or F2.
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
def get_img_from_exp_id(exp_id, hdr, db_file, config_file):
    """
    Pull out an image from the fits file, given the exposure identifier.

    Parameters
    ----------
    exp_id : str
      the identifier for the exposure (usu something like 'E001')
    hdr : str or int 
      which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']
    db_file : str or Path
      path to the active database file

    Returns:
    img : numpy.array
      2-D image from the fits file
    """
    file_mapper = table_io.load_table("lookup_files", db_file)
    flt_name = get_file_name_from_exp_id(exp_id, file_mapper)
    flt_path = shared_utils.get_data_file(flt_name, config_file)
    img = fits.getdata(flt_path, hdr)
    return img

def get_img_from_file_id(file_id, hdr, db_file, config_file):
    """
    Pull out an image from the fits file, given the exposure identifier.

    Parameters
    ----------
    file_id : str
      the identifier for the file/exposure (usu something like 'E001')
    hdr : str or int ['SCI']
      which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']
    db_file : str or Path
      path to the active database file
    config_file : str or Path
      config file that tells you where the FITS files are

    Returns:
    img : numpy.array
      2-D image from the fits file
    """
    file_mapper = table_io.load_table("lookup_files", db_file)
    flt_name = get_file_name_from_file_id(file_id, file_mapper)
    flt_path = shared_utils.get_data_file(flt_name, config_file)
    img = fits.getdata(flt_path, hdr)
    return img

def get_hdr_from_exp_id(exp_id, hdr, db_file):
    """
    Pull out a header from the fits file, given the exposure identifier.

    Parameters
    ----------
    exp_id : str
      the identifier for the exposure (usu something like 'E001')
    hdr : str or int ['SCI']
      which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']

    Returns:
    hdr : fits header object
    """
    flt_name = get_file_name_from_exp_id(exp_id)
    flt_path = shared_utils.get_data_file(flt_name)
    hdr = fits.getheader(flt_path, hdr)
    return hdr


"""
If the stamps are written to file, you can just find them using the ps_id or stamp_id
"""

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
def load_header(extname, db_file):
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
        dfs = {e.lower(): table_io.load_table(f"hdr_{e.lower()}", db_file=db_file)
               for e in header_utils.all_headers}
        return dfs
    # otherwise, check that the input is OK
    try:
        assert(extname.upper() in header_utils.all_headers)
    except AssertionError:
        print(f"{extname} not one of {all_headers}, please try again.")
        return None
    df = table_io.load_table(f"hdr_{extname.lower()}", db_file=db_file)
    return df


#################################
# Table manipulation and lookup #
#################################
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


def create_database_subset(list_of_stars, tables=None):
    """
    Load a subset of the database: pick a subset of stars to work on, and make
    sure you have a set of tables that are self-consistent

    Parameters
    ----------
    list_of_stars: list (or iterable)
      a list of star_ids to use for generating a new database
    tables : list [None]
      list of tables (as dataframes) to include in the subset parsing.
      Must have the star_id stored in a column. If None, then
      ['stars', 'point_sources'] is used and tables are read from file.

    Output
    ------
    subset_tables : list
      a list of tables - one for each that was passed in, and in the same order

    """
    if tables is None:
        tables = [table_io.load_table(t) for t in ['stars', 'point_sources']]
    subset_tables = []
    for table in tables:
        star_col = shared_utils.find_star_id_col(table.columns)
        #subset_tables.append(index_into_table(table, star_col, list_of_stars))
        subset_tables.append(table.query(f"{star_col} in @list_of_stars"))
    return subset_tables


def get_header_kw_for_exp(exp_id, kw, hdr_id, db_file):
    """
    Shortcut to get the value of a particular header keyword for an exposure

    Parameters
    ----------
    exp_id : str
      exposure identifier (e.g. E123)
    kw : str
      header keyword whose value you want
    hdr: str 
      which header to search for the keyword
    db_file : str or Path
      path to the database file

    Output
    ------
    kw_val : the value stored at the header keyword
    """

    #hdr = load_header(hdr_id, db_file)
    hdr = table_io.load_table('hdr_'+hdr_id, db_file)
    file_mapper = table_io.load_table('lookup_files', db_file)
    filename = get_file_name_from_exp_id(exp_id, file_mapper, root=True)
    try:
        kw_val = hdr.query('rootname == @filename')[kw.lower()].squeeze()
    except:
        print(f"Keyword `{kw} == {filename}` not found in {hdr_id} header")
        return None
    return kw_val


def get_wcs_from_exp_id(exp_id, file_mapper, config_file):
    """
    Given an exposure ID, get the WCS header for the file

    Parameters
    ----------
    exp_id : str
      a string of format E000 corresponding to an exposure that comes from
      a unique fits file
    file_mapper : pd.DataFrame
      dataframe linking the exposure IDs to the file names
    config_file : str or Path
      path to the config file that stores folder where the FITS files are kept

    Output
    ------
    wcs : a WCS object 

    """
    file_name = get_file_name_from_exp_id(exp_id, file_mapper)
    file_path = shared_utils.get_data_file(file_name, config_file)
    wcs = WCS(fits.getheader(file_path, 'SCI'))
    return wcs

####################
# Catalog cleaning #
####################
# see cleaning_utils.py


###########
# Setters #
###########

def set_reference_quality_flag(stamp_ids, flag=True, stamp_table=None):
    """
    Set the reference quality flag for the given stamp ids.
    True -> stamp *can* be used as a reference
    False -> stamp *cannot* be used as a reference

    Parameters
    ----------
    stamp_ids : string or list-like
      one or more stamp IDs whose reference flags need to be set.
    flag : bool [True]
      the value of the flag
    stamp_table : pd.DataFrame [None]
      the table to modify, passed by reference. If None, read from the default file

    Output
    ------
    None: stamp_table is modified in-place

    """
    if isinstance(stamp_ids, str):
        stamp_ids = [stamp_ids]
    if not isinstance(stamp_table, pd.DataFrame):
        print("None value for stamp_table not yet enabled, quitting")
        return
    ind = stamp_table.query("stamp_id in @stamp_ids").index
    stamp_table.loc[ind, 'stamp_ref_flag'] = flag
    # done
