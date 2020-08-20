"""
Includes shared useful stuff, like path definitions and file formats
"""

import re
from pathlib import Path
import configparser

#########
# PATHS #
#########
# the paths are stored in the config file
config_file = (Path(__file__).parent.absolute() / "../config.ini").resolve()
config = configparser.ConfigParser()
config.read(config_file)
"""
This block defines some useful paths, as well as a wrapper function for loading paths from the config file and handling them properly, like turning relative paths into absolute paths
"""
def load_config_path(key):
    """
    Load a path from the config file. Also handles case of key not found
    Parameters
    ----------
    key : str
      a key in the PATHS section of the config file
    Returns
    -------
    path : pathlib.Path or None
      absolute path to the target in the config file. Prints warning if the path does not exist.
    """
    key = key.upper()
    path = Path(config['PATHS'][key]).resolve()
    # test that the path exists
    try:
        assert(path.exists())
    except AssertionError:
        print(f"Error: {path} not found.")
    return path

# HST data files for manipulation
data_path = load_config_path("data_path")
# Database tables
table_path = load_config_path("table_path")
db_file = load_config_path("db_file")
# Gaia catalog and source matches
align_path = load_config_path("align_path")
# KS2 output files
ks2_path = load_config_path("ks2_path")

#############
# FLT files #
#############
"""
Here's a helper function for loading a data file
"""
def get_data_file(file_name):
    """
    Given a file name (no path), return the full path the file so it can be read

    Parameters
    ----------
    file_name : str or pathlib.Path
      the file name, e.g. icct01hrq_flt.fits

    Returns
    -------
    file_path : pathlib.Path
      the full file path
    """
    file_path = data_path / file_name
    try:
        assert(file_path.exists())
    except AssertionError:
        print(f"Error: {file_path.as_posix()} does not exist.")
        return None
    return file_path.resolve()



def find_star_id_col(columns):
    """
    Given a list of columns, this function returns the name of the column that
    contains the star_id identifier (e.g. star_id, ps_star_id, stamp_star_id)

    Parameters
    ----------
    columns : list-like
      list of columns

    Output
    ------
    star_col : str
      name of the column with the star_id

    """
    qstr = 'star_id'
    regex = re.compile(qstr)
    matches = [i for i in columns if regex.search(i) is not None]
    if len(matches) > 0:
        return matches[0]
    else:
        return None

def find_column(columns, qstr):
    """
    Given a list of columns, this function returns the name of the column that
    contains the string in qstr

    Parameters
    ----------
    columns : list-like
      list of columns
    qstr : string to search for (e.g. star_id, u_mast, v_mast)

    Output
    ------
    col : str
      name of the desired column

    """
    regex = re.compile(qstr)
    matches = [i for i in columns if regex.search(i) is not None]
    if len(matches) > 0:
        return matches[0]
    else:
        return None
