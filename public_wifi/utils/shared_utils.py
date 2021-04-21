"""
Includes shared useful stuff, like path definitions and file formats
"""

import configparser
import re
from pathlib import Path
import os
import abc
import yaml
from collections import defaultdict

# universal debug printer
def debug_print(debug_flag=True, *debug_args):
    """
    Easy-to-enable/disable function for debug printing.

    Parameters
    ----------
    debug_flag : bool [True]
      Turns printing on and off. You can set a flag at the top of the module, or
      turn on/off each statement individually.
    debug_args : arbitrary number of arguments, they get passed to print()
      Optional string to print. If empty, just prints 'DEBUG'
    """
    if debug_flag == True:
        print('START DEBUG ----- ')
        print(debug_args)
        print('END DEBUG ----- ')
    else:
        pass

#########
# PATHS #
#########
# the paths are stored in the config file
config_file = (Path(__file__).parent.absolute() / "../config.cfg").resolve()
"""
This block defines some useful paths, as well as a wrapper function for loading paths from the config file and handling them properly, like turning relative paths into absolute paths
"""
def load_config_path(sec, key, as_str=False, config_file=config_file):
    """
    Load a path from the config file. Also handles case of key not found.
    Run with empty strings for list of options.

    Parameters
    ----------
    sec : str
      the section in the config file
    key : str
      a key in the PATHS section of the config file
    as_str : bool [False]
      if True, returns path as a string (default is pathlib.Path object)

    Output
    -------
    path : pathlib.Path or None
      absolute path to the target in the config file.
      Prints warning if the path does not exist.
    """
    # reread the file whenever the function is called so you don't have to
    # reload the entire module if the config file gets updated
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file)
    sec = sec.upper()
    key = key.upper()
    try:
        path = Path(config[sec][key]).resolve()
    except KeyError:
        print("Error: bad keys for config file. Options are (section \\n\\t keys):")
        for sec in config.sections():
            print(sec)
            print("\t", ", ".join(i.upper() for i in config.options(sec)))
        return None
    # test that the path exists
    try:
        assert(path.exists())
    except AssertionError:
        print(f"Warning: {path} not found.")
    if as_str == True:
        path = path.as_posix()
    return path


# path to header tables
headers_path = load_config_path("user_paths", "headers_path")
# HST data files for manipulation
data_path = load_config_path("user_paths", "data_path")
# Database tables
table_path = load_config_path("user_paths", "table_path")
db_raw_file = load_config_path("tables", "db_raw_file")
db_file = load_config_path("tables", "db_file")
db_subcat_file = load_config_path("tables", "db_subcat_file")
db_clean_file =  load_config_path("tables", "db_clean_file")
# composite image
#composite_image_path = load_config_path("composite_img_file")
# correlation matrix
full_corr_mat_file = load_config_path("tables", "full_corr_path")

# gaia catalog and source matches
align_path = load_config_path("user_paths", "align_path")
# KS2 output files
ks2_path = load_config_path("user_paths", "ks2_path")


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
    data_path = load_config_path("user_paths", "data_path")
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


def read_instrument_config(cfg_name=None, key=None, dtype=None):
    """
    Pull the value for 'key' from the instrument configuration file.
    Run with no parameters for a list of files, or with only a file name
    for a list of keys.

    Parameters
    ----------
    cfg_name : [''] The name (not full path) of the configuration file.
      If empty, list the available files
    key : [None] The key whose value you want to pull.
      If empty, list the available keys
    dtype : [None] if provided, try to conver the output to this type.
      Otherwise, output will be a string

    Output
    ------
    the value stored at `key` in `cfg_name`

    """
    path = load_config_path("pw_paths", "instrument_path")
    try:
        file_path = path / cfg_name
        assert file_path.exists() and (cfg_name is not None)
    except (AssertionError, TypeError):
        files = path.glob('*')
        print("Available instrument configuration files:")
        for f in sorted(files):
            print(f"\t{f.name}")
        return

    # file exists, read it
    config = configparser.ConfigParser()
    config.read(file_path)
    try:
        value = config.get('Properties', key)
    except (KeyError, AttributeError, configparser.NoOptionError):
        print("Available keys:")
        for key in config['Properties']:
            print(f"\t{key}")
        return
    if dtype is not None:
        try:
            value = dtype(value)
        except:
            print(f"Error, cannot convert `{key}` value `{value}` to {str(dtype)}, returning string")
    return value
