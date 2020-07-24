"""
Includes shared useful stuff, like path definitions and file formats
"""

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
      absolute path to the target in the config file. Returns None if path does not exist.
    """
    key = key.upper()
    path = Path(config['PATHS'][key]).resolve()
    # test that the path exists
    try:
        assert(path.exists())
    except AssertionError:
        print(f"Error: {path} not found.")
        return None
    return path

# HST data files for manipulation
data_path = load_config_path("data_path")
# Database tables
table_path = load_config_path("table_path")
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
