

"""
Includes shared useful stuff, like path definitions
"""

from pathlib import Path
import configparser

# the paths are stored in the config file
config_file = (Path(__file__).parent.absolute() / "../config.ini").resolve()
config = configparser.ConfigParser()
config.read(config_file)

"""
This block defines some useful paths
"""
# HST data files for manipulation
data_path = (Path(config['PATHS']['DATA_PATH']).resolve())
# Database tables
table_path = (Path(config['PATHS']['TABLE_PATH']).resolve())
# Gaia catalog and source matches
align_path = (Path(config['PATHS']['ALIGN_PATH']).resolve())
# KS2 output files
ks2_path = (Path(config['PATHS']['KS2_PATH']).resolve())


"""
Here's a wrapper function for loading paths from the config file
t turns relative into absolute
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
