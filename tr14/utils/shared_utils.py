"""
Includes shared useful stuff, like path definitions
"""

from pathlib import Path

"""
This block defines some useful paths
"""
# HST data files for manipulation
data_path = (Path(__file__).parent.absolute() / "../../data/my_data/").resolve()
# Database tables
table_path = (Path(__file__).parent.absolute() / "../../data/tables/").resolve()
# Gaia catalog and source matches
align_path = (Path(__file__).parent.absolute() / "../../data/align_catalog/").resolve()
# KS2 output files
ks2_path = (Path(__file__).parent.absolute() / "../../data/ks2/").resolve()

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
