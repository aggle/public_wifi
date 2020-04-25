"""
This file lists the tables available in code/databases/ and has helper functions for loading them
"""

from pathlib import Path
from glob import glob
import pandas as pd


# table path - probably should set to read from config
table_path = Path("./databases")

# master list of tables stored here. format is `"filename": "description"`
list_of_tables = {}


def list_tables():
    """
    Print the tables and their descriptions, sorted alphabetically

    Parameters
    ----------
      None

    Returns
    -------
      Nothing

    """
    tables = pd.Series(list_of_tables)
    # sort index alphabetically
    for ind in tables.index:
        print(f"{ind}: {tables[ind]}")


def load_table(table_name):
    """
    Return a pandas dataframe of the requested filename

    Parameters
    -----------
      table_name :  str
        the filename of the desired table, excluding the extension ('.csv') and path

    Returns
    -------
      table : pandas.DataFrame
        the desired table, as a pandas dataframe

    """
    pass
