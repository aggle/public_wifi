"""
This module contains the API for interacting with data tables
"""

from pathlib import Path
import pandas as pd
import csv

from . import shared_utils


table_list_path = shared_utils.table_path / "list_of_tables.csv"
table_definitions = shared_utils.table_path / "table_definitions.ini"


# TABLE CSV FORMAT
csv_args  = {'sep': ',',
             'index': False,
             'header': True,
}
def write_table(table, name, descr):
    """
    Write a table to file in ../data/tables

    Parameters
    ----------
    table : pd.DataFrame
      a pandas Dataframe containing the data
    name : str
      name for the table. File name will be {0}
    descr : str
      short description of the contents, prended as the header

    Returns
    -------
    Nothing; writes to file
    """.format(shared_tils.table_path / ('_name_.csv'))
    fname = shared_tils.table_path / (name + '.csv')
    # prepend comments that contain the file name and description
    with open(fname, 'w') as ff:
        # this clobbers the file if it already exists
        ff.write(f"# {name}\n")
        ff.write(f"# {descr}\n")
    table.to_csv(fname.as_posix(), mode='a', **csv_args) 
    """
    # update the file that keeps track of the list of tables
    table_list = pd.read_csv(table_list_path)
    if table_list.query(f"name == {name}"):
        table_list.query
    """

def write_table_hdf(table, name, descr):
    """
    Write a table to an HDF file in ../data/tables/.
    Keys are TABLE, NAME, and DESCR

    Parameters
    ----------
    table : pd.DataFrame
      pandas DataFrame (or Series) containing the data you want to store
    name : str
      name for the table. File name will be {0}
    descr : str
      short description of the contents

    Returns
    -------
    Nothing; writes to file
    """.format(shared_tils.table_path / (name + '.hdf5'))

    fname = shared_tils.table_path / (name + 'hdf5')
    # Step 1: Write the table part
    table.to_hdf(fname.as_posix(), key='TABLE', mode='w')
    # Step 2: Add the NAME and DESCR keys
    with open(fname, 'w') as ff:
        # this clobbers the file if it already exists
        ff.write(f"# {name}\n")
        ff.write(f"# {descr}\n")



def load_table(name):
    """
    Load a table
    """
    pass


def list_available_tables():
    """
    Print a list of tables and their descriptions
    """
    pass


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


def get_value(table, query_column, value, return_column=None):
    """
    Retrieve a value from a table. Assumes the equality operator ('==')
    Remember to cast any field names into lower case
    Parameters
    ----------
    table : pd.DataFrame
      the table to query
    query_column: string
      the name of the column to query
    value: type(query_column)
      the value to match in the query column
    return_column : string (optional, default is None)
      the column whose value you want to return. If left as default (None),
      returns the full (queried) dataframe

    Returns
    -------
    return_value : a series (or dataframe) of all the values that match the query

    """
    # all columns are in lowercase
    query_column = query_column.lower()
    return_value = table.query(f"{query_column} == {value}")
    if return_column is not None:
        return_column = return_column.lower()
        return_value = return_value[return_column]
    return return_value



def set_value():
    """
    Set table value
    """
    pass


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

