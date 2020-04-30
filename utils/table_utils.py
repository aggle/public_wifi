"""
This module contains the API for interacting with data tables
"""

from pathlib import Path
import pandas as pd

table_path = Path(__file__).parent / "../../data/tables"
table_list_path = table_path / "list_of_tables.csv"

def write_table(table, name, descr):
    """
    Write a table to file in ../data/tables

    Parameters
    ----------
    table : None
      a pandas Dataframe
    name : None
      name for the table. File name will be {0}
    descr : None
      short description of the contents

    Returns
    -------
    Nothing; writes to file
    """.format(table_path / (name + '.csv'))
    fname = table_path / (name + '.csv')
    # prepend comments that contain the file name and description
    with open(fname, 'w') as f:
        # this clobbers the file if it already exists
        f.write(f"# {name}\n")
        f.write(f"# {descr}\n")
    table.to_csv(f, mode='a') 
    # update the file that keeps track of the list of tables
    """
    table_list = pd.read_csv(table_list_path)
    if table_list.query(f"name == {name}"):
        table_list.query
    """

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


def get_value(table, query_column, value, return_column):
    """
    Retrieve a value from a table. Assumes the equality operator ('==')
    Remember to cast any field names into lower case
    Parameters
    ----------
    table : None
      the table to query
    query_column: None
      the name of the column to query
    value: None
      the value to match in the query column
    return_column : None
      the column whose value you want to return

    Returns
    -------
    return_value : a series of all the values that match the query

    """
    # all columns are in lowercase
    query_column = query_column.lower()
    return_column = return_column.lower()
    return_value = table.query(f"{query_column} == {value}")[return_column]
    return return_value



def set_value():
    """
    Set table value
    """
    pass
