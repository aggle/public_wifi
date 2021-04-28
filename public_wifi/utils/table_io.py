"""
This module contains the API for reading, writing, and updating the data tables
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

from . import shared_utils, header_utils


# this dictionary helpfully maps the first letter of an identifier
# to its typical table name
ident_map = {'S': 'star_id', 'P': 'ps_id', 'T': 'stamp_id'}


def write_table(key, df, pk=None,
                db_file=shared_utils.db_clean_file,
                verbose=False,
                h5py_args={},
                clobber=False):
    """
    Write a table to file

    Parameters
    ----------
    key : str
      the table's key to use in the HDF file
    df : pd.DataFrame
      the dataframe with the table
    pk : str [None]
      (optional) the name of the table's primary key
    db_file : str or pathlib.Path
      the filename to write to
    h5py_args : dict [{}]
      any other keyword arguments to pass to h5py.File
    clobber : bool [False]
      overwrite a table if it exists

    Output
    ------
    Writes the table to the specified file, creating the file if necessary.

    """
    db_file = Path(db_file)
    try:
        assert db_file.parent.exists() == True
    except AssertionError:
        pass
    h5py_args.setdefault('mode', 'a')
    try:
        assert db_file.exists()
    except AssertionError:
        print(f"File {db_file} not found; creating.")
        #h5py_args['mode'] = 'w'
    with h5py.File(db_file, **h5py_args) as f:
        # test if key is already present
        if key in f:
            # key present -- check clobber
            if clobber == True:
                print(f"Key '{key}' in {db_file} already exists; overwriting...")
                f.pop(key)
            else: # if no clobbering, just return
                print(f"Error: key '{key}' in {db_file} already exists; doing nothing.")
                return
        # continue with creating the table
        try:
            g = f.create_group(key)
        except ValueError: # group already exists
            print(f"Error: key '{key}' in {db_file} already exists; doing nothing.")
            return
        # set the primary key
        if isinstance(pk, str):
            g.attrs['primary_key'] = pk
        # write each column of the dataframe
        for c in df.columns:
            col = df[c]
            orig_dtype = col.dtype
            col = np.stack(col.values) # trust numpy's automatic type conversion
            # Strings must be treated specially in HDF5
            if isinstance(col[0], str):
                col = col.astype(h5py.string_dtype('utf-8'))
            g.create_dataset(c, data=col, dtype=col.dtype, chunks=True)
        f.flush()
        f.close()
    if verbose == True:
        print(f"Wrote '{key}' to {db_file}")


def load_table(key, db_file=shared_utils.db_clean_file, verbose=False):
    """
    Load a table into a dataframe

    Parameters
    ----------
    key : str
      the key under which the table is stored
    db_file : str or pathlib.Path
      the file to read from

    Output
    ------

    """
    db_file=Path(db_file)
    try:
        assert(db_file.exists())
    except AssertionError:
        print(f"Error: {db_file} does not exists, returning None")
        return None
    with h5py.File(db_file, 'r') as f:
        g = f.get('/'+key, default=None)
        try:
            assert(g is not None)
        except AssertionError: # table not found
            print(f"Table '{key}' not found. Available tables are:")
            print('\n'.join(g for g in f))
            return g
        df = pd.DataFrame.from_dict({k: list(g[k][...]) for k in g})
        f.close()
    if verbose == True:
        print(f"Loaded '{key}' from {db_file}")
    return df


def update_table(key, pk_name, pk_val, column, val,
                 db_file=shared_utils.db_clean_file, verbose=True):
    """
    Update table value

    Parameters
    ----------
    key : str
      key under which the table is stored in the hdf5 file
    pk_name : str
      name of the table column with the primary key (serves as a proxy for the index)
    pk_val : str or list
      primary key value (can be list-like)
    column : str
      the column to update
    val : the new value (can be list-like; must be single-valued or same shape as pk_val)
    db_file : path to the table file

    Output
    ------
    None, writes updated values to file
    """
    if np.ndim(pk_val) == 0:
        pk_val = [pk_val]
    with h5py.File(db_file, 'r+') as f:
        pks = list(f[f'{key}/{pk_name}'][...]) # primary keys
        idx = [pks.index(i) for i in pk_val] # indices of keys to update
        f[f'{key}/{column}'][idx] = val
    f.close()
    if verbose:
        print(f"Updated '{key}' in {db_file}")






