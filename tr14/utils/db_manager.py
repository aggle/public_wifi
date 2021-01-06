"""
This module implements the DBManager class, which manages the storage and
manipulation of the data tables (stars, point sources, stamps, etc)
"""

import re
from pathlib import Path
import pandas as pd

from . import shared_utils
from . import table_utils

class DBManager:

    def __init__(self, db_path=shared_utils.db_clean_file):
        self.db_path = db_path
        self.tables = {} # list of loaded tables
        self.stars_tab = self.load_table('stars')
        self.ps_tab = self.load_table('point_sources')
        self.stamps_tab = self.load_table('stamps')
        self.lookup_dict = self.load_lookup_tables()

    def load_table(self, key):
        """
        Load a table.

        Parameters
        ----------
        key : string or list
        path : path to the database file that holds the table

        Output
        ------
        df : pd.DataFrame or dict of dataframes
          the table (or, if `key` is a list, then a dict of dataframes)
        """
        # if table is already loaded, print warning
        if key in self.tables:
            print(f"Warning: replacing existing table `{key}`")
            try:
                self.tables.pop(key)
            except:
                pass
        try:
            df = pd.read_hdf(self.db_path, key)
        except KeyError:
            print(f"Error: Key `{key}` not found in {str(db_file)}")
            df = None
        self.tables[key] = df
        return df

    def write_table(self, key, verbose=True, kwargs={}):
        """
        Write a table to the class DB file

        Parameters
        ----------
        key : str
          key for the table in the HDF file and in the class self.tables dict
        verbose : bool [True]
          print some output
        kwargs : dict [{}]
          any keyword arguments to pass to pd.DataFrame.to_hdf for DataFrames and
          Series, or to h5py.File for non-pandas objects

        Output
        -------
        Nothing; writes to file
        """
        # first, check if table exists
        try:
            assert(key in self.tables.keys())
        except AssertionError:
            print(f"Error: Table {key} not found, quitting.")
            return

        # option to write all the tables using recursion. will this work?????
        if key == 'all':
            for k in self.tables.keys():
                self.write_table(k)
                return

        # write a single table
        # this throws a performance warning when you store python objects with
        # mixed or complex types in an HDF file, but I want to ignore those
        with warnings.catch_warnings() as w:
            kwargs['mode'] = kwargs.get("mode", 'a')
            if hasattr(self.tables[key], 'to_hdf'):
                table.to_hdf(self.db_path, key=key, **kwargs)
                if verbose == True:
                    print(f"Table {key} written to {str(self.db_path)}")
            else:
                print("Error: cannot currently store non-pandas types")


    def list_available_tables(self, return_list=False):
        """
        Print a list of tables (and their descriptions?).
        Alternately, return a list of the tables.

        Parameters
        ----------
        return_list: bool [False]
          if True, instead of printing out the table names, return a list.

        Output
        ------
        table_names : list [optional]
          a list of available keys
        """
        with pd.HDFStore(self.db_path, mode='r') as store:
            table_names = sorted(store.keys())
            if return_list == False:
                print(f"Available tables in {self.db_path}:")
                print('\t'+'\n\t'.join(table_names))
                store.close()
        if return_list == True:
            return table_names


    def list_loaded_table(self):
        """
        Print the tables that have been loaded into the DB Manager
        """
        print("These tables have been loaded into the DB manager:")
        print("\t"+"\n\t".join(sorted(self.tables.keys())))


    def load_lookup_tables(self):
        """
        Load all the tables that start with *lookup_*

        Parameters
        ----------
        path : path to the database file

        Output
        ------
        lkp_dict : dictionary of lookup tables
        """
        with pd.HDFStore(self.db_path, mode='r') as store:
            lkp_dict = {key[1:]: store[key] for key in store.keys()
                        if key.startswith('/lookup')}
            return lkp_dict



    def find_lookup_table(self, lkp1, lkp2):
        """
        Find the lookup table that matches lkp1 to lkp2, using the table names
        lkp1 and lkp2 must be strings in the table name.
        For a list of choices, see self.tables.keys()

        Output:
        lkp_table : pd.DataFrame or None
          a lookup table

        """

        # now find the lookup table that has both these kinds
        for key in self.lookup_dict.keys():
            if (key.find(lkp1) >= 0) and (key.find(lkp2) >= 0):
                lkp_tab = self.lookup_dict[key]
                return lkp_tab
        print(f"Error: Lookup table linking {lkp1} and {lkp2} not found")
        return None


    def lookup_id(self, ids, want_this_id):
        """
        Given a set of ids (star IDs, stamp IDs, etc), find the matching IDs
        in the lookup table specified.

        Parameters
        ----------
        ids : string or list-like
          identifiers you want to look up
        want_this_id : str
          'star','stamp', 'ps', etc - the kind of identifier you want

        Output
        ------
        targ_ids : str or list-like
          the corresponding identifiers
        """
        id_dict = {'S': 'star', 'P': 'ps', 'T': 'stamp'}
        # which kind of identifier do you have?
        if isinstance(ids, str):
            id_lead = ids[0]
        else:
            # check that all are the same type
            id_lead = [i[0] for i in ids]
            try:
                assert len(set(id_lead)) == 1
            except AssertionError:
                print("Error: list of IDs has more than one kind:")
                print(set(id_lead))
                return ids
            id_lead = id_lead[0]

        start_id_type = id_dict[id_lead]

        # which kind do you want?
        if want_this_id == 'point_sources':
            want_this_id = 'ps'

        lkp_tab = self.find_lookup_table(start_id_type, want_this_id)
        try:
            assert isinstance(lkp_tab, pd.DataFrame)
        except AssertionError:
            print("No lookup table found, exiting...")
            return None
        matching_ids = lkp_tab.set_index(start_id_type+"_id").loc[ids, want_this_id+"_id"]
        return matching_ids

    def set_reference_quality_flag(self, stamp_ids, flag=True):
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
        if not isinstance(self.stamps_tab, pd.DataFrame):
            print("`None`` value for self.stamps_tab not yet enabled, quitting")
            return
        ind = self.stamps_tab.query("stamp_id in @stamp_ids").index
        self.stamps_tab.loc[ind, 'stamp_ref_flag'] = flag


