"""
This module implements the DBManager class, which manages the storage and
manipulation of the data tables (stars, point sources, stamps, etc)
"""

import re
from pathlib import Path
import pandas as pd

from . import shared_utils
from . import table_utils

def copy_attrs(obj1, obj2):
    """
    WARNING WARNING WARNING: COPIED ATTRIBUTES SEEM TO KEEP TRACK OF WHAT INSTANCE THEY CAME FROM
    Do not use this method until this issue is resolved.
    Copy attributes from object 1 to object 2

    Parameters
    ----------
    obj1 : object
      copy attributes *from* here
    obj2 : object
      copy attributes *to* here

    Output
    ------
    modifies obj2 in-place
    """
    #for name, attr in obj1.__dict__.items():
    #    try:
    #        setattr(obj2, name, attr).copy()
    #    except AttributeError:
    #        setattr(obj2, name, attr)
    #    shared_utils.debug_print(f"{name} set", False)
    for name in dir(obj1):
        if name.startswith('__'):
            # don't move internal attributes
            # be careful if you set these!
            continue
        try:
            obj1_attr = getattr(obj1, name).copy()
        except AttributeError:
            obj1_attr = getattr(obj1, name)
        finally:
            setattr(obj2, name, obj1_attr)
            shared_utils.debug_print(f"{name} set", True)


class DBManager:

    def __init__(self, db_path=shared_utils.db_clean_file):
        self.db_path = db_path
        # load the principle tables directly as class members
        self.tables = {} # list of loaded tables
        self.stars_tab = self.load_table('stars')
        self.ps_tab = self.load_table('point_sources')
        self.stamps_tab = self.load_table('stamps')
        self.grid_defs_tab = self.load_table('grid_sector_definitions')
        self.comp_status_tab = self.load_table('companion_status')
        # dicts of tables that group naturally together
        self.lookup_dict = self.load_lookup_tables()
        self.header_dict = self.load_header_tables()
        # finally, group the subtraction subsets together!
        self._do_subtr_groupby()

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


    def load_header_tables(self):
        """
        Load all the tables that start with *hdr_*

        Parameters
        ----------
        path : path to the database file

        Output
        ------
        hdr_dict : dictionary of header tables
        """
        with pd.HDFStore(self.db_path, mode='r') as store:
            hdr_dict = {key[1:]: store[key] for key in store.keys()
                        if key.startswith('/hdr')}
            return hdr_dict


    def find_lookup_table(self, lkp1, lkp2):
        """
        Find the lookup table that matches lkp1 to lkp2, using the table names
        lkp1 and lkp2 must be strings in the table names.

        Output
        ------
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


    def find_matching_id(self, ids, want_this_id):
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


    def find_sector(self, ps_id=None, sector_id=None):
        """
        Given one or several point source identifiers, find their sectors.
        Or, given one or several sectors, find their point sources
        One of ps_id and sector_id must be None, and the other must not be

        Parameters
        ----------
        ps_id : str or list [None]
          string or list of point source ids (format: P000000)
        sector_id : int [None]
          int or list of ints for the sector IDs

        Output
        ------
        sector_df: pd.DataFrame, columns are 'ps_id' and 'sector_id'
          the sector number(s) for each given point sources
        """
        # first, make sure that one of the inputs is None and the other isn't
        try:
            assert( ((ps_id is None) and (sector_id is None)) == False )
        except AssertionError:
            print("Error: both ps_id and sector_id cannot be None")
            return None
        # next, make sure that only *one* is None
        try:
            assert( ((ps_id is None) or (sector_id is None)) == True )
        except AssertionError:
            print("Error: one of ps_id and sector_id must be None")
            return None

        if ps_id is not None:
            if isinstance(ps_id, str):
                query_str = "ps_id == @ps_id"
            else:
                query_str = "ps_id in @ps_id"
        elif sector_id is not None:
            if isinstance(sector_id, int):
                query_str = "sector_id == @sector_id"
            else:
                query_str = "sector_id in @sector_id"
        else:
            print("Failed to construct query, returning None")
            return None

        sector_df = self.lookup_dict['lookup_point_source_sectors'].query(query_str)
        return sector_df


    def query_table(self, table, query_str, **kwargs):
        """
        Interface for processing table queries, so you don't access the tables directly.
        So far, not working because cannot pass variables to the query string

        Parameters
        ----------
        table : pd.DataFrame
          the table (an object member) to query
        query_str : str
          the query string to pass to table.query
        kwargs : any arguments that need to be passed to the query

        Output
        ------
        query_results : pd.DataFrame
          results of the query
        """
        #for kw in kwargs:
        #    print(kw)
        # make all the variables in kwargs into local variables
        #for k, val in kwargs.items():
        #    exec(key+"=val")
        return table.query(query_str)


    def _cut_lookup_tables_to_local(self):
        """
        Cut down the star, point source, and stamp lookup tables to only
        the IDs present in the current instance of DBManager.
        This is basically to prevent stamps/point sources that show up in
        multiple sectors from being selected when you just search by star_id

        Parameters
        ----------
        None, uses self.stars_tab, self.ps_tab, and self.stamps_tab

        Output
        ------
        None, operates on self.lookup_dicts entries

        """
        lkp_table_names = [i for i in self.lookup_dict.keys() if i.endswith('_id')]
        pairs = [i.split('_')[1].split('-') for i in lkp_table_names]
        cut_tables = {}
        for lkp_name in lkp_table_names:
            pair = lkp_name.split('_')[1].split('-')
            # construct the table name
            tab_names = [i+'_tab' if i == 'ps' else i+'s_tab' for i in pair]
            # construct the identifier name
            id_names = [i+'_id' for i in pair]
            query_str = ' and '.join([f'{i} in @self.{t}.{i}' for t, i in zip(tab_names, id_names)])
            cut_table = self.lookup_dict[lkp_name].query(query_str).copy()
            shared_utils.debug_print(cut_table.shape, False)
            cut_tables[lkp_name] = cut_table
            #self.lookup_dict[lkp_name].update(cut_table)
        self.lookup_dict.update(cut_tables)


    def group_by_filter_epoch(self):
        """
        Group the stars, stamps, and point sources by filter and epoch.
        Not sure what to do with these. Return them as separate objects?
        Assign them as object attributes?

        Parameters
        ----------
        self

        Output
        ------
        assigns to self.fe_dict, which is a dict whose keys are the filter
        and epoch keys, and each value is a dict that stores the star, ps, and
        stamp tables for that filter and epoch combo
        dict keys are the concatenated group keys
        """
        # group just the point source IDs
        fe_groups = self.ps_tab.groupby(["ps_filt_id", "ps_epoch_id"])['ps_id']
        self.fe_dict = {''.join(k): {} for k in fe_groups.groups.keys()}
        for k, dk in zip(sorted(fe_groups.groups.keys()),
                     [''.join(k) for k in sorted(fe_groups.groups.keys())]):
            k_ps = fe_groups.get_group(k)
            self.fe_dict[dk]['ps'] = self.ps_tab.query('ps_id in @k_ps')
            k_stamps = self.find_matching_id(k_ps, 'stamp')
            self.fe_dict[dk]['stamps'] = self.stamps_tab.query('stamp_id in @k_stamps')
            k_stars = self.find_matching_id(k_ps, 'star')
            self.fe_dict[dk]['stars'] = self.stars_tab.query('star_id in @k_stars')


    def _do_subtr_groupby(self):
        """
        Group the database into self-contained units for PSF subtraction.
        These quantities are as follows:
        filter, epoch, sector

        Parameters
        ----------
        None

        Output
        ------
        sets self.subtr_groups : pd.groupby object
          each group in this grouopby has the star, point source, and stamp IDs
          that go together to make a database subset for PSF subtraction.
          use like subtr_groups.get_group(key) and pass the results to DBSubset
        """
        ps_tab_sector = pd.merge(self.ps_tab,
                                 self.lookup_dict['lookup_point_source_sectors'],
                                 on='ps_id')
        ps_gb = ps_tab_sector.groupby(['ps_filt_id', 'ps_epoch_id', 'sector_id'])['ps_id']
        # now get the corresponding star and stamp groups
        subtr_groups = pd.merge(ps_gb.apply(lambda x: self.find_matching_id(x, 'star').reset_index()),
                                ps_gb.apply(lambda x: self.find_matching_id(x, 'stamp').reset_index()),
                                on='ps_id', left_index=True)
        self.subtr_groups = subtr_groups.groupby(subtr_groups.index.names[:3])

    def create_subtr_subset_db(self, key):
        """
        Generate a database subset for subtraction according to the groups in self.subtr_groups

        Parameters
        ----------
        key : tuple
          key for the self.subtr_groups groupby object

        Output
        ------
        DBSubset object with the subset

        """
        group = self.subtr_groups.get_group(key)
        return DBSubset(group['star_id'],
                        group['ps_id'],
                        group['stamp_id'],
                        db_master=None, db_path=self.db_path)

class DBSector(DBManager):
    """
    This class holds a subset of star, point source, and stamp tables for a WFC3 detector sector.
    Basically it initializes a regular DBManager object and then filters down to the requested sector
    """
    def __init__(self, sector_id, db_master=None, db_path=shared_utils.db_clean_file):
        """
        Give it the stars, ps, and stamps tables. All other tables are copied
        """
        #if db_master is not None:
        #    # copy the attributes over
        #    copy_attrs(db_master, self)
        #else:
        # run the regular initialization
        DBManager.__init__(self, db_path=db_path)
        self.select_sector(sector_id)
        self.group_by_filter_epoch()
        self._cut_lookup_tables_to_local()

    def select_sector(self, sector_id):
        """
        Given a sector ID, return only the point sources, stamps, and stars in that sector

        Output:
        None; sets self.star_tab, self.ps_tab, and self.stamps_tab
        """
        # point sources
        sec_ps_ids = self.find_sector(sector_id = sector_id)['ps_id']
        idx = self.ps_tab.query('ps_id not in @sec_ps_ids').index
        self.ps_tab.drop(idx, axis=0, inplace=True)
        # now stamps
        stamp_ids = self.find_matching_id(self.ps_tab['ps_id'].values, 'stamp')
        idx = self.stamps_tab.query("stamp_id not in @stamp_ids").index
        self.stamps_tab.drop(idx, axis=0, inplace=True)
        # now stars
        star_ids = self.find_matching_id(self.ps_tab['ps_id'].values, 'star')
        idx = self.stars_tab.query("star_id not in @star_ids").index
        self.stars_tab.drop(idx, axis=0, inplace=True)




class DBSubset(DBManager):
    """
    This class holds a subset of star, point source, and stamp tables.
    All other tables contain their full versions.
    It is more generic than DB Sector.
    It accepts as arguments a list of star IDs, point source IDs, and stamp IDs.
    if db_master is given (i.e. not None), uses the passed DBManager instance
    to initialize instead of initializing from scratch
    """
    def __init__(self,
                 star_ids, ps_ids, stamp_ids,
                 db_master=None,
                 db_path=shared_utils.db_clean_file):
        """
        Give it the stars, ps, and stamps tables. All other tables are copied
        It is more generic than DBSector
        """
        #if db_master is not None:
        #    # copy the attributes over
        #    copy_attrs(db_master, self)
        #else:
        # run the regular initialization
        DBManager.__init__(self, db_path=db_path)

        self.get_db_subset(star_ids, ps_ids, stamp_ids)
        self._cut_lookup_tables_to_local()

    def get_db_subset(self, star_ids, ps_ids, stamp_ids):
        """
        query the tables down to the reduced set and update the table list

        Parameters
        ----------
        star_ids : list-like of star IDs
        ps_ids : list-like of point source IDs
        stamp_ids : list-like of stamp IDs

        Output
        ------
        sets self.stars_tab, self.ps_tab, and self.stamps_tab,
        and updates self.tables

        """
        self.stars_tab = self.stars_tab.query('star_id in @star_ids')
        self.tables['stars'] = self.stars_tab
        self.ps_tab = self.ps_tab.query('ps_id in @ps_ids')
        self.tables['point_sources'] = self.ps_tab
        self.stamps_tab = self.stamps_tab.query('stamp_id in @stamp_ids')
        self.tables['stamps'] = self.stamps_tab

