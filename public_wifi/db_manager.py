"""
This module implements the DBManager class, which manages the storage and
manipulation of the data tables (stars, point sources, stamps, etc)
"""

import re
from pathlib import Path
import h5py
import pandas as pd
import warnings

from astropy.wcs import WCS
from astropy.io import fits

from .utils import shared_utils
from .utils import table_utils



class DBManager:

    def __init__(self, db_path=shared_utils.db_clean_file):
        """
        Usage:
        Initialize using the path to the database

        Useful attributes
        -----------------
        self.subtr_groupby_keys : keys in self.ps_tab used to group objects for psf subtraction
        """
        self.db_path = db_path
        # load the principle tables directly as class members
        self.tables = {} # list of loaded tables
        # dicts of tables that group naturally together
        self.lookup_dict = self.load_lookup_tables()
        self.header_dict = self.load_header_tables()
        # data tables
        self.stars_tab = self.load_table('stars', verbose=True)
        self.ps_tab = self.load_table('point_sources', verbose=True)
        self.stamps_tab = self.load_table('stamps', verbose=True)
        self.grid_defs_tab = self.load_table('grid_sector_definitions', verbose=True)
        self.comp_status_tab = self.load_table('companion_status', verbose=True)
        # finally, group the subtraction subsets together with the default keys
        self.subtr_groupby_keys = ['ps_filt_id', 'ps_epoch_id', 'sector_id']
        self.do_subtr_groupby(keys=self.subtr_groupby_keys)

    @property
    def subtr_groupby_keys(self):
        return self._subtr_groupby_keys
    @subtr_groupby_keys.setter
    def subtr_groupby_keys(self, new_keys):
        self._subtr_groupby_keys = new_keys
        # and re-do the groupby automatically


    def load_table(self, key, verbose=False):
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
            df = table_utils.load_table(key, self.db_path, verbose=verbose)
        except KeyError:
            print(f"Error: Key `{key}` not found in {str(self.db_path)}")
            df = None
        self.tables[key] = df
        return df


    def write_table(self, key, table=None, verbose=False):
        """
        Write a table to the class DB file

        Parameters
        ----------
        key : str
          key for the table in the HDF file and in the class self.tables dict
        table : pd.DataFrame [None]
          the table to write. if None, use the key to look it up in self.tables
        verbose : bool [True]
          print some output

        Output
        -------
        Nothing; writes to file
        """
        # option to write all the tables using recursion. will this work?????
        if key == 'all':
            for k, table in self.tables.items():
                self.write_table(k, table=table, verbose=verbose)
            return

        # write a single table
        # first, check if table exists
        try:
            assert(key in self.tables.keys())
        except AssertionError:
            print(f"Error: Table {key} not found, quitting.")
            return
        if table is None:
            table = self.tables[key]
        # this throws a performance warning when you store python objects with
        # mixed or complex types in an HDF file, but I want to ignore those
        with warnings.catch_warnings() as w:
            kwargs['mode'] = kwargs.get("mode", 'a')
            # if hasattr(self.tables[key], 'to_hdf'):
            #     table.to_hdf(self.db_path, key=key, **kwargs)
            #     if verbose == True:
            #         print(f"Table {key} written to {str(self.db_path)}")
            # else:
            #     print("Error: cannot currently store non-pandas types")
            table_utils.write_table(key, table, db_file=self.db_path, verbose=verbose)


    def update_table(self, key, pk_name, pk_val, column, val,
                     verbose=False):
        """
        Update a table entry to disk

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
        table_file : path to the table file

        Output
        ------
        None, writes updated values to file
        """
        table_utils.update_table(key, pk_name, pk_val, column, val,
                                 db_file=self.db_path, verbose=verbose)


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
        with h5py.File(self.db_path, mode='r') as f:
            table_names = sorted(f.keys())
            if return_list == False:
                print(f"Available tables in {self.db_path}:")
                print('\t'+'\n\t'.join(table_names))
            f.close()
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
        # with pd.HDFStore(self.db_path, mode='r') as store:
        #with h5py.File(self.db_path, mode='r') as f:
        #     lkp_dict = {key: f[key] for key in f.keys()
        #                 if key.startswith('lookup')}
        # return lkp_dict
        # get list of keys
        with h5py.File(self.db_path, mode='r') as f:
            keys = [key for key in f.keys() if key.startswith('lookup')]
        # load keys into dict
        lkp_dict = {key: self.load_table(key) for key in keys}
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
        #with h5py.File(self.db_path, mode='r') as f: # pd.HDFStore(self.db_path, mode='r') as store:
        #    hdr_dict = {key: f[key] for key in f.keys()
        #                if key.startswith('hdr')}
        #return hdr_dict
        with h5py.File(self.db_path, mode='r') as f:
            keys = [key for key in f.keys() if key.startswith('hdr')]
            # load keys into dict
        hdr_dict = {key: self.load_table(key) for key in keys}
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
          The kind of identifier you wish to return.
          'S' : star, 'P': point source, 'T': stamp
          obsolete: 'star','stamp', 'ps', etc - the kind of identifier you want

        Output
        ------
        targ_ids : str or list-like
          the corresponding identifiers
        """
        want_this_id = want_this_id.upper()

        id_dict = {'S': 'star', 'P': 'ps', 'T': 'stamp'}
        di_dict = {v: k for k, v in id_dict.items()} # reversed

        try:
            assert want_this_id in id_dict.keys()
        except AssertionError:
            print(f"Error: Please provide one of {list(id_dict.keys())}")
            print(f"You provided: {want_this_id}.")
            return ids

        # which kind of identifier do you have?
        if isinstance(ids, str):
            id_lead = ids[0]
            ids = [ids]
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

        # if you accidentally ask for the same kind of ID that you provide,
        # return the ids
        if id_lead.upper() == want_this_id.upper():
            #print("find_matching_id: same type of ID requested as provided.")
            return pd.Series(ids)


        start_id_type = id_dict[id_lead]

        # which kind do you want?
        requested_id = id_dict[want_this_id]

        lkp_tab = self.find_lookup_table(start_id_type, requested_id)
        try:
            assert isinstance(lkp_tab, pd.DataFrame)
        except AssertionError:
            print("No lookup table found, exiting...")
            return None
        try:
            # honestly this implementation could be much better
            # use .query instead of .set_index to handle missing identifiers
            matching_ids = lkp_tab.set_index(start_id_type+"_id").loc[ids, requested_id+"_id"]
        except KeyError:
            print("One or more of the provided identifiers were not found.")
            print("This shouldn't occur unless the database was constructed improperly.")
            return pd.Series(ids)
        return matching_ids.squeeze()

    def join_all_tables(self):
        """
        Merge the stamp, star, and point source tables

        Parameters
        ----------
        None

        Output
        ------
        full_table : pd.DataFrame
          the full merged table!

        """
        stars_ps = self.find_matching_id(self.stars_tab['star_id'], 'P').reset_index()
        ps_stamps = self.find_matching_id(self.ps_tab['ps_id'], 'T').reset_index()
        # merge the IDs
        full_table = self.stars_tab.merge(stars_ps, on='star_id') # add ps_id
        full_table = full_table.merge(ps_stamps, on='ps_id') # add stamp_id
        # merge the tables
        full_table = full_table.merge(self.ps_tab, on='ps_id').merge(self.stamps_tab, on='stamp_id')
        shared_utils.debug_print(False, self.stars_tab.shape, full_table.shape)
        return full_table

    def find_stamp_mag(self, stamp_id):
       """
       Given a stamp id or list of stamps, find their corresponding magnitudes
       from the point source table.

       Parameters
       ----------
       stamp_id : str
         stamp ID of the form T000000

       Output
       ------
       mags : float or series
         if one stamp_id is passed, then one float
         if multiple stamp_ids are passed, then series indexed by the stamp_id

       """
       if isinstance(stamp_id, str):
           stamp_id = pd.Series(stamp_id)
       ps_ids = self.find_matching_id(stamp_id, 'P').reset_index()
       df = pd.merge(left=ps_ids, right=self.ps_tab, on='ps_id')
       mags = df.set_index('stamp_id')['ps_mag']
       if mags.size == 1:
           mags = mags.values[0]
       return mags


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

    def find_header_key(self, ident, key, header='SCI'):
        """
        Given a point source or stamp identifier, find the exposure it came
        from and pull out the requested header keywords

        Parameters
        ----------
        ident : str or list of str
          either a point source or stamp identifier
        key : str
          a valid header keyword. If "*", return all values in the header
        header : str [SCI]
          which header to read from?

        Output
        ------
        header_values : pd.DataFrame
          returns a dataframe of the requested keywords along with the associated identifiers
          and exposure IDs

        """
        # make sure it's a valid header
        try:
            hdr_df = self.header_dict['hdr_'+header.lower()]
        except KeyError:
            print(f"Error: Header {header} not found.")
            print(f"Choices are: {' '.join(i.split('_')[1].upper() for i in dbm.header_dict.keys())}")
            raise KeyError

        # make sure it's a valid key
        if key == '*':
            key = list(hdr_df.columns)
        if isinstance(key, str):
            key = [key]
        try:
            # use set logic
            assert set(key).issubset(hdr_df.columns)
        except AssertionError:
            print(f"Error: {key} is not a valid key for header {header}.")
            raise AssertionError

        # ident needs to be a list too
        if isinstance(ident, str):
            ident = [ident]
        # remove duplicate identifiers
        ident = list(set(ident))
        ps_id = self.find_matching_id(ident, 'p')
        exp_id = self.ps_tab.set_index('ps_id').loc[ps_id, 'ps_exp_id']
        file_name = self.lookup_dict['lookup_files'].set_index('file_id').loc[exp_id, 'file_name']
        file_name = file_name.apply(lambda x: x.split('_flt')[0])
        hdr_rows = hdr_df.query("rootname in @file_name").drop_duplicates()
        # return the header, along with the identifiers
        hdr_rows = hdr_rows.merge(file_name.reset_index(), left_on='rootname', right_on='file_name')
        hdr_rows = hdr_rows.merge(exp_id.reset_index(), left_on='file_id', right_on='ps_exp_id')
        hdr_rows = hdr_rows.merge(ps_id.reset_index())
        return hdr_rows[['ps_exp_id'] + list(ps_id.reset_index().columns) + key]

    def get_exposure_from_id(self, obj_id, hdr='SCI'):
        """
        Pull out an image from the fits file, given the exposure identifier.
        if more than one obj_id is given, returns a list of images

        Parameters
        ----------
        obj_id : str or list
          the object identifier whose exposure you want
        hdr : str or int ['SCI']
          which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']

        Returns:
        img : numpy.array or pandas series containing 2-D image from the fits file
        """
        if isinstance(obj_id, str):
            obj_id = [obj_id]
        # get the point source IDs - make list to force pd.Series return object
        ps_ids = self.find_matching_id(obj_id, 'P')
        # get the exposures
        exp_ids = self.ps_tab.set_index('ps_id').loc[ps_ids, 'ps_exp_id']
        # images
        imgs = exp_ids.apply(lambda x: table_utils.get_img_from_exp_id(x, hdr))
        imgs.index = obj_id
        imgs.index.name = 'exposure'
        return imgs.squeeze()

    def get_header_from_id(self, obj_id, hdr='SCI'):
        """
        Pull out a header from a fits file, given the target object identifier.
        if more than one obj_id is given, returns a list of headers

        Parameters
        ----------
        obj_id : str or list
          the object identifier whose exposure you want
        hdr : str or int ['SCI']
          which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']

        Returns:
        hdr : fits.header or pandas series containing headers
        """
        if isinstance(obj_id, str):
            obj_id = [obj_id]
        # get the point source IDs - make list to force pd.Series return object
        ps_ids = self.find_matching_id(obj_id, 'P')
        # get the exposures, ensure exp_ids is a pandas-like object
        exp_ids = self.ps_tab.query("ps_id in @ps_ids").set_index('ps_id')['ps_exp_id']
        # images
        hdrs = exp_ids.apply(lambda x: table_utils.get_hdr_from_exp_id(x, hdr))
        hdrs.index = obj_id
        hdrs.index.name = 'exposure'
        return hdrs.squeeze()
    
    def get_wcs_from_id(self, obj_id):
        """
        Given an identifier (star, point source, stamp), get the corresponding WCS headers.
        Class wrapper for table_utils.get_wcs_from_exp_id
        WARNING: this gives a WCS centered at the exposure, not the stamp

        Parameters
        ----------
        obj_id : star, point source, or stamp id (or IDs)

        Output
        ------
        series with obj_id as the index and corresponding wcs as the values

        """
        if isinstance(obj_id, str):
            obj_id = [obj_id]
        # get the point source IDs
        ps_ids = self.find_matching_id(obj_id, 'P')
        # ensure exp_ids is a pandas-like object
        exp_ids = self.ps_tab.query("ps_id in @ps_ids").set_index('ps_id')['ps_exp_id']
        wcs = exp_ids.apply(table_utils.get_wcs_from_exp_id)
        wcs.name = 'wcs'
        wcs.index = obj_id
        #wcs.index.name = "wcs"
        return wcs#.squeeze()

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
            shared_utils.debug_print(False, cut_table.shape)
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
            k_stamps = self.find_matching_id(k_ps, 'T')
            self.fe_dict[dk]['stamps'] = self.stamps_tab.query('stamp_id in @k_stamps')
            k_stars = self.find_matching_id(k_ps, 'S')
            self.fe_dict[dk]['stars'] = self.stars_tab.query('star_id in @k_stars')


    def do_subtr_groupby(self, keys=None):
        """
        Group the database into self-contained units for PSF subtraction.
        These quantities are as follows:
        filter, epoch, sector

        Parameters
        ----------
        keys : str or list [self.subtr_groupby_keys]
          key or list of keys in the point source table to use for grouping the
          psf subtraction targets

        Output
        ------
        sets self.subtr_groups : pd.groupby object
          each group in this grouopby has the star, point source, and stamp IDs
          that go together to make a database subset for PSF subtraction.
          use like subtr_groups.get_group(key) and pass the results to DBSubset
        """
        if keys == None:
            keys = self.subtr_groupby_keys
        ps_tab_sector = pd.merge(self.ps_tab,
                                 self.lookup_dict['lookup_point_source_sectors'],
                                 on='ps_id')
        ps_gb = ps_tab_sector.groupby(keys)['ps_id']
        # now get the corresponding star and stamp groups
        subtr_groups = pd.merge(ps_gb.apply(lambda x: self.find_matching_id(x, 'S').reset_index()),
                                ps_gb.apply(lambda x: self.find_matching_id(x, 'T').reset_index()),
                                on='ps_id', left_index=True)
        self.subtr_groups = subtr_groups.groupby(keys)
        self.subtr_groupby_keys = keys

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
        key_dict = dict(zip(self.subtr_groupby_keys, key))
        dbm_sub = DBSubset(group['star_id'],
                           group['ps_id'],
                           group['stamp_id'],
                           db_master=None, db_path=self.db_path)
        dbm_sub.keys = key_dict
        return dbm_sub


class DBSubset(DBManager):
    """
    This class holds a subset of star, point source, and stamp tables.
    All other tables contain their full versions.
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
        super().__init__(db_path=db_path)
        self.get_db_subset(star_ids, ps_ids, stamp_ids)
        self._cut_lookup_tables_to_local()
        # if the subset was made by filtering the ps table with a groupby, give keys here
        self.keys = None

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
        super().__init__(db_path=db_path)
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
        stamp_ids = self.find_matching_id(self.ps_tab['ps_id'].values, 'T')
        idx = self.stamps_tab.query("stamp_id not in @stamp_ids").index
        self.stamps_tab.drop(idx, axis=0, inplace=True)
        # now stars
        star_ids = self.find_matching_id(self.ps_tab['ps_id'].values, 'S')
        idx = self.stars_tab.query("star_id not in @star_ids").index
        self.stars_tab.drop(idx, axis=0, inplace=True)
