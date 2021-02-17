"""
Tests for utils.subtr_utils
"""

import pytest
from pathlib import Path

import random
from tr14.utils import db_manager


class TestDBManager():
    """"Suite of tests for merging tables and related"""

    def test_join_all_tables(self, dbm_mast):
        """Join the star, ps, and stamp tables"""

        full_table = dbm_mast.join_all_tables()
        # the table should be as long as the longest table
        assert len(full_table) == max([len(dbm_mast.stars_tab),
                                       len(dbm_mast.ps_tab),
                                       len(dbm_mast.stamps_tab)])
        # the table should contain all the columns
        all_columns = list(dbm_mast.stars_tab.columns) +\
            list(dbm_mast.ps_tab.columns) +\
            list(dbm_mast.stamps_tab.columns)
        assert all([i in all_columns for i in full_table.columns])


    @pytest.mark.parametrize('nident,key,header,test_header',
                             [(1, 'naxis', 'sci', 'hdr_sci'), # 1 stamp, 1 key, sci
                              (2, 'naxis', 'sci', 'hdr_sci'), # 2 stamps, 1 key, sci
                              (2, '*', 'err', 'hdr_err'), # 2 stamps, 2 keys, err
                              ]
                             )
    def test_find_header_key(self, dbm_mast, nident, key, header, test_header):
        """
        Given a stamp or point source ID, find the file it came from and get a
        value from the header

        Parameters
        ----------
        dbm_mast : fixture
        nident : the number of stamp IDs to choose
        key : the header key(s) to find
        header : which header to look for
        test_header : the correct header name to test against
        """
        if nident == 1:
            # return string
            ident = random.choice(dbm_mast.stamps_tab['stamp_id'].values)
        else:
            # return list
            ident = random.choices(dbm_mast.stamps_tab['stamp_id'].values, k=nident)

        hdr_vals = dbm_mast.find_header_key(ident, key, header)

        # test that the number of rows is the number of unique identifiers
        if isinstance(ident, str):
            ident = [ident]
        ident = set(ident)
        assert hdr_vals.shape[0] == len(ident)

        # test that the number of columns is 3+nkeys
        hdr_df = dbm_mast.header_dict[test_header]
        if key == '*':
            key = list(hdr_df.columns)
        elif isinstance(key, str):
            key = [key]
        else:
            pass
        assert hdr_vals.shape[1] == len(key)+3


    
