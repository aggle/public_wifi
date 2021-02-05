"""
Tests for utils.subtr_utils
"""

import pytest
from pathlib import Path

import random
from tr14.utils import db_manager

# OK, I don't know how to test classes. 

class TestMerge():
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
