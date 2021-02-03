"""
Tests for utils.subtr_utils
"""

import pytest
from pathlib import Path

import random
from tr14.utils import db_manager

# OK, I don't know how to test classes. 

def test_dbm_init(dbm_mast):
    """Test that all the attributes get initialized"""
    list_of_attrs = ['db_path', 'tables', 'stars_tab', 'ps_tab', 'stamps_tab',
                     'grid_defs_tab', 'comp_status_tab', 'lookup_dict',
                     'header_dict', 'subtr_groupby_keys']
    
