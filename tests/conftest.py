import pytest
from tr14.utils import shared_utils
from tr14.utils import db_manager
from tr14.utils import subtr_utils


#############################
# DBManager and Subtraction #
#############################
# this is probably not the best way of doing things
#_dbm_mast = db_manager.DBManager(db_manager.shared_utils.db_clean_file)
_subset_key = ('F2', # filter ID
              'D1', # epoch ID
              15    # sector ID
              )
#_dbm_sec = _dbm_mast.create_subtr_subset_db(_subset_key)
#_subtr_mgr = subtr_utils.SubtrManager(_dbm_sec, calc_corr_flag=False)

@pytest.fixture()
def dbm_mast():
    """Database Manager with the master catalog"""
    return _dbm_mast

@pytest.fixture()
def dbm_sec():
    """Database Manager for Epoch 1, Filter 2, Sector 15"""
    _dbm_mast = db_manager.DBManager(db_manager.shared_utils.db_clean_file)
    _dbm_sec = _dbm_mast.create_subtr_subset_db(_subset_key)
    return _dbm_sec

@pytest.fixture()
def subtr_mgr():
    """Subtraction Manager for the test sector"""
    _dbm_mast = db_manager.DBManager(db_manager.shared_utils.db_clean_file)
    _dbm_sec = _dbm_mast.create_subtr_subset_db(_subset_key)
    _subtr_mgr = subtr_utils.SubtrManager(_dbm_sec, calc_corr_flag=False)
    return _subtr_mgr
