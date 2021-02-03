import pytest
from tr14.utils import shared_utils
from tr14.utils import db_manager
from tr14.utils import subtr_utils


#############################
# DBManager and Subtraction #
#############################
@pytest.fixture()
def dbm_mast():
    """Database Manager with the master catalog"""
    dbm = db_manager.DBManager(db_manager.shared_utils.db_clean_file)

    # testing happens here
    yield

    del dbm

@pytest.fixture()
def dbm_sec(dbm_mast):
    """Database Manager for Epoch 1, Filter 2, Sector 15"""
    filter_id = 'F2'
    epoch_id = 'D1'
    sector_id = 15
    key = (filter_id, epoch_id, sector_id)
    dbm_sec = dbm_mast.create_subtr_subset_db(key)

    yield

    del dbm_sec

@pytest.fixture()
def subtr_mgr(dbm_sec):
    """Subtraction Manager for the test sector"""
    subtr_mgr = subtr_utils.SubtrManager(dbm_fe, calc_corr_flag=False)

    yield

    del subtr_mgr
