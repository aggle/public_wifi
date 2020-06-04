"""
This module contains tests for tr14.utils.ks2_utils.py
"""
import pytest
from pathlib import Path

import tr14.utils.ks2_utils as ks2_utils

############
# Fixtures #
############
@pytest.fixture()
def ps_cat_raw():
    """
    The original KS2 point source catalog
    """
    return ks2_utils.get_point_source_catalog(raw=True)

@pytest.fixture()
def mast_cat_raw():
    """
    The original KS2 master catalog
    """
    return ks2_utils.get_master_catalog(raw=True)

@pytest.fixture()
def ps_cat_nodup():
    """
    The duplicate-cleaned KS2 point source catalog
    """
    return ks2_utils.get_point_source_catalog(raw=False)

@pytest.fixture()
def mast_cat_nodup():
    """Return the duplicate-cleaned master catalog"""
    return ks2_utils.get_master_catalog(raw=False)

#####################
# Duplicate entries #
#####################

catalogs = [ # For some reason, the fixtures can't be used directly here
    ks2_utils.get_point_source_catalog(raw=False),#ps_cat_nodup,
    ks2_utils.get_point_source_catalog(raw=False),#mast_cat_nodup,
]
drop_cols = [
    ['NMAST', 'ps_tile_id', 'tile_id', 'exp_id', 'filt_id', 'unk', 'chip_id'],
    ['NMAST'],
]
@pytest.mark.parametrize('cat, drop_cols',
                         [(i, j) for i, j in zip(catalogs, drop_cols)],
)
def test_remove_duplicates(cat, drop_cols):
    """
    make sure all duplicate stars are removed from the dataframes
    Parameters
    ----------
    cat : pd.DataFrame
      a catalog (either point sources, or the master catalog)
    drop_cols : list
      a list of columns to drop 
    """
    cols_test = list(cat.columns)
    # drop all the columns that shouldn't be used for comparison
    for i in drop_cols:
        cols_test.pop(cols_test.index(i))
    # get duplicates, if any
    dups = cat[cat.duplicated(subset=cols_test, keep=False)]
    assert dups.empty == True


@pytest.mark.skip("Skip this test because it takes a long time to run")
def test_remove_duplicates_raw(ps_cat_raw, mast_cat_raw):
    """
    Test that ks2_utils.remove_duplicates() works as billed
    It's only necessary to run if you have to regenerate the `nodup.csv` files
    """
    ps_cat_nodup, mast_cat_nodup = ks2_utils.remove_duplicates(ps_cat_raw,
                                                               mast_cat_raw)
    # test point source catalog
    cols_test = list(ps_cat_nodup.columns)
    for i in ['NMAST', 'ps_tile_id', 'tile_id', 'exp_id',
              'filt_id', 'unk', 'chip_id']:
        cols_test.pop(cols_test.index(i))
    ps_dups = ps_cat_nodup[ps_cat_nodup.duplicated(subset=cols_test, keep=False)]
    assert ps_dups.empty == True

    # test master catalog
    cols_test = list(mast_cat_nodup.columns)
    for i in ['NMAST']:
        cols_test.pop(cols_test.index(i))
    mast_dups = mast_cat_nodup[mast_cat_nodup.duplicated(subset=cols_test,
                                                         keep=False)]
    assert mast_dups.size == 0

