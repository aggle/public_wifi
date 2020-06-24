"""
This module contains tests for tr14.utils.ks2_utils.py
"""
import pytest
from pathlib import Path

import random
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
    ks2_utils.get_master_catalog(raw=False),#mast_cat_nodup,
]
drop_cols = [
    ['NMAST', 'ps_tile_id', 'tile_id', 'exp_id', 'filt_id', 'unk', 'chip_id'],
    ['NMAST'],
]
@pytest.mark.parametrize('cat, drop_cols',
                         [(i, j) for i, j in zip(catalogs, drop_cols)],
)
@pytest.mark.clean
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


@pytest.mark.skip(reason="Skip this test because it takes a long time to run")
@pytest.mark.clean
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

@pytest.mark.clean
def test_clean_point_source_catalog(ps_cat_nodup):
    """
    Test that the cleaned catalog doesn't contain any point sources with cut values.
    Current cuts are:
    1. q > 0 (for all the q's)
    2. z > 0 (for all the z's)
    This starts from the 'nodup' catalog that has been cleaned of duplicate entries
    There should be a way to parameterize this
    """
    cut_cat = ks2_utils.clean_point_source_catalog(ps_cat_nodup)
    # test q cuts
    # get 'q' columns
    qstr = '^q[0-9]+'
    cols = [i for i in ps_cat_nodup.columns if ks2_utils.re.search(qstr, i) is not None]
    # all test results should be empty
    for col in cols:
        cut = f"{col} <= 0"#"@col <= 0"
        assert ps_cat_nodup.query(cut).empty == False

    # test z cuts
    # get 'z' columns
    zstr = '^z[0-9]+'
    cols = [i for i in ps_cat_nodup.columns if ks2_utils.re.search(zstr, i) is not None]
    # all test results should be empty
    for col in cols:
        cut = f"{col} <= 0"
        assert ps_cat_nodup.query(cut).empty == False


def test_get_exposure_neighbors(ps_cat_nodup):
    """
    get_exposure_neighbors retrieves all the point sources in a particular
    exposure that are within some distance from a specified point source,
    and returns a dataframe of those neighbors.
    How can I test this?

    """
    pass

@pytest.mark.parametrize("hdr", ['SCI','ERR','DQ','SAMP','TIME', 1, 2, 3, 4, 5])
def test_get_img_from_ks2_file_id(ps_cat_nodup, hdr):
    """
    get_img_from_ks2_file_id should return a 2-D array
    """
    exp_id = random.choice(ps_cat_nodup['exp_id'].values)
    img = ks2_utils.get_img_from_ks2_file_id(exp_id, hdr)
    assert ks2_utils.np.ndim(img) == 2

##########################
# CATALOG CLEANING TESTS #
##########################
def test_fix_catalog_dtypes(mast_cat_raw):
    """
    Make sure all the entries in the catalog have the right dtype
    """
    cat = ks2_utils.fix_catalog_dtypes(mast_cat_raw, ks2_utils.master_dtypes)
    cat_dtypes = cat.dtypes
    for col in cat_dtypes.index:
        assert(cat_dtypes[col] == ks2_utils.master_dtypes[col])

def test_clean_ks2_input_catalogs(mast_cat_raw, ps_cat_raw):
    """
    Test that after the cuts are applied, no sources remain in violation of the cuts
    """
    # 
