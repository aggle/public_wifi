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
    return ks2_utils.get_point_source_catalog(clean=False)

@pytest.fixture()
def mast_cat_raw():
    """
    The original KS2 master catalog
    """
    return ks2_utils.get_master_catalog(clean=False)

@pytest.fixture()
def ps_cat_clean():
    """
    The cleaned KS2 point source catalog
    """
    return ks2_utils.get_point_source_catalog(clean=True)

@pytest.fixture()
def mast_cat_clean():
    """Return the cleaned master catalog"""
    return ks2_utils.get_master_catalog(clean=True)

#####################
# Duplicate entries #
#####################

catalogs = [ # For some reason, the fixtures can't be used directly here
    ks2_utils.get_point_source_catalog(clean=True),#ps_cat_clean,
    ks2_utils.get_master_catalog(clean=True),#mast_cat_clean,
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
    It's only necessary to run if you have to regenerate the `clean.csv` files
    """
    ps_cat_clean, mast_cat_clean = ks2_utils.remove_duplicates(ps_cat_raw,
                                                               mast_cat_raw)
    # test point source catalog
    cols_test = list(ps_cat_clean.columns)
    for i in ['NMAST', 'ps_tile_id', 'tile_id', 'exp_id',
              'filt_id', 'unk', 'chip_id']:
        cols_test.pop(cols_test.index(i))
    ps_dups = ps_cat_clean[ps_cat_clean.duplicated(subset=cols_test, keep=False)]
    assert ps_dups.empty == True

    # test master catalog
    cols_test = list(mast_cat_clean.columns)
    for i in ['NMAST']:
        cols_test.pop(cols_test.index(i))
    mast_dups = mast_cat_clean[mast_cat_clean.duplicated(subset=cols_test,
                                                         keep=False)]
    assert mast_dups.size == 0


def test_get_exposure_neighbors(ps_cat_clean):
    """
    get_exposure_neighbors retrieves all the point sources in a particular
    exposure that are within some distance from a specified point source,
    and returns a dataframe of those neighbors.
    How can I test this?

    """
    pass

@pytest.mark.parametrize("hdr", ['SCI','ERR','DQ','SAMP','TIME', 1, 2, 3, 4, 5])
def test_get_img_from_ks2_file_id(ps_cat_clean, hdr):
    """
    get_img_from_ks2_file_id should return a 2-D array
    """
    exp_id = random.choice(ps_cat_clean['exp_id'].values)
    img = ks2_utils.get_img_from_ks2_file_id(exp_id, hdr)
    assert ks2_utils.np.ndim(img) == 2

##########################
# CATALOG CLEANING TESTS #
##########################
@pytest.mark.clean
def test_clean_catalog_dtypes_mast(mast_cat_raw):
    """
    Make sure all the entries in the catalog have the right dtype
    """
    cat = ks2_utils.clean_catalog_dtypes(mast_cat_raw, ks2_utils.master_dtypes)
    cat_dtypes = cat.dtypes
    for col in cat_dtypes.index:
        assert(cat_dtypes[col] == ks2_utils.master_dtypes[col])

@pytest.mark.clean
def test_clean_catalog_dtypes_ps(ps_cat_raw):
    """
    Make sure all the entries in the catalog have the right dtype
    """
    cat = ks2_utils.clean_catalog_dtypes(ps_cat_raw, ks2_utils.nimfo_dtypes)
    cat_dtypes = cat.dtypes
    for col in cat_dtypes.index:
        print(col, cat_dtypes[col], ks2_utils.nimfo_dtypes[col])
        assert(cat_dtypes[col] == ks2_utils.nimfo_dtypes[col])


@pytest.mark.clean
@pytest.mark.parametrize('ndet_min', [0, 5, 10, 15, 50])
def test_catalog_cut_ndet(ps_cat_raw, ndet_min):
    """Cut on the number of detections in the point source catalog"""

    cut_cat = ks2_utils.catalog_cut_ndet(ps_cat_raw, ndet_min=ndet_min)
    ndet = cut_cat.groupby("NMAST").size()
    assert all(ndet >= ndet_min)


@pytest.mark.clean
def test_clean_point_source_catalog(ps_cat_clean):
    """
    Test that the cleaned catalog doesn't contain any point sources with cut values.
    Current cuts are:
    1. q > 0 (for all the q's)
    2. z > 0 (for all the z's)
    This starts from the 'clean' catalog that has been cleaned of duplicate entries
    There should be a way to parameterize this.
    The idea is that if you apply the *opposite* criterion, you should be left with an empty catalog in the end.
    """
    cut_cat = ks2_utils.clean_point_source_catalog(ps_cat_clean,
                                                   cut_ndet_args={'ndet_min': 5})
    # test q cuts
    # get 'q' columns
    qstr = '^q[1-9][0-9]*'
    cols = [i for i in ps_cat_clean.columns
            if ks2_utils.re.search(qstr, i) is not None]
    # all test results should be empty
    for col in cols:
        cut = f"{col} <= 0"#"@col <= 0"
        assert ps_cat_clean.query(cut).empty == True

    # test z cuts
    # get 'z' columns
    zstr = '^z[1-9][0-9]*'
    cols = [i for i in ps_cat_clean.columns
            if ks2_utils.re.search(zstr, i) is not None]
    # all test results should be empty
    for col in cols:
        cut = f"{col} <= 0"
        print(col)
        assert ps_cat_clean.query(cut).empty == True
    # 


@pytest.mark.clean
def test_recompute_master_catalog(mast_cat_clean, ps_cat_clean):
    pass

@pytest.mark.clean
def test_clean_master_catalog(mast_cat_raw, ps_cat_clean):
    """
    Check that the cleaned master catalog is consistent with the point source catalog.
    Test: NMAST values, q range, z range

    Parameters
    ----------
    mast_cat_raw : pd.DataFrame
      raw master catalog
    ps_cat_clean : pd.DataFrame
      cleaned point source catalog

    """
    # given the master catalog
    # when you apply the cleaning function
    # then you should only have the same objects as in the point source catalog
    mast_cat_clean = ks2_utils.clean_master_catalog(mast_cat_raw, ps_cat_clean)

    # assert that all the stars in mast_cat_clean are also in ps_cat_clean
    mast_names = set(mast_cat_clean['NMAST'])
    ps_names = set(ps_cat_clean['NMAST'].unique())
    assert mast_names.issubset(ps_names)

    """
    # assert that the q-ranges are compatible
    ps_cat_qrange = (ps_cat[['q1','q2','q3']].min().min(),
                     ps_cat[['q1','q2','q3']].max().max())
    mast_cat_qrange = (mast_cat[['q1','q2']].min().min(),
                       mast_cat[['q1','q2']].max().max())
    # ps min should be lower or equal to mast min
    assert ps_cat_qrange[0] <= mast_cat_qrange[0]
    # ps max should be lower or equal to mast max
    assert ps_cat_qrange[1] >= mast_cat_qrange[1]

    # assert that the z-ranges are compatible
    ps_cat_zrange = (ps_cat[['z1','z2','z3']].min().min(),
                     ps_cat[['z1','z2','z3']].max().max())
    mast_cat_zrange = (mast_cat[['zmast1','zmast2']].min().min(),
                       mast_cat[['zmast1','zmast2']].max().max())
    # ps min should be lower or equal to mast min
    assert ps_cat_zrange[0] <= mast_cat_zrange[0]
    # ps max should be lower or equal to mast max
    assert ps_cat_zrange[1] >= mast_cat_zrange[1]
    """

@pytest.mark.clean
def test_clean_ks2_input_catalogs(mast_cat_raw, ps_cat_raw):
    """
    Test that after the cuts are applied, no sources remain in violation of the cuts
    """
    pass
