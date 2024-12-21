import pytest
import numpy as np
from public_wifi import catalog_processing as catproc
# from tests.conftest import data_folder

DEBUG = True # use to turn on debug printing

def test_load_catalog(catalog, catalog_file):
    assert(isinstance(catalog, catproc.pd.DataFrame))
    # make sure you have the required columns
    required_columns = ['target', 'file', 'x', 'y']
    assert(all([c in catalog.columns for c in required_columns]))
    # make sure you subtract off the 1 from the original coordinates
    default_catalog = catproc.pd.read_csv(str(catalog_file), dtype=str)
    # make sure the entries match
    default_catalog = default_catalog.query(f"target in {list(catalog['target'].unique())}")
    default_xy = default_catalog[['x','y']].astype(float)
    # print(default_xy.iloc[0].values, catalog[['x','y']].iloc[0].values)
    thresh = 1e-10
    coord_diff = (np.array(default_xy) - np.abs(catalog[['x','y']]) - 1)
    assert(all([all(i < thresh) for i in coord_diff.values]))


def test_catalog_initialization(catalog, data_folder):
    bad_reference = 'J042705.86+261520.3'
    stars = catproc.catalog_initialization(
        catalog,
        star_id_column='target',
        match_references_on=['filter'],
        data_folder=data_folder,
        stamp_size=15,
        bad_references=bad_reference,
        min_nrefs = 10,
    )
    unique_stars = catalog['target'].unique()
    assert(len(stars) == len(unique_stars))
    assert(all(stars.apply(lambda el: isinstance(el, catproc.sc.Star))))
    # check that the attributes are there
    attrs = ["star_id", "stamp_size", "is_good_reference", "data_folder",
             "cat", "has_candidates", "match_by"]
    for attr in attrs:
        if DEBUG:
            print(attr)
        assert(all(stars.apply(hasattr, args=[attr])))

def test_catalog_subtraction(catalog, data_folder):
    bad_reference = 'J042705.86+261520.3'
    stars = catproc.catalog_initialization(
        catalog,
        star_id_column='target',
        match_references_on=['filter'],
        data_folder=data_folder,
        stamp_size=15,
        bad_references=bad_reference,
        min_nrefs = 10,
    )

    sim_thresh = -1
    min_nref = 1
    catproc.catalog_subtraction(
        stars,
        sim_thresh=sim_thresh,
        min_nref=min_nref
    )
    assert(all(hasattr(star, 'results') for star in stars))
    # check that the bad reference is flagged correctly
    assert(all(
        stars.apply(
            lambda star: bad_reference not in star.references.reset_index()['target']
        )
    ))
    # check that the subtraction columns are there
    subtr_cols = ['klip_sub', 'klip_basis', 'klip_model']
    for star in stars:
        assert(all([ c in star.results.columns for c in subtr_cols]))

def test_catalog_detection(catalog, data_folder):
    bad_reference = 'J042705.86+261520.3'
    stars = catproc.catalog_initialization(
        catalog,
        star_id_column='target',
        match_references_on=['filter'],
        data_folder=data_folder,
        stamp_size=15,
        bad_references=bad_reference,
        min_nrefs = 10,
    )
    sim_thresh = -1
    min_nref = 1
    catproc.catalog_subtraction(
        stars,
        sim_thresh=sim_thresh,
        min_nref=min_nref
    )
    snr_thresh = 3
    n_modes = 2
    catproc.catalog_detection(stars, snr_thresh, n_modes)
    # check that the columns were added
    det_cols = ['detmap', 'snrmap', 'snr_candidates']
    for star in stars:
        assert(all([ c in star.results.columns for c in det_cols]))

def test_process_catalog(catalog, data_folder):
    stars = catproc.process_catalog(
        input_catalog=catalog,
        star_id_column='target',
        match_references_on=['filter'],
        data_folder=data_folder,
        stamp_size=15,
        min_nref=1,
        sim_thresh=-1.0,
    )
    assert(isinstance(stars, catproc.pd.Series))
    # check that one star is made for each unique star in the catalog
    star_ids = set(stars.apply(lambda s: s.star_id))
    cat_stars = set(catalog['target'].unique())
    assert(star_ids == cat_stars)
    # assert(len(stars) == len(catalog['target'].unique()))
    # check that attributes have been assigned
    assert(all(stars.apply(lambda s: hasattr(s, 'cat'))))

