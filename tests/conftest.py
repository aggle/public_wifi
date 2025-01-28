"""
This test uses data from HST-17167 for testing
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# from public_wifi import db_manager
from public_wifi import starclass as sc
from public_wifi import catalog_processing as catproc


# default initialization args
star_id_column='target'
match_references_on=['filter']
stamp_size = 15
# default subtraction args
min_nref = 20
sim_thresh = 0.9
# default detection args
snr_thresh = 5.
n_modes = 3
mf_width = 11

@pytest.fixture(scope='session')
def catalog_file():
    catalog_file = Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc.csv")
    return catalog_file

@pytest.fixture(scope='session')
def data_folder():
    return Path("/Users/jaguilar/Projects/Research/hst17167-ffp/data/HST/")

@pytest.fixture(scope='session')
def catalog(catalog_file):
    catalog = catproc.load_catalog(catalog_file)[:20]
    return catalog

@pytest.fixture(scope='session')
def random_cat_rows(catalog):
    """Get a group of rows at random to initialize a Star object"""
    star_id = np.random.choice(catalog['target'])
    rows = catalog.query(f"target == '{star_id}'")
    return dict(star_id=rows)

@pytest.fixture(scope='session')
def high_snr_catalog(catalog):
    high_snr_stars = catalog.groupby("target")['snr'].sum().sort_values(ascending=False)[:20]
    high_snr_rows = catalog.query(f"target in {list(high_snr_stars.index)}").copy()
    return high_snr_rows


@pytest.fixture(scope='session')
def star(catalog, data_folder):
    star_id = np.random.choice(catalog['target'].unique())
    star = sc.Star(star_id, catalog.query(f"target == '{star_id}'"), data_folder=data_folder)
    return star

@pytest.fixture(scope='session')
def all_stars(catalog, data_folder):
    # all the stars, ready for PSF subtraction
    # stars = catalog.groupby("target").apply(
    #     lambda group: sc.Star(group.name, group, data_folder=data_folder),
    #     include_groups=False
    # )
    stars = catproc.catalog_initialization(
        catalog,
        star_id_column=star_id_column,
        match_references_on=match_references_on,
        data_folder=data_folder,
        stamp_size=stamp_size,
        bad_references=[],
    )
    return stars

@pytest.fixture(scope='session')
def stars_with_references(all_stars):
    all_stars.apply(lambda star: star.set_references(
        compute_similarity=True,
        other_stars = all_stars[all_stars.index != star.star_id]
    ))
    return all_stars

@pytest.fixture(scope='session')
def subtracted_stars(catalog, data_folder):
    # perform PSF subtraction
    stars = catproc.catalog_initialization(
        catalog,
        star_id_column=star_id_column,
        match_references_on=match_references_on,
        data_folder=data_folder,
        stamp_size=stamp_size,
        bad_references=[],
    )
    catproc.catalog_subtraction(
        stars,
        # psf subtraction args
        min_nref = min_nref,
        sim_thresh = sim_thresh,
    )
    return stars

@pytest.fixture(scope='session')
def processed_stars(catalog, data_folder):
    stars = catproc.process_catalog(
        input_catalog=catalog,
        star_id_column=star_id_column,
        match_references_on=match_references_on,
        data_folder=data_folder,
        stamp_size=stamp_size,
        bad_references = [],
        scale_stamps = False,
        # psf subtraction args
        min_nref = min_nref,
        sim_thresh = sim_thresh,
        # detection args
        snr_thresh = snr_thresh,
        n_modes = n_modes,
    )
    return stars

# individual stars
@pytest.fixture(scope='session')
def random_subtracted_star(subtracted_stars):
    """Get a star with subtraction and detection results attached"""
    return np.random.choice(subtracted_stars)

@pytest.fixture(scope='session')
def nonrandom_subtracted_star(subtracted_stars):
    """Get a star with subtraction and detection results attached"""
    # return subtracted_stars.iloc[2]
    return subtracted_stars['J042930.88+264433.3']

@pytest.fixture(scope='session')
def random_processed_star(processed_stars):
    """Get a star with subtraction and detection results attached"""
    return np.random.choice(processed_stars)

@pytest.fixture(scope='session')
def nonrandom_processed_star(processed_stars):
    """Get a star with subtraction and detection results attached"""
    # return processed_stars.iloc[2]
    return processed_stars['J042930.88+264433.3']
@pytest.fixture(scope='session')
def star_with_candidates(processed_stars):
    return processed_stars.loc["J041947.39+281534.6"]
