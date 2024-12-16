"""
This test uses data from HST-17167 for testing
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# from public_wifi import db_manager
from public_wifi import starclass as sc

@pytest.fixture()
def catalog_file():
    catalog_file = Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc.csv")
    return catalog_file

@pytest.fixture()
def data_folder():
    return Path("/Users/jaguilar/Projects/Research/hst17167-ffp/data/HST/")

@pytest.fixture()
def catalog(catalog_file):
    catalog = sc.load_catalog(catalog_file)
    return catalog

@pytest.fixture()
def random_cat_rows(catalog):
    """Get a group of rows at random to initialize a Star object"""
    star_id = np.random.choice(catalog['target'])
    rows = catalog.query(f"target == '{star_id}'")
    return dict(star_id=rows)


@pytest.fixture()
def high_snr_catalog(catalog):
    high_snr_stars = catalog.groupby("target")['snr'].sum().sort_values(ascending=False)[:10]
    high_snr_rows = catalog.query(f"target in {list(high_snr_stars.index)}").copy()
    return high_snr_rows

@pytest.fixture()
def star(catalog, data_folder):
    star_id = np.random.choice(catalog['target'].unique())
    star = sc.Star(star_id, catalog.query(f"target == '{star_id}'"), data_folder=data_folder)
    return star

@pytest.fixture()
def all_stars(catalog, data_folder):
    # all the stars, ready for PSF subtraction
    stars = catalog.groupby("target").apply(
        lambda group: sc.Star(group.name, group, data_folder=data_folder),
        include_groups=False
    )
    return stars

@pytest.fixture()
def processed_stars(high_snr_catalog, data_folder):
    stars = sc.process_stars(high_snr_catalog, 'target', ['filter'], data_folder, 11)
    return stars

@pytest.fixture()
def random_processed_star(processed_stars):
    """Get a star with subtraction and detection results attached"""
    return sc.np.random.choice(processed_stars)
