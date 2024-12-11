"""
This test uses data from HST-17167 for testing
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# from public_wifi import db_manager
from public_wifi import starclass

@pytest.fixture()
def catalog_file():
    catalog_file = Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc.csv")
    return catalog_file

@pytest.fixture()
def data_folder():
    return Path("/Users/jaguilar/Projects/Research/hst17167-ffp/data/HST/")

@pytest.fixture()
def catalog(catalog_file):
    dtypes = {
        'target': str,
        'file': str,
        'filter': str,
        'ra': float,
        'dec': float,
        'x': float,
        'y': float,
        'mag_aper': float,
        'e_mag_aper': float,
        'dist': float,
        'snr': float,
    }
    catalog = pd.read_csv(str(catalog_file), dtype=dtypes)
    catalog['x'] = catalog['x'] - 1
    catalog['y'] = catalog['y'] - 1
    return catalog

@pytest.fixture()
def star(catalog, data_folder):
    star_id = np.random.choice(catalog['target'].unique())
    star = starclass.Star(star_id, catalog.query(f"target == '{star_id}'"), data_folder=data_folder)
    return star

@pytest.fixture()
def all_stars(catalog, data_folder):
    # all the stars, ready for PSF subtraction
    stars = catalog.groupby("target").apply(
        lambda group: starclass.Star(group.name, group, data_folder=data_folder),
        include_groups=False
    )
    return stars


