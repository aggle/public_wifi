import pytest
from pathlib import Path
import numpy as np
from public_wifi import catalog_processing as catproc
from public_wifi import misc
from public_wifi import psf_fitting




### Fixtures ###

# default initialization args
star_id_column='target'
match_references_on=['filter']
stamp_size = 21
# default subtraction args
min_nref = 30
sim_thresh = 0.5
bad_references = ['J042705.86+261520.3']

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
def all_stars(catalog, data_folder):
    # all the stars, ready for PSF subtraction
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
def subtracted_stars(all_stars):
    catproc.catalog_subtraction(
        all_stars,
        sim_thresh=sim_thresh,
        min_nref=min_nref,
        verbose=False,
    )
    return all_stars


### Tests ###

def test_star2epsf(all_stars):
    cat_row_ind = 0
    random_star = np.random.choice(all_stars)
    epsf = psf_fitting.star2epsf(random_star, cat_row_ind)
    assert(isinstance(epsf, psf_fitting.pupsf.epsf_stars.EPSFStar))

def test_construct_epsf(all_stars):
    cat_row_ind = 0
    epsf = psf_fitting.construct_epsf(all_stars, cat_row_ind)
    assert(isinstance(epsf, psf_fitting.pupsf.image_models.ImagePSF))

def test_FitStar(all_stars):
    cat_row_ind = 0
    random_star = np.random.choice(all_stars)
    epsf = psf_fitting.construct_epsf(all_stars, cat_row_ind)
    fs = psf_fitting.FitStar(random_star.cat.loc[cat_row_ind, 'stamp'], epsf)
    assert(hasattr(fs, "epsf"))
    stamp = random_star.cat.loc[cat_row_ind, 'stamp']
    x0, y0 = misc.get_stamp_center(stamp)
    f0 = stamp.max()
    assert(isinstance(fs.log_probability([f0, x0, y0, 0.5]), float))
    
def test_FitStar_mcmc(all_stars):
    cat_row_ind = 0
    random_star = np.random.choice(all_stars)
    epsf = psf_fitting.construct_epsf(all_stars, cat_row_ind)
    fs = psf_fitting.FitStar(
        random_star.cat.loc[cat_row_ind, 'stamp'], epsf
    )
    sampler = fs.run_mcmc()
    print(sampler)
    # print(sampler)

@pytest.mark.parametrize('cat_index', [1])
def test_make_forward_modeled_psf(subtracted_stars, cat_index):
    star = np.random.choice(subtracted_stars)
    klip_basis = star.results.loc[cat_index, 'klip_basis']
    epsf = psf_fitting.construct_epsf(subtracted_stars, cat_index, 3)
    center = misc.get_stamp_center(stamp_size)

    fmstamp = psf_fitting.make_forward_modeled_psf(
        epsf, center[0], center[1], klip_basis,
    )
    fmshape = misc.get_stamp_shape(fmstamp)
    assert(fmshape == (stamp_size, stamp_size))
