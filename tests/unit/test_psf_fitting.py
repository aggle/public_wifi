import pytest
import numpy as np
from public_wifi import misc
from public_wifi import psf_fitting


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
    fs = psf_fitting.FitStar(random_star, epsf, cat_row_ind)
    assert(hasattr(fs, "epsf"))
    stamp = random_star.cat.loc[cat_row_ind, 'stamp']
    x0, y0 = misc.get_stamp_center(stamp)
    f0 = stamp.max()
    assert(isinstance(fs.log_probability([f0, x0, y0, 0.5]), float))
    
def test_FitStar_mcmc(all_stars):
    cat_row_ind = 0
    random_star = np.random.choice(all_stars)
    epsf = psf_fitting.construct_epsf(all_stars, cat_row_ind)
    fs = psf_fitting.FitStar(random_star, epsf, cat_row_ind)
    sampler = fs.run_mcmc()
    print(sampler)
    # print(sampler)

