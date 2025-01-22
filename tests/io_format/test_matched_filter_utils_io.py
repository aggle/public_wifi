"""
These tests just check that the input and output of each method is what you
expect; they do not check that the algorithms are correct
Only methods that return something are tested here
"""
import pytest
import numpy as np
from public_wifi import matched_filter_utils as mf_utils

DEBUG = False # use to turn on debug printing


def test_make_normalized_psf(star):
    stamp = star.cat.loc[0, 'stamp']
    psf = mf_utils.make_normalized_psf(stamp)
    assert(isinstance(psf, np.ndarray))
    assert(psf.ndim == 2)

def test_make_matched_filter(star):
    stamp = star.cat.loc[0, 'stamp']
    psf = mf_utils.make_matched_filter(stamp)
    assert(isinstance(psf, np.ndarray))
    assert(psf.ndim == 2)

def test_apply_matched_filter():
    pass

def test_compute_throughput(random_processed_star):
    star = random_processed_star
    row = star.results.iloc[0]
    kklip = row['klip_basis'].index[-1]
    psf = row['klip_model'].loc[kklip]
    klip_basis = row['klip_basis'].loc[:kklip]
    mf = mf_utils.make_matched_filter(psf, 7)
    # KLIP correction not included
    thpt = mf_utils.compute_throughput(mf, None)
    assert(isinstance(thpt, float))
    thpt = mf_utils.compute_throughput(mf, klip_basis)
    assert(isinstance(thpt, mf_utils.pd.Series))
    assert(len(thpt) == len(klip_basis))

def test_compute_mf_norm(random_processed_star):
    star = random_processed_star
    row = star.results.iloc[0]
    kklip = row['klip_basis'].index[-1]
    psf = row['klip_model'].loc[kklip]
    klip_basis = row['klip_basis'].loc[:kklip]
    mf = mf_utils.make_matched_filter(psf, 7)
    norm = mf_utils.compute_mf_norm(mf)
    assert(isinstance(norm, float))

def test_compute_pca_bias(random_processed_star):
    star = random_processed_star
    row = star.results.iloc[0]
    kklip = row['klip_basis'].index[-4]
    psf = row['klip_model'].loc[kklip]
    klip_basis = row['klip_basis'].loc[:kklip]
    mf = mf_utils.make_matched_filter(psf, 7)
    bias = mf_utils.compute_pca_bias(mf, klip_basis)
    assert(len(bias) == len(klip_basis))
    assert(isinstance(bias, mf_utils.pd.Series))
    assert(isinstance(bias.iloc[-1], np.ndarray))

def test_make_gaussian_psf():
    g2d = mf_utils.make_gaussian_psf(stamp_size=13, filt='F850LP')
    assert(isinstance(g2d, np.ndarray))
