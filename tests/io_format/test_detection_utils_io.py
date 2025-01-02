"""
These tests just check that the input and output of each method is what you
expect; they do not check that the algorithms are correct
Only methods that return something are tested here
"""
import pytest
import numpy as np
from public_wifi import detection_utils as dutils

DEBUG = False # use to turn on debug printing


def test_make_normalized_psf(star):
    stamp = star.cat.loc[0, 'stamp']
    psf = dutils.make_normalized_psf(stamp)
    assert(isinstance(psf, np.ndarray))
    assert(psf.ndim == 2)

def test_make_matched_filter(star):
    stamp = star.cat.loc[0, 'stamp']
    psf = dutils.make_matched_filter(stamp)
    assert(isinstance(psf, np.ndarray))
    assert(psf.ndim == 2)

def test_compute_throughput(random_processed_star):
    star = random_processed_star
    row = star.results.iloc[0]
    kklip = row['klip_basis'].index[-1]
    psf = row['klip_model'].loc[kklip]
    klip_basis = row['klip_basis'].loc[:kklip]
    mf = dutils.make_matched_filter(psf, 7)
    # KLIP correction not included
    thpt = dutils.compute_throughput(mf, None)
    assert(isinstance(thpt, float))
    thpt = dutils.compute_throughput(mf, klip_basis)
    assert(isinstance(thpt, dutils.pd.Series))
    assert(len(thpt) == len(klip_basis))

