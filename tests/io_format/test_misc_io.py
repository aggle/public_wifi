"""
These tests just check that the input and output of each method is what you
expect; they do not check that the algorithms are correct
Only methods that return something are tested here
"""
import pytest
import numpy as np
from public_wifi import misc
from astropy.modeling.models import Gaussian2D

DEBUG = False # use to turn on debug printing


@pytest.mark.parametrize("stamp", [
    11,
    np.ones((11, 11), dtype=float),
    misc.pd.Series({i: np.ones((11, 11)) for i in range(5)})
])
def test_get_stamp_center(stamp):
    center = misc.get_stamp_center(stamp)
    assert(isinstance(center, np.ndarray))
    assert(center.shape == (2,))

def test_compute_psf_center():
    gfunc = Gaussian2D(1, 0, 0, )
    g2d = Gaussian2D()
