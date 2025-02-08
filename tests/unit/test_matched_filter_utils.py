import pytest
import numpy as np
from matplotlib import pyplot as plt

from public_wifi import misc
from public_wifi import matched_filter_utils as mfutils


def test_compute_pca_bias_fixed_pos(random_processed_star):
    """
    Test computing the PCA bias at a given position instead of convolving the MF
    with the whole Kklip basis array.
    The bias should be the same at that pixel whether you convolve it or not
    """
    star = random_processed_star
    print("Using star ", star.star_id)
    center = misc.get_stamp_center(star.stamp_size)
    pos = (-3, -1)
    pos = misc.center_to_ll_coords(star.stamp_size, pos)[::-1]
    cat_row = 1
    kklip = 5
    mf = star.results['mf'].loc[cat_row, kklip]
    modes = star.results['klip_basis'].loc[cat_row, :kklip]
    pca_bias_full = mfutils.compute_pca_bias(mf, modes, True, None)
    pca_bias_pos = mfutils.compute_pca_bias(mf, modes, True, pos)
    # print(pca_bias_full.apply(lambda img: img[*pos]))
    # print(pca_bias_pos)
    assert(((pca_bias_full.apply(lambda img: img[*pos]) - pca_bias_pos).abs() < 1e-15).all())
    return
