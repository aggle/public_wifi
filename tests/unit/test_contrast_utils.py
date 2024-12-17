import pytest
import numpy as np
from matplotlib import pyplot as plt
from public_wifi import contrast_utils as cutils
from public_wifi import detection_utils as dutils


def test_inject_psf(random_processed_star):
    star = random_processed_star
    print("Testing injections on ", star.star_id)
    stamp = star.cat.loc[0, 'stamp']
    # psf = star.results.loc[0, 'klip_model'][1][2:-2, 2:-2].copy()
    psf = star.results.loc[0, 'klip_model'][1][2:-2, 2:-2].copy()
    stamp_center = np.floor(np.array(stamp.shape)/2).astype(int)
    psf_center = np.floor(np.array(psf.shape)/2).astype(int)

    pos = (2, -3)
    pos_index = stamp_center + pos[::-1]

    injected_stamp = cutils.inject_psf(stamp, psf, pos)
    injected_flux = injected_stamp[*pos_index]
    test_flux = stamp[*pos_index] + psf[*psf_center]
    # fig, ax = plt.subplots(nrows=1, ncols=1,)
    # ax.imshow(injected_stamp, origin='lower')
    # plt.show()
    assert(cutils.np.abs(injected_flux - test_flux) < 1e-6)

def test_scale_psf(random_processed_star):
    star = random_processed_star
    print("Testing injections on ", star.star_id)
    # stamp = star.cat.loc[0, 'stamp']
    psf = star.results.loc[0, 'klip_model'][1][2:-2, 2:-2].copy()
    scale = 100
    scaled_psf = cutils.dutils.make_normalized_psf(psf, None, 100)
    scaled_psf_flux = cutils.dutils.apply_matched_filter(
        scaled_psf, psf, correlate_mode='valid'
    ).squeeze()
    assert(np.abs(scaled_psf_flux - scale) < 1e-10)
    assert(np.abs(scaled_psf.sum() - scale) < 1e-10)

def test_inject_subtract_detect(nonrandom_processed_star):
    star = nonrandom_processed_star
    print("Testing injections on ", star.star_id)
    stamps = cutils.inject_subtract_detect(star, (-3, -3), 1)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for ax, stamp in zip(axes, stamps):
        ax.imshow(stamp, origin='lower')
    plt.show()
