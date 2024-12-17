import pytest
import numpy as np
from matplotlib import pyplot as plt
from public_wifi import contrast_utils as cu


def test_inject_psf(processed_stars):
    # star = random_processed_star
    star = processed_stars.iloc[3]
    print("Testing injections on ", star.star_id)
    stamp = star.cat.loc[0, 'stamp']
    psf = star.results.loc[0, 'klip_model'][1][2:-2, 2:-2].copy()
    stamp_center = np.floor(np.array(stamp.shape)/2).astype(int)
    psf_center = np.floor(np.array(psf.shape)/2).astype(int)

    pos = (2, -3)
    pos_index = stamp_center + pos[::-1]

    injected_stamp = cu.inject_psf(stamp, psf, pos)
    injected_flux = injected_stamp[*pos_index]
    test_flux = stamp[*pos_index] + psf[*psf_center]
    # fig, ax = plt.subplots(nrows=1, ncols=1,)
    # ax.imshow(injected_stamp, origin='lower')
    # plt.show()
    assert(cu.np.abs(injected_flux - test_flux) < 1e-6)

def test_scale_psf():
    assert(False)
