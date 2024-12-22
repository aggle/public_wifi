import pytest
import numpy as np
from matplotlib import pyplot as plt
from public_wifi import contrast_utils as cutils
from public_wifi import detection_utils as dutils
from tests.conftest import nonrandom_processed_star


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


@pytest.mark.parametrize('scale', list(range(1, 21)))
def test_row_inject_psf(nonrandom_processed_star, scale):
    star = nonrandom_processed_star
    row = star.cat.iloc[1]
    # scale = 10
    inj_row = cutils.row_inject_psf(row, star, (0, 0), scale, -1)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axes[0].imshow(row['stamp'])
    # axes[1].imshow(inj_row['stamp'])
    # plt.show()
    inj_flux = cutils.measure_primary_flux(inj_row['stamp'], row['stamp'])
    stamp_flux = cutils.measure_primary_flux(row['stamp'], row['stamp'])
    flux_ratio = inj_flux/stamp_flux
    print(inj_flux, stamp_flux, flux_ratio)
    # let's give ourselves a 5% margin
    assert(np.abs(flux_ratio/(scale+1) - 1) <= 0.05)

def test_make_injected_cat(nonrandom_processed_star):
    star = nonrandom_processed_star
    # center = np.floor(
    #     np.array(np.stack(star.cat['stamp']).shape[-2:])/2
    # ).astype(int)
    center = cutils.misc.get_stamp_center(star.cat.iloc[0]['stamp'])
    pos = np.array([ -3, 3 ])
    scale = 1
    cat = cutils.make_injected_cat(star, pos, scale, -1)
    inj_stamps = cat['stamp']
    # the flux at the injection site should be greater than the central flux
    # for scale=1
    inj_pos = center + pos[::-1]
    assert(
        inj_stamps.apply(
            lambda stamp: stamp[*inj_pos] > stamp[*center]
        ).all()
    )

def test_inject_subtract_detect(nonrandom_processed_star):
    star = nonrandom_processed_star
    print("Testing injections on ", star.star_id)
    center = cutils.misc.get_stamp_center(star.cat.iloc[0]['stamp'])
    pos = np.array((-3, -3))
    inj_pos = center + pos[::-1]
    results = cutils.inject_subtract_detect(star, pos, 1, 0.5, 5)
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # for ax, stamp in zip(axes, stamps):
    #     ax.imshow(stamp, origin='lower')
    # plt.show()
    print(results)
    # assert('klip_sub' in results.columns)

