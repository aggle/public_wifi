import pytest
import numpy as np
from matplotlib import pyplot as plt

from public_wifi import contrast_utils as cutils
from public_wifi import detection_utils as dutils


def test_measure_primary_flux(random_processed_star):
    """Check that you are measuring the correct flux of a star"""
    # construct a PSF with known flux
    star = random_processed_star
    print("Testing injections on ", star.star_id)
    row = star.cat.loc[0]
    stamp = row['stamp']
    input_flux = 10
    psf = dutils.make_normalized_psf(stamp, width=7, scale=input_flux)
    # make sure the psf was set correctly
    assert(np.abs(input_flux - psf.sum()) <= 1e-5)
    # measure the flux
    meas_flux = cutils.measure_primary_flux(psf, psf.copy())
    print("measured:", meas_flux)
    assert(np.abs(meas_flux - input_flux) <= 1e-5)

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

# @pytest.mark.parametrize(
#     ['scale', 'is_detectable'],
#     [(1, True), (1e-10, False)] # one is detectable the other isn't
# )
def test_inject_subtract_detect(nonrandom_processed_star, scale=1., is_detectable=True):
    star = nonrandom_processed_star
    print("Testing injections on ", star.star_id)
    center = cutils.misc.get_stamp_center(star.cat.iloc[0]['stamp'])
    pos = np.array((-2, -1))
    inj_pos = center + pos[::-1]
    row = star.cat.iloc[1]
    results = cutils.row_inject_subtract_detect(
        star,
        row,
        pos,
        contrast=scale,
        sim_thresh=0.5,
        min_nref=5,
        snr_thresh=5,
        n_modes=3
    )
    print(results)
    # assert(results.all() == is_detectable)
    # print(results.all(), is_detectable)

def test_build_contrast_curve(nonrandom_processed_star):
    star = nonrandom_processed_star
    row = star.cat.iloc[-1]
    result = cutils.build_contrast_curve(star, row, 0.5, 5, 5, 3)
    # print(result)
    # print(cutils.pd.concat(result, axis=1))
    print(result)

