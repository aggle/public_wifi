import pytest

import itertools
import pandas as pd
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from public_wifi import misc
from public_wifi import matched_filter_utils as mfutils
from public_wifi import contrast_utils as cutils

def load_random_star(list_of_stars):
    star = np.random.choice(list_of_stars)
    print(f"Randomly selected star: {star.star_id}")
    return star

@pytest.mark.parametrize("scale", [0.1, 1, 10])
def test_make_normalized_psf(all_stars, scale):
    """Check that you can normalize a PSF to arbitrary flux"""
    star = load_random_star(all_stars)
    psf = star.cat.loc[0, 'stamp']
    normed_psf =  mfutils.make_normalized_psf(psf, scale=scale)
    assert(np.abs(normed_psf.sum()/scale - 1) <= 1e-5)


def test_make_matched_filter(subtracted_stars):
    """Test the matched filtering"""
    # compute literally all the matched filters
    mfs = np.concatenate(subtracted_stars.apply(
            lambda star: np.concatenate(star.results['klip_model'].apply(
                lambda klip_model: np.stack(klip_model.apply(
                    # the klip_model entries are series
                    mfutils.make_matched_filter,
                    width=7
                ).values)
            ).values)
    ).values)


    # check that they have no nans
    assert(~np.isnan(mfs).any())
    # check that they have mean 0
    mf_means = np.mean(mfs, axis=(-1, -2))
    assert((np.abs(mf_means) <= 1e-15).all())



def test_apply_matched_filter(random_processed_star):
    star = random_processed_star
    print("Testing matched filter accuracy on ", star.star_id)
    mf = mfutils.make_matched_filter(star.results.loc[0, 'klip_model'][1], 7)
    stamp = (mf - mf.min())/mfutils.np.ptp(mf)
    stamp = stamp / stamp.sum()
    print(f"Stamp sum: {stamp.sum():0.10e}")
    stamp_center = misc.get_stamp_center(stamp)
    mf_flux = mfutils.apply_matched_filter_to_stamp(stamp, mf)[*stamp_center]
    print(f"MF flux: {mf_flux:0.10e}")
    assert(mfutils.np.abs(mf_flux - stamp.sum()) < 1e-10)
    mf_flux = mfutils.apply_matched_filter_to_stamp(
        stamp, mf, correlate_mode='valid'
    ).squeeze()
    print(f"MF flux, 'valid' mode: {mf_flux:0.10e}")
    assert(mfutils.np.abs(mf_flux - stamp.sum()) < 1e-10)

def test_matched_filter_on_normalized_psf(processed_stars):
    for star in processed_stars:
        star.results['matched_filter'] = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: mfutils.pd.Series(
                {series.name: series.apply(
                    mfutils.make_matched_filter
                )}
            ),
        )
        star.results['normalized_psf'] = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: mfutils.pd.Series(
                {series.name: series.apply(
                    mfutils.make_normalized_psf
                )}
            ),
        )
        test_columns = ['matched_filter', 'normalized_psf']
        assert(all([i in star.results.columns for i in test_columns])
        )

def test_row_apply_matched_filter(processed_stars):
    star = processed_stars[mfutils.np.random.choice(processed_stars.index)]
    row = star.results.iloc[0]
    row_detmaps = star._row_apply_matched_filter(row)['detmap']
    assert(len(row_detmaps) == len(row['klip_model']))
    all_detmaps = star.results.apply(
        star._row_apply_matched_filter,
        axis=1
    )
    assert(len(all_detmaps) == len(star.results))

def test_compute_throughput_noKLmodes(random_processed_star):
    star = random_processed_star
    row = star.results.loc[1]
    mf = mfutils.make_matched_filter(row['klip_model'].iloc[-1], 7)
    # compute throughput with no KL basis
    thpt = mfutils.compute_throughput(mf)
    dot = mfutils.np.dot(mf.ravel(), mf.ravel())
    # print(thpt, dot)
    assert(mfutils.np.abs(thpt - dot) <= 1e-15)


def test_compute_throughput_KLmodes(random_processed_star):
    star = random_processed_star
    row = star.results.loc[1]
    kl_basis = row['klip_basis']
    mf = mfutils.make_matched_filter(row['klip_model'].iloc[-1], 7)
    # compute throughput with no KL basis
    # print("klbasis", dutils.np.stack(kl_basis).shape)
    thpt = mfutils.compute_throughput(mf, kl_basis)
    dot = mfutils.np.dot(mf.ravel(), mf.ravel())
    # print(dot, thpt)
    # print(thpt.shape)
    # check that the throughput is always less than the flux with no KL basis
    assert(np.stack(kl_basis).shape == np.stack(thpt).shape)
    assert(np.ndim(np.stack(thpt)) == 3)
    assert(((dot - thpt).apply(lambda el: (el >= 0).all())).all())



positions = [
    (3, 3), (3, 0), (3, -3),
    (0, 3), (0, -3),
    (-3, 3), (-3, 0), (-3, -3)
]
contrasts = [1, 0.01]

@pytest.mark.parametrize(
    "pos,contrast",
    list(itertools.product(positions, contrasts))
)
def test_apply_matched_filter_with_throughput(
        nonrandom_subtracted_star,
        pos,
        contrast
):
    """
    Construct a situation where you should be able to correct for the
    throughput exactly
    - inject an off-center PSF into a stamp of the PSF model
    - subtract the existing PSF model from the injected stamp
    - run the matched filter
    - the recovered throughput should be exact
    """
    star = nonrandom_subtracted_star
    print("Testing injections on ", star.star_id)
    pos = np.array(pos)
    contrast = float(contrast)

    row = star.cat.iloc[1].copy()

    # pull out the different KLIP products
    klip_cols = ['klip_model', 'klip_sub', 'klip_basis']
    klip_df = pd.concat(
        star.results.loc[row.name, klip_cols].to_dict(),
        axis=1
    )

    # pick a Kklip
    kklip = klip_df.index[-1]
    psf_model = klip_df.loc[kklip, 'klip_model']
    resid_stamp = klip_df.loc[kklip, 'klip_sub']
    klip_basis = klip_df.loc[:kklip, 'klip_basis']
    assert(kklip == len(klip_basis))

    row['stamp'] = psf_model.copy()
    center = misc.get_stamp_center(row['stamp'])
    inj_pos = center + pos[::-1]
    inj_row = cutils.row_inject_psf(
        row, star=star, pos=pos, contrast=contrast, kklip=-1
    )
    resid_stamp = inj_row['stamp'] - psf_model
    # apply the matched filter with throughput correction
    kklip = -1
    primary_flux = cutils.measure_primary_flux(row['stamp'], psf_model)
    detmap = mfutils.apply_matched_filter_to_stamp(
        resid_stamp, psf_model, throughput_correction=True, kl_basis=None
    )
    mf_result = mfutils.apply_matched_filter_to_stamp(
        resid_stamp, psf_model, throughput_correction=True, kl_basis=klip_basis
    )  / primary_flux

    # recover flux
    recovered_contrast = mf_result[*inj_pos]

    print(f"Primary flux: {primary_flux}")
    print(f"Input contrast: {contrast:0.2e}")
    print(f"Recovered contrast: {recovered_contrast:0.2e}")
    print(f"Error: {(np.abs(contrast-recovered_contrast)/contrast)*100:0.2f} %")

    # plt.imshow(resid_stamp, origin='lower')
    # plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.suptitle(f"Injection: {tuple(pos)} @ {contrast:0.2e}")
    ax = axes[0]
    ax.set_title("Unsubtracted injection")
    imax = ax.imshow(inj_row['stamp'], origin='lower')
    ax.scatter(*inj_pos[::-1], marker='x', c='k')
    fig.colorbar(imax, ax=ax)
    ax = axes[1]
    ax.set_title("Detection map")
    imax = ax.imshow(detmap, origin='lower')
    ax.scatter(*inj_pos[::-1], marker='x', c='k')
    fig.colorbar(imax, ax=ax)
    ax = axes[2]
    ax.set_title("Matched filter result (contrast)")
    imax = ax.imshow(mf_result, origin='lower')
    ax.scatter(*inj_pos[::-1], marker='x', c='k')
    fig.colorbar(imax, ax=ax)
    plt.show()

    # test if you're within 20% of the injected flux
    # assert(np.abs(recovered_contrast / contrast - 1) <= 2e-1)
