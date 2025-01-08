import pytest

import itertools
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from public_wifi import misc
from public_wifi import detection_utils as dutils
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
    normed_psf =  dutils.make_normalized_psf(psf, 7, scale=scale)
    assert(np.abs(normed_psf.sum()/scale - 1) <= 1e-5)


def test_make_matched_filter(subtracted_stars):
    """Test the matched filtering"""
    # compute literally all the matched filters
    mfs = np.concatenate(subtracted_stars.apply(
            lambda star: np.concatenate(star.results['klip_model'].apply(
                lambda klip_model: np.stack(klip_model.apply(
                    # the klip_model entries are series
                    dutils.make_matched_filter,
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
    mf = dutils.make_matched_filter(star.results.loc[0, 'klip_model'][1], 7)
    stamp = (mf - mf.min())/dutils.np.ptp(mf)
    stamp = stamp / stamp.sum()
    print(f"Stamp sum: {stamp.sum():0.10e}")
    stamp_center = misc.get_stamp_center(stamp)
    mf_flux = dutils.apply_matched_filter(stamp, mf)[*stamp_center]
    print(f"MF flux: {mf_flux:0.10e}")
    assert(dutils.np.abs(mf_flux - stamp.sum()) < 1e-10)
    mf_flux = dutils.apply_matched_filter(
        stamp, mf, correlate_mode='valid'
    ).squeeze()
    print(f"MF flux, 'valid' mode: {mf_flux:0.10e}")
    assert(dutils.np.abs(mf_flux - stamp.sum()) < 1e-10)

def test_matched_filter_on_normalized_psf(processed_stars):
    for star in processed_stars:
        star.results['matched_filter'] = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: dutils.pd.Series(
                {series.name: series.apply(
                    dutils.make_matched_filter
                )}
            ),
        )
        star.results['normalized_psf'] = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: dutils.pd.Series(
                {series.name: series.apply(
                    dutils.make_normalized_psf
                )}
            ),
        )
        test_columns = ['matched_filter', 'normalized_psf']
        assert(all([i in star.results.columns for i in test_columns])
        )

def test_row_apply_matched_filter(processed_stars):
    star = processed_stars[dutils.np.random.choice(processed_stars.index)]
    row = star.results.iloc[0]
    row_detmaps = star.row_convolve_psf(row)['detmap']
    assert(len(row_detmaps) == len(row['klip_model']))
    all_detmaps = star.results.apply(
        star.row_convolve_psf,
        axis=1
    )
    assert(len(all_detmaps) == len(star.results))

def test_compute_throughput_noKLmodes(random_processed_star):
    star = random_processed_star
    row = star.results.loc[1]
    mf = dutils.make_matched_filter(row['klip_model'].iloc[-1], 7)
    # compute throughput with no KL basis
    thpt = dutils.compute_throughput(mf)
    dot = dutils.np.dot(mf.ravel(), mf.ravel())
    # print(thpt, dot)
    assert(dutils.np.abs(thpt - dot) <= 1e-15)


def test_compute_throughput_KLmodes(random_processed_star):
    star = random_processed_star
    row = star.results.loc[1]
    kl_basis = row['klip_basis']
    mf = dutils.make_matched_filter(row['klip_model'].iloc[-1], 7)
    # compute throughput with no KL basis
    # print("klbasis", dutils.np.stack(kl_basis).shape)
    thpt = dutils.compute_throughput(mf, kl_basis)
    dot = dutils.np.dot(mf.ravel(), mf.ravel())
    # print(dot, thpt)
    # print(thpt.shape)
    # check that the throughput is always less than the flux with no KL basis
    assert(np.stack(kl_basis).shape == np.stack(thpt).shape)
    assert(np.ndim(np.stack(thpt)) == 3)
    assert(((dot - thpt).apply(lambda el: (el >= 0).all())).all())


# @pytest.mark.xfail
# @pytest.mark.parametrize(
#     "pos,contrast",
#     [
#         # ((-3, -2), 0.1), ((1, -1), 0.5)
#         ((-1, -1), 1),((-1, -1), 0.1),((-1, -1), 0.01),((-1, -1), 0.001),
#     ]
# )
# def test_apply_matched_filter_with_throughput(
#         star_with_candidates,
#         pos, contrast,
# ):
#     star = star_with_candidates
#     print("Testing injections on ", star.star_id)
#     pos = np.array(pos)
#     contrast = float(contrast)

#     row = star.cat.iloc[1]
#     center = cutils.misc.get_stamp_center(row['stamp'])
#     inj_pos = center + pos[::-1]
#     inj_row = cutils.row_inject_psf(
#         row, star=star, pos=pos, contrast=contrast, kklip=-1
#     )
#     results = star._row_klip_subtract(
#         inj_row,
#         **star.subtr_args,
#     )
#     # apply the matched filter with throughput correction
#     kklip = -1
#     psf_model = results['klip_model'].iloc[kklip]
#     resid_stamp = results['klip_sub'].iloc[kklip]
#     klip_basis = results['klip_basis'][:kklip]
#     primary_flux = cutils.measure_primary_flux(row['stamp'], psf_model)
#     mf_results = dutils.apply_matched_filter(
#         resid_stamp, psf_model, throughput_correction=True, kl_basis=klip_basis
#     )  / primary_flux

#     # recover flux
#     recovered_contrast = mf_results[*inj_pos]
#     print(f"Input contrast: {contrast:0.2e}")
#     print(f"Recovered contrast: {recovered_contrast:0.2e}")
#     print(f"Error: {(np.abs(contrast-recovered_contrast)/contrast)*100:0.2f} %")
    

positions = [
    (1, 1), (1, 0), (1, -1),
    (0, 1), (0, -1),
    (-1, 1), (-1, 0), (-1, -1)
]
contrasts = [1, 0.01]

@pytest.mark.parametrize(
    "pos,contrast",
    list(itertools.product(positions, contrasts))
)
def test_apply_matched_filter_with_throughput(
        nonrandom_subtracted_star,
        pos,
        contrast,
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
    kklip = -1
    psf_model = star.results.loc[row.name, 'klip_model'].iloc[kklip].copy()
    resid_stamp = star.results.loc[row.name, 'klip_sub'].iloc[kklip].copy()
    klip_basis = star.results.loc[row.name, 'klip_basis'][:kklip].copy()
    row['stamp'] = psf_model.copy()
    center = cutils.misc.get_stamp_center(row['stamp'])
    inj_pos = center + pos[::-1]
    inj_row = cutils.row_inject_psf(
        row, star=star, pos=pos, contrast=contrast, kklip=-1
    )
    resid_stamp = inj_row['stamp'] - psf_model
    # apply the matched filter with throughput correction
    kklip = -1
    primary_flux = cutils.measure_primary_flux(row['stamp'], psf_model)
    detmap = dutils.apply_matched_filter(
        resid_stamp, psf_model, throughput_correction=True, kl_basis=None
    )
    mf_result = dutils.apply_matched_filter(
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
