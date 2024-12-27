import pytest

from public_wifi import starclass as sc
from public_wifi import detection_utils as dutils
from public_wifi import misc


def test_make_matched_filter(processed_stars):
    """Test the matched filtering"""
    mfs = processed_stars.apply(
        # apply to each star
        lambda star: star.results['klip_model'].apply(
            # apply to each row of the results dataframe
            lambda klip_model: klip_model.apply(
                # the klip_model entries are series
                dutils.make_matched_filter,
                width=5
            )
        )
    )
    # each matched filter should have mean 0
    mf_means = mfs.apply(
        lambda star: star.apply(
            lambda row: row.apply(dutils.np.nanmean)
        )
    )
    for mf in mf_means:
        for i, row in mf.iterrows():
            assert((row.dropna()**2 < 1e-30).all())


def test_apply_matched_filter(random_processed_star):
    star = random_processed_star
    print("Testing matched filter accuracy on ", star.star_id)
    mf = dutils.make_matched_filter(star.results.loc[0, 'klip_model'][1], 7)
    stamp = (mf - mf.min())/dutils.np.ptp(mf)
    stamp = stamp / stamp.sum()
    print(f"Stamp sum: {stamp.sum():0.10e}")
    stamp_center = misc.get_stamp_center(stamp)
    mf_flux = dutils.apply_matched_filter(
        stamp, mf, throughput_correction=True, kl_basis=None
    )[*stamp_center]
    print(f"MF flux: {mf_flux:0.10e}")
    assert(dutils.np.abs(mf_flux - stamp.sum()) < 1e-10)

def test_matched_filter_on_normalized_psf(processed_stars):
    """What is the goal of this test?"""
    for star in processed_stars:
        matched_filter = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: dutils.pd.Series(
                {series.name: series.apply(
                    dutils.make_matched_filter
                )}
            ),
        )
        normalized_psf = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: dutils.pd.Series(
                {series.name: series.apply(
                    dutils.make_normalized_psf
                )}
            ),
        )

def test_row_convolve_psf(processed_stars):
    """Test that _row_convolve_psf returns the right types and shapes"""
    star = processed_stars[dutils.np.random.choice(processed_stars.index)]
    row = star.results.iloc[0]
    row_detmaps = star._row_convolve_psf(row)['detmap']
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
    """
    in the case of no KL modes, throughput should return a float
    in the case of KL modes, throughput should return an array
    """
    star = random_processed_star
    row = star.results.loc[1]
    n_modes = -2
    kl_basis = row['klip_basis'].iloc[:n_modes]
    mf = dutils.make_matched_filter(row['klip_model'].iloc[n_modes], 7)
    # compute throughput with no KL basis
    # print("klbasis", dutils.np.stack(kl_basis).shape)
    thpt = dutils.compute_throughput(mf, kl_basis)
    dot = dutils.np.dot(mf.ravel(), mf.ravel())
    # print(dot, thpt)
    # print(thpt.shape)
    # check that the throughput is always less than the flux with no KL basis
    assert(thpt.ndim == 3)
    assert(len(thpt) == len(kl_basis))
    assert(((dot - thpt) >= 0).all())
    assert(dutils.np.stack(kl_basis).shape == dutils.np.stack(thpt).shape)


def test_apply_matched_filter_with_throughput(random_processed_star):
    """Test that asking for throughput correction gets the right value"""
    star = random_processed_star
    ind = 1
    row = star.results.loc[ind]
    kl_basis = row['klip_basis']
    stamp = star.results.loc[ind, 'klip_sub']
    print("Testing matched filter throughput on ", star.star_id)
    mf_flux_noKL = star.apply_matched_filter(contrast=True, throughput_correction=False).loc[ind]
    mf_flux_KL = star.apply_matched_filter(contrast=True, throughput_correction=True).loc[ind]
    print(mf_flux_noKL)
    print(mf_flux_KL)
    assert(dutils.np.stack(mf_flux_noKL).shape == dutils.np.stack(mf_flux_KL).shape)
    assert(
        (dutils.np.stack(mf_flux_noKL).std(axis=0) <= dutils.np.stack(mf_flux_KL).std(axis=0)).all()
    )

def test_matched_filter_throughput_photometry(
        random_processed_star
):
    """
    Make up a fake system and do PSF subtraciton on it. Then recover the
    photometry with throughput correction and check that it's correct.
    """
    star = random_processed_star
    pass
