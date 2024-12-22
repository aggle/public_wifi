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

    mf_means = mfs.apply(
        lambda star: star.apply(
            lambda row: row.apply(dutils.np.nanmean)
        )
    )
    for star_id in mf_means.index:
        for row_id in mf_means.loc[star_id].index:
            for kklip in mf_means.loc[star_id].loc[row_id].index:
                val = mf_means.loc[star_id].loc[row_id].loc[kklip]
                assert(dutils.np.abs(val) < 1e-15)

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
    mf_flux = dutils.apply_matched_filter(stamp, mf, correlate_mode='valid').squeeze()
    print(f"MF flux, 'valid' mode: {mf_flux:0.10e}")
    assert(dutils.np.abs(mf_flux - stamp.sum()) < 1e-10)

def test_make_normalized_psf(processed_stars):

    psfs = processed_stars.apply(
        # apply to each star
        lambda star: star.results['klip_model'].apply(
            # apply to each row of the results dataframe
            lambda klip_model: klip_model.apply(
                # the klip_model entries are series
                dutils.make_normalized_psf,
            )
        )
    )
    psf_sums = psfs.apply(
        lambda star: star.apply(
            lambda row: row.apply(dutils.np.nansum)
        )
    )
    for star_id in psf_sums.index:
        for row_id in psf_sums.loc[star_id].index:
            for kklip in psf_sums.loc[star_id].loc[row_id].index:
                val = psf_sums.loc[star_id].loc[row_id].loc[kklip]
                # print(f"{star_id} {row_id} {kklip} {val:0.2e}")
                assert(val - 1 < 1e-15)

def test_matched_filter_on_normalized_psf(processed_stars):
    for star in processed_stars:
        star.results['matched_filter'] = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: dutils.pd.Series({series.name: series.apply(dutils.make_matched_filter)}),
        )
        star.results['normalized_psf'] = star.results['klip_model'].apply(
            # the klip_model entries are series
            lambda series: dutils.pd.Series({series.name: series.apply(dutils.make_normalized_psf)}),
        )
        assert(
            all([i in star.results.columns for i in ['matched_filter', 'normalized_psf']])
        )

def test_row_convolve_psf(processed_stars):
    star = processed_stars[dutils.np.random.choice(processed_stars.index)]
    row = star.results.iloc[0]
    row_detmaps = star.row_convolve_psf(row)['detmap']
    assert(len(row_detmaps) == len(row['klip_model']))
    all_detmaps = star.results.apply(
        star.row_convolve_psf,
        axis=1
    )
    assert(len(all_detmaps) == len(star.results))

def test_snr_map(random_processed_star):
    star = random_processed_star
    print(f"Randomly chosen star: {star.star_id}")
    assert('snrmap' in star.results.columns)
    # assert(len(star.results.snrmap.iloc[0]) == len(star.results['klip_sub'] c)
    assert(
        all(
            star.results.apply(
                lambda row: dutils.np.shape(sc.np.stack(row['klip_sub'])) == sc.np.shape(sc.np.stack(row['snrmap'])),
                axis=1
            )
        )
    )


def test_flag_candidate_pixels(star_with_candidates):
    star = star_with_candidates
    snrmap = star.results['snrmap'].iloc[1]
    flags = dutils.flag_candidate_pixels(snrmap, 3, 3)
    assert(flags.ndim == 2)
    assert(flags.dtype == bool)
    assert(flags.any())

def test_detect_snrmap(star_with_candidates):
    star = star_with_candidates
    snrmap = star.results['snrmap'].iloc[1]
    candidates = dutils.detect_snrmap(snrmap, 5, 3)
    print(candidates)
    assert(isinstance(candidates, dutils.pd.DataFrame))
    assert(all([c in candidates.columns for c in ['cand_id', 'pixel']]))


@pytest.mark.parametrize('snr_thresh', [100, 5])
def test_detect_snrmap_dev(star_with_candidates, snr_thresh):
    star = star_with_candidates
    snrmap = star.results['snrmap'].iloc[1]
    candidates = dutils.detect_snrmap_dev(snrmap, snr_thresh, 3)
    print(snr_thresh, candidates)
    assert(isinstance(candidates, dutils.pd.DataFrame))
    assert(all([c in candidates.columns for c in ['cand_id', 'pixel']]))
