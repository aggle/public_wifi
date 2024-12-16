import pytest
from public_wifi import starclass as sc


def test_make_matched_filter(processed_stars):
    """Test the matched filtering"""
    assert(len(processed_stars) == 10)
    mfs = processed_stars.apply(
        # apply to each star
        lambda star: star.results['klip_model'].apply(
            # apply to each row of the results dataframe
            lambda klip_model: klip_model.apply(
                # the klip_model entries are series
                sc.make_matched_filter,
                width=5
            )
        )
    )

    mf_means = mfs.apply(
        lambda star: star.apply(
            lambda row: row.apply(sc.np.nanmean)
        )
    )
    for star_id in mf_means.index:
        for row_id in mf_means.loc[star_id].index:
            for kklip in mf_means.loc[star_id].loc[row_id].index:
                val = mf_means.loc[star_id].loc[row_id].loc[kklip]
                if sc.np.isnan(val):
                    print(f"NaN encountered: {star_id} {row_id} {kklip} {val}")
                assert(sc.np.abs(val) < 1e-15)

def test_make_normalized_psf(processed_stars):

    psfs = processed_stars.apply(
        # apply to each star
        lambda star: star.results['klip_model'].apply(
            # apply to each row of the results dataframe
            lambda klip_model: klip_model.apply(
                # the klip_model entries are series
                sc.make_normalized_psf,
            )
        )
    )
    psf_sums = psfs.apply(
        lambda star: star.apply(
            lambda row: row.apply(sc.np.nanmean)
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
            sc.make_matched_filter,
        )
        star.results['normalized_psf'] = star.results['klip_model'].apply(
            # the klip_model entries are series
            sc.make_normalized_psf,
        ) 
        assert(
            all([i in star.results.columns for i in ['matched_filter', 'normalized_psf']])
        )

def test_row_make_detection_map(processed_stars):
    star = processed_stars[sc.np.random.choice(processed_stars.index)]
    row = star.results.iloc[0]
    row_detmaps = star.row_make_detection_maps(row)
    assert(len(row_detmaps) == len(row['klip_model']))
    all_detmaps = star.results.apply(
        star.row_make_detection_maps,
        axis=1
    )
    print(all_detmaps)

def test_snr_map(random_processed_star):
    star = random_processed_star
    print(f"Randomly chosen star: {star.star_id}")
    assert('snrmap' in star.results.columns)
    # assert(len(star.results.snrmap.iloc[0]) == len(star.results['kl_sub'] c)
    assert(
        all(
            star.results.apply(
                lambda row: sc.np.shape(sc.np.stack(row['kl_sub'])) == sc.np.shape(sc.np.stack(row['snrmap'])),
                axis=1
            )
        )
    )
