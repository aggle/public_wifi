import pytest
from public_wifi import starclass as sc
from tests.conftest import data_folder


def test_load_catalog(catalog, catalog_file):
    assert(isinstance(catalog, sc.pd.DataFrame))
    # make sure you have the required columns
    required_columns = ['target', 'file', 'x', 'y']
    assert(all([c in catalog.columns for c in required_columns]))
    # make sure you subtract off the 1 from the original coordinates
    default_catalog = sc.pd.read_csv(str(catalog_file), dtype=str)
    default_xy = default_catalog[['x','y']].astype(float)
    # print(default_xy.iloc[0].values, catalog[['x','y']].iloc[0].values)
    thresh = 1e-10
    coord_diff = (default_xy - catalog[['x','y']] - 1).apply(sc.np.abs)
    assert(all([all(i < thresh) for i in coord_diff.values]))


def test_starclass_init(star):
    assert(isinstance(star, sc.Star))
    assert(hasattr(star, 'star_id'))
    assert(hasattr(star, 'cat'))
    assert(isinstance(star.cat, sc.pd.DataFrame))
    assert(isinstance(star.cat.iloc[0]['x'], float))
    assert('cat_id' in star.cat.columns)

def test_starclass_check_reference(star):
    assert(star.has_companions == False)
    print(star.is_good_reference)
    # test that updating has_companions also updates is_good_reference
    star.has_companions = True
    assert(star.is_good_reference == False)
    star.has_companions = False
    assert(star.is_good_reference == True)

def test_starclass_get_stamp(star, data_folder):
    print("Getting stamp from " + star.star_id)
    assert(data_folder.exists())
    stamp_size = 15
    stamps = star.cat.apply(lambda row: star.get_stamp(row, stamp_size, data_folder), axis=1)
    assert(all(stamps.apply(lambda el: el.shape == (stamp_size, stamp_size))))
    assert(all(stamps.apply(lambda el: isinstance(el, sc.Cutout2D))))
    maxes = stamps.apply(lambda s: sc.np.unravel_index(s.data.argmax(), s.data.shape))
    centers = stamps.apply(lambda s: tuple(int(i) for i in s.center_cutout)[::-1])
    # print('maxes', maxes.values)
    # print('centers', centers.values)
    assert(all([m == c for m, c in zip(maxes, centers)]))

def test_set_references(catalog, data_folder):
    stars = catalog.groupby("target").apply(
        lambda group: sc.Star(group.name, group, data_folder=data_folder),
        include_groups=False
    )
    # make a bad reference and make sure it is not included
    bad_star = stars.iloc[1]
    bad_star.is_good_reference = False
    star = stars.iloc[0]
    star.set_references(stars)
    # print(star.references)
    assert(isinstance(star.references, sc.pd.DataFrame))
    assert(len(star.references) < len(catalog))
    assert(bad_star.star_id not in star.references.index.get_level_values("target"))

def test_query_references(all_stars):
    # test that the references are all appropriate
    star_id = sc.np.random.choice(all_stars.index)
    print("Reference query tested on ", star_id)
    star = all_stars.loc[star_id]
    star.set_references(all_stars)
    # split the references up for each stamp
    ref_subsets = star.cat.apply(
        lambda row: star.references.query(star.generate_match_query(row)),
        axis=1
    )
    # test that the references all match on the queried values
    assert(all(star.cat.apply(
        lambda row: all(
            [row[m] == ref_subsets.loc[row.name][m].unique().squeeze()
             for m in star.match_by]
        ),
        axis=1
    )))
    # test that the references do not overlap
    for ind in ref_subsets.index:
        row_refs = ref_subsets[ind].index
        other_index = [j for j in ref_subsets.index if j != ind]
        for oth in other_index:
            other_refs = ref_subsets[oth].index
            assert(set(row_refs).isdisjoint(set(other_refs)))

def test_klip_subtract(all_stars):
    star_id = sc.np.random.choice(all_stars.index)
    print("KLIP subtraction tested on ", star_id)
    star = all_stars.loc[star_id]
    star.set_references(all_stars, compute_similarity=True)
    star.subtraction = star.cat.apply(star.row_klip_subtract, axis=1)
    # the RMS should be monotonically declining
    rms_descent = star.subtraction['kl_sub'].apply(
        lambda sub: all(sc.np.diff(sub.apply(sc.np.nanstd)) < 0)
    )
    assert(rms_descent.all())

def test_similarity(all_stars):
    star_id = sc.np.random.choice(all_stars.index)
    print("SSIM tested on ", star_id)
    star = all_stars.loc[star_id]
    star.set_references(all_stars)
    star.compute_similarity()
    # now see if the similarity was set
    assert('sim' in star.references.columns)
    assert(all(star.references['sim']**2 <= 1.))
    assert(any(star.references['sim'].isna()) == False)


def test_process_stars(catalog, data_folder):
    stars = sc.process_stars(
        catalog,
        'target',
        match_references_on=['filter'],
        data_folder=data_folder,
        stamp_size=15
    )
    assert(isinstance(stars, sc.pd.Series))
    # check that one star is made for each unique star in the catalog
    star_ids = set(stars.apply(lambda s: s.star_id))
    cat_stars = set(catalog['target'].unique())
    assert(star_ids == cat_stars)
    # assert(len(stars) == len(catalog['target'].unique()))
    # check that attributes have been assigned
    assert(all(stars.apply(lambda s: hasattr(s, 'cat'))))


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

