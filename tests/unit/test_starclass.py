import pytest
from public_wifi import starclass as sc
from tests.conftest import data_folder

DEBUG = False # use to turn on debug printing

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
    assert("stamp" in star.cat.columns)
    assert("cutout" in star.cat.columns)
    assert("bgnd" in star.cat.columns)


def test_starclass_check_reference(star):
    assert(star.has_companions == False)
    print(star.is_good_reference)
    # test that updating has_companions also updates is_good_reference
    star.has_companions = True
    assert(star.is_good_reference == False)
    star.has_companions = False
    assert(star.is_good_reference == True)


def test_starclass_get_cutout(star, data_folder):
    print("Getting stamp from " + star.star_id)
    assert(data_folder.exists())
    stamp_size = 15
    stamps = star.cat.apply(lambda row: star.get_cutout(row, stamp_size), axis=1)
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


def test_initialize_stars(catalog, data_folder):
    bad_reference = 'J162810.30-264024.2'
    stars = sc.initialize_stars(
        catalog,
        star_id_column='target',
        match_references_on=['filter'],
        data_folder=data_folder,
        stamp_size=15,
        bad_references=bad_reference,
    )
    unique_stars = catalog['target'].unique()
    
    assert(len(stars) == len(unique_stars))
    assert(all(stars.apply(lambda el: isinstance(el, sc.Star))))
    # check that the attributes are there
    attrs = ["star_id", "stamp_size", "is_good_reference", "data_folder",
             "cat", "has_companions", "match_by", "references"]
    for attr in attrs:
        if DEBUG:
            print(attr)
        assert(all(stars.apply(hasattr, args=[attr])))
    # check that the bad reference is flagged correctly
    assert(stars[bad_reference].is_good_reference == False)
    assert(all(
        stars.apply(
            lambda star: bad_reference not in star.references.reset_index()['target']
        )
    ))


def test_scale_stamp(star):
    scaled_stamps = star.cat['stamp'].apply(star.scale_stamp)
    for stamp in scaled_stamps:
        assert(sc.np.nanmin(stamp) < 1e-10)
        assert(sc.np.abs(sc.np.nanmax(stamp) - 1) < 1e-10)

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


def test_row_snr_map(random_processed_star):
    star = random_processed_star
    assert(hasattr(star, 'results'))
    assert(hasattr(star, 'row_make_snr_map'))
    row = star.results.iloc[0]
    snr_map = star.row_make_snr_map(row)
    assert(len(snr_map['snr_map'])==len(row['kl_sub']))
    snr_maps = star.results.apply(star.row_make_snr_map, axis=1)
    assert(isinstance(snr_maps, sc.pd.DataFrame))
    assert('snr_map' in star.results.join(snr_maps).columns)
