import pytest
from public_wifi import catalog_processing as catproc
from tests.conftest import data_folder

DEBUG = False # use to turn on debug printing

def test_load_catalog(catalog, catalog_file):
    assert(isinstance(catalog, catproc.pd.DataFrame))
    # make sure you have the required columns
    required_columns = ['target', 'file', 'x', 'y']
    assert(all([c in catalog.columns for c in required_columns]))
    # make sure you subtract off the 1 from the original coordinates
    default_catalog = catproc.pd.read_csv(str(catalog_file), dtype=str)
    # make sure the entries match
    default_catalog = default_catalog.query(f"target in {list(catalog['target'].unique())}")
    default_xy = default_catalog[['x','y']].astype(float)
    # print(default_xy.iloc[0].values, catalog[['x','y']].iloc[0].values)
    thresh = 1e-10
    coord_diff = (catproc.np.array(default_xy) - catproc.np.abs(catalog[['x','y']]) - 1)
    assert(all([all(i < thresh) for i in coord_diff.values]))


def test_initialize_stars(catalog, data_folder):
    bad_reference = 'J162810.30-264024.2'
    stars = catproc.initialize_stars(
        catalog,
        star_id_column='target',
        match_references_on=['filter'],
        data_folder=data_folder,
        stamp_size=15,
        bad_references=bad_reference,
        min_nrefs = 10,
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
    assert('snrmap' in star.results.columns)
    # try one row
    row = star.results.iloc[0]
    snr_map = star.row_make_snr_map(row)
    assert(len(snr_map['snrmap'])==len(row['kl_sub']))
    snr_maps = star.results.apply(star.row_make_snr_map, axis=1)
    assert(isinstance(snr_maps, sc.pd.DataFrame))
