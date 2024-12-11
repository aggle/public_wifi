import pytest
from public_wifi import starclass


def test_load_catalog(catalog, catalog_file):
    assert(isinstance(catalog, starclass.pd.DataFrame))
    # make sure you subtracted off the 1 from the original coordinates
    default_catalog = starclass.pd.read_csv(str(catalog_file), dtype=str)
    default_xy = default_catalog[['x','y']].astype(float)
    # print(default_xy.iloc[0].values, catalog[['x','y']].iloc[0].values)
    thresh = 1e-10
    coord_diff = (default_xy - catalog[['x','y']] - 1).apply(starclass.np.abs)
    assert(all([all(i < thresh) for i in coord_diff.values]))



def test_starclass_init(star):
    assert(isinstance(star, starclass.Star))
    assert(hasattr(star, 'star_id'))
    assert(hasattr(star, 'meta'))
    assert(isinstance(star.meta, starclass.pd.DataFrame))
    assert(isinstance(star.meta.iloc[0]['x'], float))
    assert('stamp_id' in star.meta.columns)


def test_starclass_check_reference(star):
    assert(star.has_companions == False)
    print(star.is_good_reference)
    # test that updating has_companions also updates is_good_reference
    star.has_companions = True
    assert(star.is_good_reference == False)
    star.has_companions = False
    assert(star.is_good_reference == True)

def test_starclass_get_stamp(star, data_folder):
    print(star.star_id)
    assert(data_folder.exists())
    stamp_size = 15
    stamps = star.meta.apply(lambda row: star.get_stamp(row, stamp_size, data_folder), axis=1)
    assert(all(stamps.apply(lambda el: el.shape == (stamp_size, stamp_size))))
    assert(all(stamps.apply(lambda el: isinstance(el, starclass.Cutout2D))))
    maxes = stamps.apply(lambda s: starclass.np.unravel_index(s.data.argmax(), s.data.shape))
    centers = stamps.apply(lambda s: tuple(int(i) for i in s.center_cutout)[::-1])
    print('maxes', maxes.values)
    print('centers', centers.values)
    assert(all([m == c for m, c in zip(maxes, centers)]))
    
def test_set_references(catalog, data_folder):
    stars = catalog.groupby("target").apply(
        lambda group: starclass.Star(group.name, group, data_folder=data_folder),
        include_groups=False
    )
    # make a bad reference and make sure it is not included
    bad_star = stars.iloc[1]
    bad_star.is_good_reference = False
    star = stars.iloc[0]
    references = star.set_references(stars)
    assert(isinstance(references, starclass.pd.DataFrame))
    assert(len(references) < len(catalog))
    assert(bad_star.star_id not in references.index.get_level_values("target"))

def test_klip_subtract(all_stars):
    star_id = starclass.np.random.choice(all_stars.index)
    print("KLIP subtraction tested on ", star_id)
    star = all_stars.loc[star_id]
    star.set_references(all_stars)
    star.subtraction = star.meta.apply(star.row_klip_subtract, axis=1)
    # the RMS should be monotonically declining
    rms_descent = star.subtraction['subtracted'].apply(
        lambda sub: all(starclass.np.diff(sub.apply(starclass.np.nanstd)) < 0)
    )
    assert(rms_descent.all())

def test_construct_psf_model(all_stars):
    star_id = starclass.np.random.choice(all_stars.index)
    print("PSF model construction tested on ", star_id)
    star = all_stars.loc[star_id]
    star.set_references(all_stars)
    star.subtraction = star.meta.apply(star.row_klip_subtract, axis=1)
    star.results = star.meta.join(star.subtraction)
    psf_models = star.results.apply(
        star.row_build_psf_model,
        axis=1
    )
    print(psf_models[0])
