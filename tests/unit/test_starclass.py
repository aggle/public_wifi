import pytest
from public_wifi import starclass as sc

DEBUG = False # use to turn on debug printing

def test_load_catalog(catalog, catalog_file):
    assert(isinstance(catalog, sc.pd.DataFrame))
    # make sure you have the required columns
    required_columns = ['target', 'file', 'x', 'y']
    assert(all([c in catalog.columns for c in required_columns]))
    # make sure you subtract off the 1 from the original coordinates
    default_catalog = sc.pd.read_csv(str(catalog_file), dtype=str)
    # make sure the entries match
    default_catalog = default_catalog.query(f"target in {list(catalog['target'].unique())}")
    default_xy = default_catalog[['x','y']].astype(float)
    # print(default_xy.iloc[0].values, catalog[['x','y']].iloc[0].values)
    thresh = 1e-10
    coord_diff = (sc.np.array(default_xy) - sc.np.abs(catalog[['x','y']]) - 1)
    assert(all([all(i < thresh) for i in coord_diff.values]))


def test_starclass_init(star):
    assert(isinstance(star, sc.Star))
    assert(hasattr(star, 'star_id'))
    assert(hasattr(star, 'cat'))
    assert(hasattr(star, 'subtr_args'))
    assert(hasattr(star, 'det_args'))
    assert(isinstance(star.cat, sc.pd.DataFrame))
    assert(isinstance(star.cat.iloc[0]['x'], float))
    assert('cat_id' in star.cat.columns)
    assert("stamp" in star.cat.columns)
    assert("cutout" in star.cat.columns)
    assert("bgnd" in star.cat.columns)


def test_starclass_check_reference(star):
    assert(star.has_candidates == False)
    print(star.is_good_reference)
    # test that updating has_candidates also updates is_good_reference
    star.has_candidates = True
    assert(star.is_good_reference == False)
    star.has_candidates = False
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
    print(star.nrefs)
    assert(isinstance(star.nrefs, sc.pd.Series))
    assert(all(star.nrefs > 0))

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
    # run with default parameters
    star.subtraction = star.run_klip_subtraction()
    # the RMS should be monotonically declining
    rms_descent = star.subtraction['klip_sub'].apply(
        lambda sub: all(sc.np.diff(sub.apply(sc.np.nanstd)) < 0)
    )
    assert(rms_descent.all())

def test_jackknife_subtraction(star_with_candidates):
    star = star_with_candidates
    ref = sc.np.random.choice(star.references.index)[0]
    klsub = star.run_klip_subtraction(jackknife_reference=ref)['klip_sub']
    # get the number of jackknife reductions
    n_jackknife = klsub.apply(len).sum()
    # get the number of references
    n_refs = star.nrefs['Nrefs'].sum()
    # Kklip_max = n_refs - 1, and jackknife removes another reference
    # so there should be 2 fewer references for each reduction
    assert((n_refs - n_jackknife) == 2*len(star.cat))
    # star_jackknife = star.jackknife_analysis()

def test_row_snr_map(random_processed_star):
    star = random_processed_star
    assert(hasattr(star, 'results'))
    assert(hasattr(star, '_row_make_snr_map'))
    assert('snrmap' in star.results.columns)
    # try one row
    row = star.results.iloc[0]
    snr_map = star._row_make_snr_map(row)
    assert(len(snr_map['snrmap'])==len(row['klip_sub']))
    snr_maps = star.results.apply(star._row_make_snr_map, axis=1)
    assert(isinstance(snr_maps, sc.pd.DataFrame))

def test_run_make_snr_maps(random_processed_star):
    """
    Make sure self.run_make_snr_maps() properly assigns the SNR maps to the
    results dataframe
    """
    star = random_processed_star
    assert(hasattr(star, 'results'))
    assert(hasattr(star, 'run_make_snr_maps'))
    star.run_make_snr_maps()
    assert('snrmap' in star.results.columns)

@pytest.mark.parametrize('scale', list(range(1, 21)))
def test_row_inject_psf(nonrandom_processed_star, scale):
    star = nonrandom_processed_star
    row = star.cat.iloc[1]
    # scale = 10
    # inj_row = cutils.row_inject_psf(row, star, (0, 0), scale, -1)
    inj_row = star.row_inject_psf(row, (0, 0), scale, -1)
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axes[0].imshow(row['stamp'])
    # axes[1].imshow(inj_row['stamp'])
    # plt.show()
    inj_flux = sc.cutils.measure_primary_flux(inj_row['stamp'], row['stamp'])
    stamp_flux = sc.cutils.measure_primary_flux(row['stamp'], row['stamp'])
    flux_ratio = inj_flux/stamp_flux
    print(inj_flux, stamp_flux, scale, flux_ratio)
    # let's give ourselves a 5% margin
    assert(sc.np.abs(flux_ratio/(scale+1) - 1) <= 0.05)

@pytest.mark.parametrize('scale', list(range(1, 21)))
def test_inject_subtract_detect(nonrandom_processed_star, scale):
    star = nonrandom_processed_star
    print("Testing injections on ", star.star_id)
    center = sc.cutils.misc.get_stamp_center(star.cat.iloc[0]['stamp'])
    pos = sc.np.array((-2, -1))
    row = star.cat.iloc[1]
    results = star.row_inject_subtract_detect(
        row,
        pos,
        contrast=scale,
    )
    snr, is_detected = results
    if (snr >= star.det_args['snr_thresh']):
        assert(is_detected)
    elif (snr < star.det_args['snr_thresh']):
        assert(not is_detected)

def test_set_subtr_parameters(nonrandom_processed_star):
    star = nonrandom_processed_star
    assert(hasattr(star, 'subtr_args'))
