"""
These tests just check that the input and output of each method is what you
expect; they do not check that the algorithms are correct
Only methods that return something are tested here
"""
import pytest
import numpy as np
from public_wifi import starclass as sc

DEBUG = False # use to turn on debug printing


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


def test_star_generate_match_query(star):
    for i, row in star.cat.iterrows():
        query = star.generate_match_query(row)
        assert(isinstance(query, str))


def test_starclass_get_cutout(star, data_folder):
    print("Getting stamp from " + star.star_id)
    assert(data_folder.exists())
    stamp_size = 15
    cutouts = star.cat.apply(lambda row: star._get_cutout(row, stamp_size), axis=1)
    # check that it's a cutout object
    assert(all(cutouts.apply(lambda el: isinstance(el, sc.Cutout2D))))
    # check that it has the right shape
    cutout_size = stamp_size + star._cutout_pad*2
    print(cutout_size)
    assert(all(cutouts.apply(lambda el: el.shape == (cutout_size, cutout_size))))

def test_measure_bgnd(star) :
    bgnd = star.measure_bgnd()
    assert(isinstance(bgnd, sc.pd.Series))
    assert(bgnd.apply(lambda el: type(el) == tuple).all())

def test_set_references(all_stars):
    star = all_stars.iloc[0]
    star.set_references(all_stars)
    assert(hasattr(star, "references"))
    assert(isinstance(star.references, sc.pd.DataFrame))

def test_row_get_references(random_processed_star):
    star = random_processed_star
    for i, row in star.cat.iterrows():
        ref_rows = star._row_get_references(row)
        assert(isinstance(ref_rows, sc.pd.DataFrame))

# def test_scale_stamp(star):
#     scaled_stamps = star.cat['stamp'].apply(star.scale_stamp)
#     for stamp in scaled_stamps:
#         assert(sc.np.nanmin(stamp) < 1e-10)
#         assert(sc.np.abs(sc.np.nanmax(stamp) - 1) < 1e-10)


def test_row_klip_subtract(all_stars):
    star_id = sc.np.random.choice(all_stars.index)
    print("KLIP subtraction tested on ", star_id)
    star = all_stars.loc[star_id]
    star.set_references(all_stars, compute_similarity=True)
    row = star.cat.iloc[0]
    subtraction = star._row_klip_subtract(row, 0.5, 5)
    assert(isinstance(subtraction, sc.pd.Series))
    columns = ['klip_model', 'klip_basis', 'klip_sub']
    assert(all([(c in subtraction.index) for c in columns]))

def test_run_klip_subtraction(all_stars):
    star_id = sc.np.random.choice(all_stars.index)
    print("KLIP subtraction tested on ", star_id)
    star = all_stars.loc[star_id]
    star.set_references(all_stars, compute_similarity=True)
    # run with default parameters
    subtraction = star.run_klip_subtraction(sim_thresh = 0.5, min_nref = 5)
    assert(isinstance(subtraction, sc.pd.DataFrame))
    columns = ['klip_model', 'klip_basis', 'klip_sub']
    assert(all([(c in subtraction.columns) for c in columns]))

def test_run_make_snr_maps(random_processed_star):
    """
    Make sure self.run_make_snr_maps() properly assigns the SNR maps to the
    results dataframe
    """
    star = random_processed_star
    assert(hasattr(star, 'results'))
    assert(hasattr(star, 'run_make_snr_maps'))
    snrmaps = star.run_make_snr_maps()
    # assert('snrmap' in star.results.columns)
    assert(isinstance(snrmaps, sc.pd.Series))
    assert(snrmaps.name == 'snrmap')

def test_apply_matched_filter(random_processed_star):
    star = random_processed_star
    result = star.apply_matched_filter(
        contrast=True, throughput_correction=False
    )
    assert(isinstance(result, sc.pd.Series))
    assert(len(result) == len(star.cat))
    assert(result.name == 'detmap')
    
### Functions that operate on a row of the results dataframe and return a
### column should all conform to this format
@pytest.mark.parametrize(
    'func_name,result_name',
    [
        ('_row_make_snr_map', 'snrmap'), # SNR map
        ('_row_apply_matched_filter', 'detmap'), # Source det. w PSF convolution
        ('_row_make_mf_flux_map', 'fluxmap'), # MF flux calibration
    ]
)
def test_row_result_func(random_processed_star, func_name, result_name):
    star = random_processed_star
    row = star.results.loc[1]
    func = getattr(star, func_name)
    result = func(row)
    assert(isinstance(result, sc.pd.Series))
    assert(len(result) == 1)
    assert(result.index[0] == result_name)
    assert(result[result_name].index.name == 'numbasis')

### Functions that aggregate row operations
### not implemented

# @pytest.mark.xfail
def test_jackknife_drop_ref(all_stars):
    star = np.random.choice(all_stars)
    star.set_references(all_stars)
    star.subtr_args=dict(sim_thresh=0.8, min_nref=5)
    # toss the reference with the highest sim score to ensure it will have the
    # "used" flag
    ref = star.references.groupby("target")['sim'].sum().idxmax()
    # for one reference, do the jackknife test
    klsub = star.run_klip_subtraction(jackknife_reference=ref)['klip_sub']
    # get the number of jackknife reductions
    n_jackknife = klsub.apply(len).sum()
    # get the number of references
    n_refs = star.nrefs['Nrefs'].sum()
    # Kklip_max = n_refs - 1, and jackknife removes another reference
    # so there should be 2 fewer references for each reduction
    assert((n_refs - n_jackknife) == 2*len(star.cat))
    # star_jackknife = star.jackknife_analysis()

def test_jackknife_analysis(all_stars):
    star = np.random.choice(all_stars)
    star.set_references(all_stars)
    star.subtr_args=dict(sim_thresh=0.8, min_nref=10)
    # before doing jackknife, you have to do subtraction first
    # really the purpose is to set the "used" flag on the references
    star.results = star.results.join(star.run_klip_subtraction())
    jackknife = star.jackknife_analysis()
    # this should be a dataframe with a single column
    jackknife = jackknife.squeeze()
    # print(jackknife.loc[1])
    assert(isinstance(jackknife, sc.pd.Series))

# @pytest.mark.parametrize('scale', list(range(1, 21)))
# def test_row_inject_psf(nonrandom_processed_star, scale):
#     star = nonrandom_processed_star
#     row = star.cat.iloc[1]
#     # scale = 10
#     # inj_row = cutils.row_inject_psf(row, star, (0, 0), scale, -1)
#     inj_row = star.row_inject_psf(row, (0, 0), scale, -1)
#     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
#     # axes[0].imshow(row['stamp'])
#     # axes[1].imshow(inj_row['stamp'])
#     # plt.show()
#     inj_flux = sc.cutils.measure_primary_flux(inj_row['stamp'], row['stamp'])
#     stamp_flux = sc.cutils.measure_primary_flux(row['stamp'], row['stamp'])
#     flux_ratio = inj_flux/stamp_flux
#     print(inj_flux, stamp_flux, scale, flux_ratio)
#     # let's give ourselves a 5% margin
#     assert(sc.np.abs(flux_ratio/(scale+1) - 1) <= 0.05)

# @pytest.mark.parametrize('scale', list(range(1, 21)))
# def test_inject_subtract_detect(nonrandom_processed_star, scale):
#     star = nonrandom_processed_star
#     print("Testing injections on ", star.star_id)
#     center = sc.cutils.misc.get_stamp_center(star.cat.iloc[0]['stamp'])
#     pos = sc.np.array((-2, -1))
#     row = star.cat.iloc[1]
#     results = star.row_inject_subtract_detect(
#         row,
#         pos,
#         contrast=scale,
#     )
#     snr, is_detected = results
#     if (snr >= star.det_args['snr_thresh']):
#         assert(is_detected)
#     elif (snr < star.det_args['snr_thresh']):
#         assert(not is_detected)

def test_set_subtr_parameters(nonrandom_processed_star):
    star = nonrandom_processed_star
    assert(hasattr(star, 'subtr_args'))

def test_run_make_mf_flux_map(nonrandom_processed_star):
    star = nonrandom_processed_star
    fluxmaps = star.run_make_mf_flux_maps(contrast=True)
    assert(isinstance(fluxmaps, sc.pd.Series))
    assert(fluxmaps.name == 'fluxmap')
    assert(len(fluxmaps) == len(star.results))
    assert(all([isinstance(row, sc.pd.Series) for row in fluxmaps]))

