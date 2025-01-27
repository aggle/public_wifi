import pytest
import numpy as np
from matplotlib import pyplot as plt

from public_wifi import subtraction_utils as sub_utils



def test_klip_subtract(stars_with_references):
    """Test that klip_subtract gives you back a dataframe with three columns"""
    star = np.random.choice(stars_with_references)
    print(f"Star: {star.star_id}")
    row = star.cat.loc[1]
    reference_rows = star._row_get_references(row, **star.subtr_args)
    star.row_set_reference_status(row, reference_rows)
    reference_stamps = reference_rows['stamp']
    results = sub_utils.klip_subtract(
        row['stamp'],
        reference_stamps,
        numbasis=np.arange(1, reference_stamps.size+1),
    )
    print(results)
    assert(isinstance(results, sub_utils.pd.DataFrame))



def test_measure_primary_flux(random_processed_star):
    """Check that you are measuring the correct flux of a star"""
    # construct a PSF with known flux
    star = random_processed_star
    print("Testing injections on ", star.star_id)
    row = star.cat.loc[0]
    stamp = row['stamp']
    input_flux = 10
    psf = cutils.mf_utils.make_normalized_psf(stamp, scale=input_flux)
    # make sure the psf was set correctly
    assert(np.abs(input_flux - psf.sum()) <= 1e-5)
    # measure the flux
    meas_flux = cutils.measure_primary_flux(psf, psf.copy())
    print("measured:", meas_flux)
    assert(np.abs(meas_flux - input_flux) <= 1e-5)

