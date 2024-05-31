"""
This module contains code used to define what is and isn't a detection
"""

from pathlib import Path
import pandas as pd
import numpy as np

import itertools

from scipy import stats

from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from importlib import reload

import public_wifi as pw


from public_wifi.utils import table_utils
from public_wifi.utils import shared_utils
from public_wifi.utils import initialize_tables

import matplotlib as mpl
from matplotlib import pyplot as plt

######################
# SNR and detections #
######################

def mask_nans(arr : np.ndarray):
    """Return an array with nans masked"""
    wherenan = np.where(np.isnan(arr))
    mask = np.zeros_like(arr, dtype=int)
    mask[wherenan] = 1
    return np.ma.masked_array(arr, mask=mask)

# def normalize_stamp(stamp):
#     """
#     Normalize the stmap to the unit normal distribution.
#     This only works with KLIP residuals, which are whitened.
#     """
#     # shift to mean 0
#     normed_stamp = stamp - np.nanmean(stamp)
#     # scale to std = 1
#     normed_stamp = normed_stamp/np.nanstd(normed_stamp)
#     return normed_stamp

def normalize_stamps(residuals, sigma_clip=3):
    """
    Normalize a stamp using sigma-clipping such that the mean is 0 and the
    standard deviation is 1

    Parameters
    ----------
    residuals : pd.DataFrame
      dataframe of KLIP-subtracted stamps, where the column is the Kklip and
      the index is the parent stamp ID
    sigma_clip : float = 3:
      threshold argument to provide to sigma_clipped_stats
    """
    stats = residuals.map(
        lambda el: pd.Series(
            sigma_clipped_stats(el, sigma=sigma_clip, maxiters=20),
            index=['mean', 'median', 'std']
        ),
        na_action='ignore'
    )
    mean = stats.map(lambda el: el['mean'], na_action='ignore')
    std =  stats.map(lambda el: el['std'], na_action='ignore')
    return (residuals - mean) / std

def compute_noise_map(
        subtr_results,
        mode : str = 'pixel',
        clip_thresh : float = 3.0,
        normalize : bool = True
):
    """
    Create an STD map of the subtraction residuals
    Parameters
    ----------
    subtr_results : named tuple of PSF subtraction results
      has elements: residuals, references, models
    mode : str = 'pixel'
      alternatively, 'stamp'
    clip_thresh : float = 3.0
      sigma threshold for sigma clipping algorithm
    normalize: bool = True
      transform residual stamps to N(0, 1)

    Output
    ------
    std_map : pd.DataFrame, same form as subtr_results.residuals,
      with the standard deviations
    """
    resids = subtr_results.residuals.copy()
    refs = subtr_results.references.copy()
    if normalize:
        resids = normalize_stamps(resids)
    if mode == 'pixel':
        # # compute the SNR map by comparing the target stamp pixel to that same
        # # pixel in the residual stamps of all the other stars' residuals
        # def calc_pixelwise_std(row):
        #     # given a target row, make an SNR map for each KL mode if the residuals
        #     name = row.name
        #     ref_stamp_ids = subtr_results.references.loc[name].dropna()
        #     refs_resid = subtr_results.residuals.query("stamp_id in @ref_stamp_ids")
        #     # refs_resid = resids.loc[resids.index != name].dropna(axis=1, how='all')
        #     refs_resid = refs_resid.dropna(axis=1, how='all')
        #     refs_resid_std = refs_resid.apply(lambda col: [sigma_clipped_stats(np.stack(col.dropna()), axis=0)[-1]]).squeeze()
        #     return refs_resid_std
        # std_map = resids.apply(lambda row: calc_pixelwise_std(row), axis=1)
        noise_maps = resids.apply(
            lambda col: [sigma_clipped_stats(np.stack(col.dropna()), sigma=clip_thresh, axis=0)[-1]]
        ).squeeze()
        # expand to a dataframe
        noise_maps = pd.DataFrame.from_dict(
            {i: noise_maps for i in resids.index},
            orient='index'
        )
    elif mode == 'stamp':
        # calculate the standard deviation stampwise
        noise_maps = subtr_results.residuals.map(
            lambda stamp: sigma_clipped_stats(stamp, sigma=clip_thresh)[-1],
            na_action='ignore'
        )
    else:
        print("mode not recognized, please choose `pixel` or `stamp`")
        noise_maps = None
    return noise_maps



def create_snr_map(
        subtr_results,
        mode : str = 'pixel'
) -> pd.DataFrame:
    """
    Create and SNR map for each stamp. The SNR can be computed for each pixel
    against its stamp or against all the other pixels in its position.

    Pseudocode:
    For each stamps's residuals:
      1. Get the residuals for all its reference stamps
      2. Aggregate the residual stamps into a noise image for each Kklip
      3. Divide the target stamps by the noise stamps, aligned by Kklip

    Parameters
    ----------
    subtr_results : named tuple
      Created by the subtraction_manager class. Has three properties:
      - residuals : dataframe of PSF subtraction residuals. Index = stamp ID,
        col = Kklip
      - references : dataframe of which are OK to use as references. Index =
        stamp ID, col = stamp_id.
      - models : dataframe of PSF models for each stamp constructed from the
        references. Index = stamp ID, col = Kklip
    mode : str = 'pixel'
      should the SNR be calculated against the stamp or the other pixels in the
      same position

    Output
    ------
    snr_df : pd.DataFrame
      DataFrame of SNR maps. Index = stamp ID, column = Kklip

    """
    residual_table = subtr_results.residuals
    reference_table = subtr_results.references
    # normalize the residuals table so all the stamps have mean 0 and sigma 1
    if mode == 'pixel':
        normed_residuals = normalize_stamps(residual_table)
        snr_maps = normed_residuals.apply(
            lambda target_row: compute_snr_target_pixel(
                target_row, normed_residuals, reference_table
            ),
            axis=1
        )
    else:
        # mode == 'stamp':
        snr_maps = subtr_results.residuals.map(
            lambda arr: arr/sigma_clipped_stats(mask_nans(arr))[0],
            na_action='ignore'
        )
    return snr_maps


### pixelwise SNR code
def compute_snr_target_pixel(
        target_row : pd.Series,
        residual_table : pd.DataFrame,
        reference_table : pd.DataFrame
) -> pd.Series :
    """
    Compute the SNR images for a target stamp for each Kklip.
    First, use the references table to select the appropriate residual stamps to use. 
    Then, compute a noise image per kklip from the residuals.
    Then, divide the target images per kklip by the noise images per kklip

    Parameters
    ----------
    target_row : pd.Series
      A set of residual images for a target stamp, indexed by Kklip
    residual_table : pd.DataFrame
      A dataframe of PSF subtraction residuals, where the column is the Kklip
      and the index is the (star ID, stamp_ID)
    reference_table : pd.DataFrame
      A table of references, where the index is the target stamp and the column
      is the reference stamp. If a stamp is OK to use as a reference, the cell
      contains the stamp ID. If the stamp is not OK for a reference, the cell
      contains NaN.

    Output
    ------
    target_snr_maps : pd.Series
      SNR map of the target stamp for each Kklip

    """
    reference_ids = list(reference_table.loc[target_row.name].dropna().values)
    # filter down to the allowed references, and remove Kklip columns of all NaN
    reference_stamps = residual_table.query(f"stamp_id in {reference_ids}")
    reference_stamps = reference_stamps.dropna(axis=1, how='all')
    std_maps = reference_stamps.apply(
        lambda col: [np.nanstd(np.stack(col.dropna()), axis=0)]
    ).squeeze()
    # std_maps.name = target_row.name
    snr_maps = target_row/std_maps
    snr_maps.name = target_row.name
    snr_maps.index.name = 'kklip'
    # clean up all-nan arrays - set to nan
    where_all_nan = snr_maps.apply(lambda arr: np.all(np.isnan(arr)))
    snr_maps[where_all_nan] = np.nan
    return snr_maps



def load_snr_maps(ana_mgr):
    snr_maps = ana_mgr.results_dict['snr'].copy()
    return snr_maps


def apply_snr_threshold_by_stamp(stamp_snr_maps, threshold=5):
    """
    Apply an SNR threshold to each stamps' set of SNR maps per KKLip, and
    return the pixel positions above that threshold.
    """
    candidates = stamp_snr_maps.map(
        lambda stamp: np.where(stamp >= threshold) if np.any(np.where(stamp >= threshold)) == True else np.nan,
        na_action='ignore'
    )
    # massage the data to avoid automatic pandas broadcasting, which we don't want here
    candidates = candidates.dropna().map(lambda el: np.squeeze(el).T).to_dict()
    return candidates


# def get_candidates(snr_df, threshold=5):
#     """
#     Return a list of all the candidate pixels, organized by stamp
#     """
#     candidates = snr_df.apply(apply_snr_threshold_by_stamp, axis=1, args=[threshold])
#     # filter out stamps with no candidates
#     candidates = candidates[candidates.map(len) > 0].copy()
#     return candidates
def get_candidates(
        snr_df : pd.DataFrame,
        thresh : float = 5.0,
) -> pd.Series :
    """
    Return a list of all the candidate pixels, organized by stamp, and the number of KKlips that have a detection
    """
    # number of pixels over the threshold
    npix_over_thresh = snr_df.map(
        lambda img: len(img[img >= thresh])
    )
    # the number of KLIP modes with at least one pixel over the threshold
    nklips_over_thresh = npix_over_thresh.apply(
        lambda row: row[row > 0].size,
        axis=1
    )
    candidate_stamps = nklips_over_thresh[nklips_over_thresh > 0].sort_values(ascending=False)

    candidate_pix = snr_df.loc[candidate_stamps.index].apply(
        lambda row: set(
            itertools.chain(
                *list(row.map(lambda img: [tuple(i) for i in np.stack(np.where(img > thresh)).T]))
            )
        ),
        axis=1
    )
    num_pix = candidate_pix.apply(len)
    df = pd.concat(
        {'num_modes': candidate_stamps, 'num_pix': num_pix, 'pix': candidate_pix},
        axis=1
    )
    return df



### Normality tests ###
def normality_test(
        data : np.ndarray,
) -> np.ndarray :
    """
    Generate a normal probability plot as described here: 
    https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm
    

    Parameters
    ----------
    data : np.ndarray
      1-d array of data to test for normality

    Output
    ------
    Define your output

    """
    n = data.size
    order_statistics = (np.arange(1, n+1) - 0.3175)/(n+0.365)
    order_statistics[-1] = 0.5**(1./n)
    order_statistics[0] = 1 - order_statistics[-1]
    return order_statistics
