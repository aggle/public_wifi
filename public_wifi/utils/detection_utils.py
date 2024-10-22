"""
This module contains code used to define what is and isn't a detection
"""

from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

import itertools

from scipy import stats
from scipy.signal import convolve2d
from scipy.signal import correlate

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy import nddata

from importlib import reload

import public_wifi as pw


from public_wifi.utils import table_utils
from public_wifi.utils import shared_utils
from public_wifi.utils import initialize_tables
from public_wifi.utils import matrixDFT

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

### Photometric conversion
def get_photflam(
        stamp_id,
        db
) -> float:
    """
    Get the photometric conversion factor to scale from e/s to flux units
    """
    return 0.0


def cut_psf(
        psf_stamp : np.ndarray,
        width : int = 4,
        normalize_flux : bool = True
) -> np.ndarray:
    """
    Given a stamp of the PSF model, cut out the central region and (optional)
    normalize it to norm=1
    """
    row, col = np.unravel_index(np.argmax(psf_stamp), psf_stamp.shape)
    psf = psf_stamp[row-width:row+width+1, col-width:col+width+1].copy()
    psf = psf - psf.min()
    if normalize_flux:
        # psf = psf/np.linalg.norm(psf)
        psf = psf/np.nansum(psf)
    return psf

def make_matched_filter_from_stamp(
        stamp : np.ndarray,
        width : int | None = None,
        subtract_mean : bool = True,
) -> np.ndarray :
    """
    Generate a matched filter for use with scipy.signal.correlate by
    cutting out a PSF at the desired width and setting the scale factor
    appropriately.

    Parameters
    ----------
    stamp : np.ndarray
      a PSF instance
    width : int | None = None
      the final matched filter will have shape 2*width+1
      if None, use the whole stamp (for example, if you have already cut out the PSF)

    Output
    ------
    mf : np.ndarray
      a matched filter scaled appropriately to give back calibrated flux

    """
    if isinstance(width, int):
        mf = cut_psf(stamp, width, normalize_flux=True)
    else:
        mf = stamp.copy()
    # set it to all be positive
    mf = mf - mf.min()
    # scale so the sum is 1
    mf = mf/mf.sum()
    # a proper matched filter should be mean-subtracted
    if subtract_mean:
        mf = mf - mf.mean()
    return mf
 

def inject_psf(
        stamp : np.ndarray,
        psf : np.ndarray,
        position : tuple[int, int],
        flux : float = 1.,
) -> np.ndarray:
    """
    Inject the PSF to the stamp at the given position and scale such that the
    psf.sum = flux Assumes PSF is odd.

    Returns a stamp with the PSF added, unless there's an error in which case
    no PSF is added

    Parameters
    ----------
    stamp : np.ndarray
      The image in which you would like to add the PSF
    psf : np.ndarray
      A model of the psf. It will be adjusted such that the sum is equal to the
      flux argument, in its native units.
    position : tuple[int, int]
      The (row, col) aka (y, x) position in the stamp where the center of the
      PSF will go. For now, must be an integer.
    flux : float [1.]
      Scale the PSF image such that psf.sum() = flux
      If None, no normalization or scaling is performed

    Output
    ------
    injected_stamp : np.ndarray
    """

    injected_stamp = stamp.copy()
    psf = make_matched_filter_from_stamp(psf, width=None, subtract_mean=False)
    psf_scaled = psf * flux
    psf_shape = np.asarray(psf.shape)
    psf_halfwidth = np.floor(psf_shape/2).astype(int) 
    psf_corner = np.asarray(position)[::-1] - psf_halfwidth

    # compute the x and y injection regions
    xlim = [psf_corner[0], psf_corner[0]+psf.shape[0]]
    ylim = [psf_corner[1], psf_corner[1]+psf.shape[1]]

    # handle corner cases
    # the x-axis goes negative
    if xlim[0] < 0:
        # keep the right part of the psf
        overlap = -1*xlim[0]
        psf_scaled = psf_scaled[:, overlap:]
        xlim[0] = np.max([0, xlim[0]])
    # the x-axis goes out of range
    if xlim[1] > stamp.shape[1]:
        # keep the left part of the psf
        overlap = stamp.shape[1] - xlim[1]
        psf_scaled = psf_scaled[:, :overlap]
        xlim[1] = np.min([xlim[1], stamp.shape[1]])
    # the y-axis goes negative
    if ylim[0] < 0:
        # keep the top part of the psf
        overlap = -1*ylim[0]
        psf_scaled = psf_scaled[overlap:, :]
        ylim[0] = np.max([0, ylim[0]])
    # the y-axis goes out of range
    if ylim[1] > stamp.shape[0]:
        # keep the bottom part of the psf
        overlap = stamp.shape[0] - ylim[1]
        psf_scaled = psf_scaled[:overlap, :]
        ylim[1] = np.min([ylim[1], stamp.shape[0]])    
    try:
        injected_stamp[ylim[0]:ylim[1], xlim[0]:xlim[1]] += psf_scaled
    except ValueError:
        print("Error: No PSF added. Position is out of bounds for PSF and stamp sizes.")
    return injected_stamp


def apply_matched_filter(
        data : np.ndarray,
        psf : np.ndarray,
        correct_throughput : bool = True,
        klmodes : pd.Series | None = None
) -> np.ndarray:
    """
    apply matched filter detection using PSF models `psf` to the data `data`
    method : "convolve", "correlate" (default), "fft"
    returns an array of shape `data` that has had the matched filter applied.
    if KL modes are also supplied, it gives the flux map

    Parameters
    ----------
    data : np.ndarray,
      data possibly containing a signal
    psf : np.ndarray,
      the signal to look for
    method : _MF_METHODS = 'correlate',
      deprecated - which matched filtering method to use
    correct_throughput : bool = True
      whether or not to correct for throughput
      Normally this should be True. Set it to false when you are correlating
      with the KL modes to compute the throughput itself.
    klmodes : pd.Series | None = None
      any KLIP modes 
    """
    # prepare the matched filter
    matched_filter = make_matched_filter_from_stamp(psf - psf.min(), width=None)

    # convolve the matched filter with the data to get the signal response map
    det_map = correlate(data, matched_filter, method='direct', mode='same')
    # # use astropy.nddata.utils.Cutout2D to get back the cutout
    # uncomment this, but it takes extra time
    # det_map = correlate(data, matched_filter, method='direct', mode='full')
    # center = np.floor(np.array(det_map.shape)/2)
    # det_map = nddata.Cutout2D(det_map, center[::-1], data.shape[0]).data 
    # scale the counts to match the flux contained in the matched filter
    # klmodes can be none
    if correct_throughput:
        throughput = compute_throughput(matched_filter, klmodes)
    else:
        throughput = 1.
    det_map = det_map/throughput
    return det_map


def compute_throughput(mf, klmodes=None) -> float | np.ndarray[float]:
    """
    Make a throughput map for flux calibration

    Parameters
    ----------
    mf : np.ndarray
      The matched filter. We will compute the correlation with the KL modes to
      get the throughput.
    klmodes : pd.Series
      a pandas Series of the KL modes, reshaped into 2-D images

    Output
    ------
    throughput_map : np.ndarray
      A 2-D array, the same shape as the image, containing the throughput
      correction to correct the detection map into PSF fluxes
    """
    # mf_norm = np.dot(mf.ravel(), mf.ravel())
    # if klmodes is None:
    #     # this does not take into account when part of the flux is out of the stamp
    #     # this should not have a big effect in the middle
    #     # the effect is still small (~1%) out to one pixel in from the stamp edge
    #     throughput = mf_norm
    # else:
    #     # format kl modes as a series
    #     if not isinstance(klmodes, pd.Series):
    #         klmodes = pd.Series({i+1: mode for i, mode in enumerate(klmodes)})
    #     mf_adjust = klmodes.apply(
    #         lambda mode: apply_matched_filter(mf, mode, correct_throughput=False)**2
    #     )
    #     mf_adjust = np.sum(np.stack(mf_adjust), axis=0)
    #     throughput = mf_norm - mf_adjust
    throughput = np.dot(mf.ravel(), mf.ravel())
    if klmodes is not None:
        # format kl modes as a series
        if not isinstance(klmodes, pd.Series):
            klmodes = pd.Series({i+1: mode for i, mode in enumerate(klmodes)})
        mf_adjust = klmodes.apply(
            lambda mode: apply_matched_filter(mf, mode, correct_throughput=False)**2
        )
        mf_adjust = np.sum(np.stack(mf_adjust), axis=0)
        throughput = throughput - mf_adjust
    return throughput


