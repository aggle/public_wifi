import numpy as np
import pandas as pd

from scipy.signal import correlate
from astropy.convolution import convolve

from astropy.nddata import Cutout2D


def make_normalized_psf(
        psf_stamp : np.ndarray,
        width : int | None = None,
        scale : float = 1.,
):
    """
    normalize a PSF to have flux 1
    width : if given, the PSF will have shape (width, width)
    scale : float = 1.
      Scale the PSF so that the total flux has this value
    """
    if isinstance(width, int) and (width < min(psf_stamp.shape)):
        borders = (np.array(psf_stamp.shape) - width)/2
        borders = borders.astype(int)
        psf_stamp = psf_stamp[borders[0]:-borders[0], borders[1]:-borders[1]]

    # set min to 0 and normalized
    norm_psf = psf_stamp - np.nanmin(psf_stamp)
    norm_psf /= np.nansum(norm_psf)
    # scale to arbitrary value
    norm_psf *= scale
    return norm_psf

def make_matched_filter(stamp, width : int | None = None):
    # take in an arbitrary PSF stamp and turn it into a matched filter
    stamp = stamp.copy()
    normalized_stamp = make_normalized_psf(stamp, width)
    normalized_stamp -= np.nanmean(normalized_stamp)
    return normalized_stamp

def apply_matched_filter(
        target_stamp : np.ndarray,
        psf_model : np.ndarray,
        correlate_mode='same',
) -> np.ndarray:
    """
    Apply the matched filter as a correlation. Normalize by the matched filter norm.
    target_stamp : np.ndarray
      the stamp in which you are looking for signal
    psf_model  : np.ndarray
      the 2-D psf model
    correlate_mode : str
      'same' or 'valid'. use 'same' for searches, and 'valid' if you have an
      unsubtracted psf and want the flux
    """
    matched_filter = make_matched_filter(psf_model)
    detmap = correlate(
        target_stamp,
        matched_filter,
        method='direct',
        mode=correlate_mode)
    detmap = detmap / np.linalg.norm(matched_filter)**2
    # detmap = convolve(target_stamp, psf_model-psf_model.min(), normalize_kernel=True)
    return detmap



def detect_snrmap(snrmaps, snr_thresh=5, n_modes=3) -> pd.Series:
    """
    Detect sources using the SNR maps
    snrmaps : pd.Series
      A pd.Series where each entry is the SNR map for a Kklip mode
    thresh : float
      the SNR threshold
    n_modes : int
      the threshold on the number of modes in which a candidate must be detected

    A detection is a pixel that is over the threshold in at least three modes
    """
    # drop the first mode; it's always garbage
    stack = np.stack(snrmaps[1:])
    center_pixel = np.floor(((np.array(stack.shape[-2:])-1)/2)).astype(int)
    # get all the pixels over threshold
    initial_candidates = pd.DataFrame(
        np.where(stack >= snr_thresh),
        index=['kklip','dy','dx']
    ).T
    initial_candidates['dy'] -= center_pixel[1]
    initial_candidates['dx'] -= center_pixel[0]
    # no candidates? Quit early
    if len(initial_candidates) == 0:
        return None
    else:
        # drop the central pixel
        central_pixel_filter = initial_candidates[['dy','dx']].apply(
            lambda row: all(row.values == (0, 0)) == False,
            axis=1
        ) 
        initial_candidates = initial_candidates[central_pixel_filter].copy()
        # group by row, col and find the ones that appear more than n_modes
        candidate_filter = initial_candidates.groupby(['dy', 'dx']).size() >= n_modes
        candidates = candidate_filter[candidate_filter].index.to_frame().reset_index(drop=True)
        return candidates[['dx','dy']].apply(tuple, axis=1)
