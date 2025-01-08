import numpy as np
import pandas as pd

from scipy.signal import correlate
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
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
        mf_width : int = 7,
        correlate_mode='same',
        throughput_correction : bool = False,
        kl_basis : np.ndarray | pd.Series | None = None,
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
    throughput_correction : bool = False
      If True, correct for the matched filter response (should always be true)
    kl_basis : np.ndarray | pd.Series | None = None
      If provided, include the KLIP basis in the throughput correction
      It must be only the KLIP basis up to the Kklip of the PSF model
    """
    matched_filter = make_matched_filter(psf_model, mf_width)
    detmap = correlate(
        target_stamp,
        matched_filter,
        method='direct',
        mode=correlate_mode)
    if throughput_correction:
        throughput = compute_throughput(matched_filter, klmodes=kl_basis)
        if isinstance(throughput, pd.Series):
            throughput = throughput.iloc[-1]
        detmap = detmap / throughput
    return detmap

def compute_throughput(mf, klmodes=None) -> float | pd.Series:
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
    throughput_map : pd.Series
      A series of 2-D array throughput maps, indexed by KLIP mode
    """
    # the first term in the throughput is the amplitude of the MF
    throughput = apply_matched_filter(
        mf, mf,
        correlate_mode='valid',
        throughput_correction=False,
        kl_basis=None
    )[0, 0] # this indexing works because correlate_mode is 'valid'
    # the second term is the amount of the MF captured by the KL basis at each
    # position
    if klmodes is not None:
        # format kl modes as a series
        if not isinstance(klmodes, pd.Series):
            klmodes = pd.Series({i+1: mode for i, mode in enumerate(klmodes)})
        mf_adjust = klmodes.apply(
            lambda klmode: apply_matched_filter(
                klmode,
                mf,
                correlate_mode='same',
                throughput_correction=False
            )**2
        )
        mf_adjust = mf_adjust.cumsum()
        throughput = throughput - mf_adjust
    return throughput
