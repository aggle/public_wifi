import numpy as np
import pandas as pd

from scipy.signal import correlate
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.modeling.models import Gaussian2D

from public_wifi import misc

def make_normalized_psf(
        psf_stamp : np.ndarray,
        scale : float = 1.,
        center : bool = False,
) -> np.ndarray:
    """
    Normalize a PSF to have an arbitrary flux
    psf_stamp : np.ndarray
      2-D psf
    scale : float = 1.
      Scale the PSF so that the total flux has this value
    center : bool = True
      Shift the PSF to the center of the array. Must be done before flux normalization
    """
    if center:
        psf_stamp = misc.shift_stamp_to_center(psf_stamp)
    # set min to 0 and normalized
    norm_psf = psf_stamp - np.nanmin(psf_stamp)
    norm_psf /= np.nansum(norm_psf)
    # scale to arbitrary value
    norm_psf *= scale
    return norm_psf

def make_matched_filter(stamp, width : int | None = None):
    # take in an arbitrary PSF stamp and turn it into a matched filter
    stamp = stamp.copy()
    if isinstance(width, int) and (width < min(stamp.shape)):
        borders = (np.array(stamp.shape) - width)/2
        borders = borders.astype(int)
        stamp = stamp[borders[0]:-borders[0], borders[1]:-borders[1]]

    normalized_stamp = make_normalized_psf(stamp)
    normalized_stamp -= np.nanmean(normalized_stamp)
    return normalized_stamp

def apply_matched_filter(
        target_stamp : np.ndarray,
        psf_model : np.ndarray,
        mf_width : int = 7,
        correlate_mode='same',
        throughput_correction : bool = False,
        correct_pca_throughput : bool = False,
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
    matched_filter = make_matched_filter(psf_model, width=mf_width)
    detmap = correlate(
        target_stamp,
        matched_filter,
        method='direct',
        mode=correlate_mode)
    # if throughput_correction:
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
    # throughput = apply_matched_filter(
    #     mf, mf,
    #     correlate_mode='valid',
    #     throughput_correction=False,
    #     kl_basis=None
    # )[0, 0] # this indexing works because correlate_mode is 'valid'
    norm = compute_mf_norm(mf)
    # the second term is the amount of the MF captured by the KL basis at each
    # position
    bias = 0
    if klmodes is not None:
        # format kl modes as a series
        bias = compute_pca_bias(mf, klmodes)
    throughput = norm - bias
    return throughput

def compute_mf_norm(
        mf
) -> float :
    """
    Compute the norm of the matched filter

    Parameters
    ----------
    mf : np.ndarray
      the matched filter

    Output
    ------
    norm : float
      The 2-norm of the matched filter

    """
    norm = np.dot(mf.ravel(), mf.ravel())
    return norm

def compute_pca_bias(
        mf : np.ndarray,
        klip_modes : np.ndarray | pd.Series,
) -> pd.Series :
    """
    Compute the bias introduced by sub-optimal PSF modeling

    Parameters
    ----------
    mf : np.ndarray
      A matched filter in the shape of the PSF
    klip_modes : np.ndarray | pd.Series
      The KLIP modes used to construct the model PSF

    Output
    ------
    bias : pd.Series
      A pixel map of the bias, as a cumulative sum

    """
    if not isinstance(klip_modes, pd.Series):
        klip_modes = pd.Series({i+1: mode for i, mode in enumerate(klip_modes)})
    bias = klip_modes.apply(
        lambda klmode: apply_matched_filter(
            klmode,
            mf,
            correlate_mode='same',
            throughput_correction=False
        )**2
    )
    bias = bias.cumsum()
    return bias


def make_gaussian_psf(stamp_size, filt='F850LP') -> np.ndarray:
    xy = np.indices((stamp_size, stamp_size))
    xy = xy - misc.get_stamp_center(stamp_size)[::-1, None, None]
    fwhm2sig = lambda fwhm: fwhm/(2*np.sqrt(2*np.log(2)))
    fwhm_dict = {'F814W': 1.9, 'F850LP': 1.9}
    fwhm = fwhm_dict[filt] # pixels
    sig = fwhm2sig(fwhm)
    g2d_func = Gaussian2D(1, 0, 0, sig, sig)
    psf = g2d_func(xy[0], xy[1])
    return psf
