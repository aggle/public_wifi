import numpy as np
import pandas as pd

from scipy.signal import correlate
from astropy.convolution import convolve
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

def make_matched_filter(
        stamp : np.ndarray,
        width : int | None = None,
        center : bool = False
):
    # take in an arbitrary PSF stamp and turn it into a matched filter
    stamp = stamp.copy()
    if isinstance(width, int) and (width < min(stamp.shape)):
        borders = (np.array(stamp.shape) - width)/2
        borders = borders.astype(int)
        stamp = stamp[borders[0]:-borders[0], borders[1]:-borders[1]]

    normalized_stamp = make_normalized_psf(stamp)
    normalized_stamp -= np.nanmean(normalized_stamp)
    return normalized_stamp

def apply_matched_filter_to_stamp(
        target_stamp : np.ndarray,
        psf_model : np.ndarray,
        mf_width : int | None = None,
        correlate_mode='same',
        kl_basis : np.ndarray | pd.Series | None = None,
        nan_center : bool = True
) -> np.ndarray:
    """
    Apply the matched filter to a single stamp. Normalize by the matched filter
    norm and the Kklip projection, if provided.

    Parameters
    ----------
    target_stamp : np.ndarray
      the stamp in which you are looking for signal
    psf_model  : np.ndarray
      the 2-D psf model
    correlate_mode : str
      'same' or 'valid'. use 'same' for searches, and 'valid' if you have an
      unsubtracted psf and want the flux
    kl_basis : np.ndarray | pd.Series | None = None
      If provided, include the KLIP basis in the throughput correction
      It should be only the KLIP basis up to the Kklip of the PSF model
    nan_center : bool = True
      if True, set the center pixel to NaN. Not useful for detection, has wild values.

    Output
    ------
    mf_map : the stamp with the matched filter applied
    """
    matched_filter = make_matched_filter(psf_model, width=mf_width)
    mf_map = correlate(
        target_stamp,
        matched_filter,
        method='direct',
        mode=correlate_mode)
    # this returns the MF norm if kl_basis is None, else a pd.Series
    throughput = compute_throughput(matched_filter, klmodes=kl_basis)
    if isinstance(throughput, pd.Series):
        throughput = throughput.iloc[-1]
        if nan_center:
            center = misc.get_stamp_center(throughput)[::-1]
            throughput[*center] = np.nan
    mf_map = mf_map / throughput
    return mf_map

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
        modes : np.ndarray | pd.Series,
        nan_center : bool = True
) -> pd.Series :
    """
    Compute the bias introduced by sub-optimal PSF modeling

    Parameters
    ----------
    mf : np.ndarray
      A matched filter in the shape of the PSF
    klip_modes : np.ndarray | pd.Series
      The KLIP modes used to construct the model PSF
    nan_center : bool = True
      if True, set the center value to NaN
    Output
    ------
    bias : pd.Series
      A pixel map of the bias, as a cumulative sum

    """
    if not isinstance(modes, pd.Series):
        modes = pd.Series({i+1: mode for i, mode in enumerate(modes)})
    center = misc.get_stamp_center(modes)[::-1]
    bias = modes.apply(
        # compute the correlation with the MF for each KL mode
        lambda klmode: correlate(
            klmode, mf, mode='same', method='direct',
        )**2
    )
    # set the center to nan
    if nan_center:
        for b in bias:
            b[*center] = np.nan
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


def apply_matched_filters_from_catalog(
        target_star,
        all_stars : pd.Series
) -> pd.DataFrame :
    """
    Use other stars' PSFs as the matched filter. This allows us to sample the
    pixel phase. Match on filter and Kklip. You also have to recompute the
    pca_bias term by convolving the new matched filter with the target star's
    klip modes.

    Parameters
    ----------
    star : starclass.Star object
    all_stars : pd.Series
      the pd.Series object with one star for each entry.

    Output
    ------
    mf_results : pd.DataFrame
      dataframe indexed by [mf_star, cat_row, numbasis] with columns [mf, detmap, pca_bias, fluxmap]
      index: mf_star is the star_id of the star providing the matched filter,
        cat_row is the catalog row, numbasis is Kklip
      columns: mf is the result of the unnormalized matched filter, detmap is
        the result of the matched filter divided by the mf norm, pca_bias is the
        square of the mf convolved with the KLIP modes, and fluxmap is the mf
        result divided by the throughput (norm - pca_bias)
    """
    # match on Kklip
    max_kklip = all_stars.apply(lambda star: star.results.index.get_level_values("numbasis").max()).min()
    target_subset = target_star.results.query(f"numbasis <= {max_kklip}")

    # match Kklip and cross-correlate the catalog MF against the target's residuals
    crossmf = all_stars.apply(
        lambda mf_star: target_subset.apply(
            lambda row: correlate(
                row['klip_sub'],
                mf_star.results.loc[row.name, 'mf'],
                mode='same', method='auto'
            ),#/mf_star.results.loc[row.name, 'mf_norm'],
        axis=1
        )
    )
    crossmf = pd.concat({i: row for i, row in crossmf.iterrows()}, names=['mf_star'])
    crossmf.name = 'mf'
    crossmf = pd.DataFrame(crossmf)

    # divide by the mf's norm
    crossmf['detmap'] = crossmf.apply(
        lambda row: row['mf'] / all_stars.loc[row.name[0]].results['mf_norm'].loc[row.name[1:]],
        axis=1
    )

    # compute the pca bias of the new matched filter against the target star's KLIP modes
    crossmf_pca_bias = all_stars.apply(
        lambda mf_star: target_subset.apply(
            lambda row: compute_pca_bias(
                mf_star.results.loc[row.name, 'mf'],
                target_subset['klip_basis'].loc[row.name[0], :row.name[1]],
                nan_center=True,
            ).iloc[-1],
            axis=1
        )
    )
    crossmf_pca_bias = pd.concat({i: row for i, row in crossmf_pca_bias.iterrows()}, names=['mf_star'])
    crossmf['pca_bias'] = crossmf_pca_bias


    # divide the mf by the throughput to get the flux
    crossmf['fluxmap'] = crossmf.apply(
        lambda row: row['mf'] / (all_stars.loc[row.name[0]].results['mf_norm'].loc[row.name[1:]] - row['pca_bias']),
        axis=1
    )
    # divide by the primary flux to get the contrast
    crossmf['contrastmap'] = crossmf.apply(
        lambda row: row['fluxmap'] / target_subset['mf_prim_flux'].loc[row.name[1:]],
        axis=1
    )

    return crossmf

