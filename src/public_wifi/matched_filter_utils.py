import numpy as np
import pandas as pd

import time
from scipy.signal import correlate
from astropy.convolution import convolve
from astropy.nddata import Cutout2D
from astropy.modeling.models import Gaussian2D
from astropy.stats import sigma_clipped_stats

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

def normalize_array_sigmaclip(x):
    """Normalize an array with sigma-clipped stats"""
    mean, _, std = sigma_clipped_stats(x, sigma=3)
    return (x-mean)/std

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
        nan_center : bool = True,
        pos : tuple[int] | None = None
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
    pos : tuple[int] | None = None
      a (row, col) pixel coordinate. If given, compute the throughput only at
      this position. This makes things go a lot faster
    Output
    ------
    bias : pd.Series
      A pixel map of the bias, as a cumulative sum

    """
    if not isinstance(modes, pd.Series):
        modes = pd.Series({i+1: mode for i, mode in enumerate(modes)})
    center = misc.get_stamp_center(modes)[::-1]
    mode_shape = np.stack(modes).shape[-2:]
    # if you only want one position, skip the correlation and do a dot product
    if pos is not None:
        # if you only want one position, skip the correlation and do a dot product
        # pad the matched filter array so that the size matches up with the stamp,
        # with the central pixel in the right spot
        # pad the matched filter so that it lines up correctly
        mf_padded = line_up_matched_filter_with_pos(
            mf, modes, pos
        )
        bias = modes.apply(
            lambda klmode: np.dot(mf_padded.ravel(), klmode.ravel())**2
        )
        return bias.cumsum()

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


def cross_apply_matched_filters(
        target_star_results,
        all_stars : pd.Series,
        numbasis : list[int] | None = None,
        resid_col : str = 'klip_sub',
        normalize_residuals : bool = False,
        convert_to_flux : bool = True,
        verbose : bool = False,
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
    numbasis : int | list[int] | None = None
      if given, only use results from these principle components
    resid_col : str = 'klip_sub'
      The column to which to apply the matched filter. Use 'klip_sub' to apply
      to the residuals, 'klip_model' to apply to the primary
    normalize_residuals : bool = False
      If True, normalize the residuals such that the sigma-clipped std is 1.
      Also, skip flux conversion since that is meaningless now.
    convert_to_flux : bool = True
      If True, correct by PCA throughput to get the flux and contrast. If False, skip
    verbose : bool = False
      If True, print "finished" when finished

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
    if verbose:
        t0 = time.time()
    # Set the max Kklip value, and filter down to the requested values
    max_kklip = all_stars.apply(lambda star: star.results.index.get_level_values("numbasis").max()).min()
    if isinstance(numbasis, int):
        numbasis = [numbasis]
    if isinstance(numbasis, list):
        pass
    else:
        numbasis = list(range(1, max_kklip+1))
    # set the upper bound on numbasis
    numbasis = list(filter(lambda x: x <= max_kklip, numbasis))
    target_subset = target_star_results.query(f"numbasis in {numbasis}").copy()
    # use this column for the residuals
    # resid_col = 'klip_sub'
    if normalize_residuals:
        target_subset['klip_sub_norm'] = target_subset['klip_sub'].map(normalize_array_sigmaclip)
        resid_col = 'klip_sub_norm'

    # match Kklip and cross-correlate the catalog MF against the target's residuals
    crossmf = all_stars.apply(
        lambda mf_star: target_subset.apply(
            lambda row: correlate(
                row[resid_col],
                mf_star.results.loc[row.name, 'mf'],
                mode='same', method='auto'
            ),
        axis=1
        )
    )
    crossmf = pd.concat({i: row for i, row in crossmf.iterrows()}, names=['mf_star'])
    crossmf.name = 'mf'
    crossmf = pd.DataFrame(crossmf)
    # apply this matched filter to the primary
    # divide by the mf's norm
    crossmf['detmap'] = crossmf.apply(
        lambda row: row['mf'] / all_stars.loc[row.name[0]].results['mf_norm'].loc[row.name[1:]],
        axis=1
    )
    # record the (row, col) position of the strongest response
    crossmf['detpos'] = crossmf['detmap'].map(lambda img: np.unravel_index(np.nanargmax(img), img.shape))

    # if you have normalized the residuals, stop here, because converting to fluxes no longer makes sense
    # these must both be True to proceed
    if (not normalize_residuals) and (convert_to_flux):
        # compute the pca bias of the new matched filter against the *target star's*
        # KLIP modes.
        crossmf_pca_bias = all_stars.apply(
            lambda mf_star: target_subset.apply(
                lambda row: compute_pca_bias(
                    mf_star.results.loc[row.name, 'mf'],
                    # you need *all* the kklips, not just a subset
                    target_star_results['klip_basis'].loc[row.name[0], :row.name[1]],
                    nan_center=True,
                ).iloc[-1],
                axis=1
            )
        )
        # this gets automatically formatted into a dataframe, so let's reshape it
        # into a series
        crossmf_pca_bias = pd.concat(
            {i: row for i, row in crossmf_pca_bias.iterrows()},
            names=['mf_star']
        )
        crossmf['pca_bias'] = crossmf_pca_bias

        # divide the mf by the throughput to get the flux
        crossmf['fluxmap'] = crossmf.apply(
            lambda row: row['mf'] / (all_stars.loc[row.name[0]].results['mf_norm'].loc[row.name[1:]] - row['pca_bias']),
            axis=1
        )
        # and finally, divide by the primary flux to get the contrast
        crossmf['contrastmap'] = crossmf.apply(
            lambda row: row['fluxmap'] / target_subset['mf_prim_flux'].loc[row.name[1:]],
            axis=1
        )
    if verbose:
        t1 = time.time()
        dt = t1-t0
        print(f"Cross-matched filtering finished after {int(dt)} seconds.")

    return crossmf


def select_brightest_pixel(mf_stack, return_index=False):
    """
    For a variety of matched filters, select the strongest response at each pixel. Collapse the mf_star index
    if return_index, then instead of the brightest pixel value, you give the matched filter to use for that index
    """
    # group by all indices except `mf_star`
    names = list(mf_stack.index.names)
    names.pop(names.index("mf_star"))
    gb = mf_stack.groupby(names, group_keys=False)
    mf_max = gb.apply(
        lambda group: np.nanmax(np.stack(group.values), axis=0)
    )
    return mf_max


def line_up_matched_filter_with_pos(
        mf : np.ndarray,
        modes : pd.Series,
        pos : tuple[int]
) -> np.ndarray:
    """pos is in (row, col) convention"""
    mode_shape = np.stack(modes).shape[-2:]
    # get the boundaries
    mf_half = misc.get_stamp_center(mf)
    # if this is negative, add padding
    pad_lb = np.array(pos) - mf_half
    # if this is positive, add padding
    pad_ub = (np.array(mode_shape)-1) - (np.array(pos) + mf_half)
    mf_padded = mf.copy()
    if pad_lb[0] < 0:
        mf_padded = mf_padded[-1*pad_lb[0]:, :]
    elif pad_lb[0] > 0:
        mf_padded = np.pad(mf_padded, ((pad_lb[0], 0), (0, 0)))
    else:
        pass
    if pad_lb[1] < 0:
        mf_padded = mf_padded[:, -1*pad_lb[1]:]
    elif pad_lb[1] > 0:
        mf_padded = np.pad(mf_padded, ((0, 0), (pad_lb[1], 0)))
    else:
        pass
    if pad_ub[0] > 0:
        mf_padded = np.pad(mf_padded, ((0, pad_ub[0]), (0, 0)))
    elif pad_ub[0] < 0:
        mf_padded = mf_padded[:pad_ub[0]]
    else:
        pass
    if pad_ub[1] > 0:
        mf_padded = np.pad(mf_padded, ((0, 0), (0, pad_ub[1])))
    elif pad_ub[1] < 0:
        mf_padded = mf_padded[:, :pad_ub[1]]
    else:
        pass
    return mf_padded
