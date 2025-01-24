import itertools
import numpy as np
import pandas as pd
from scipy import optimize
from public_wifi import catalog_processing, misc
from public_wifi import detection_utils as dutils
from public_wifi import matched_filter_utils as mf_utils
from public_wifi import catalog_processing as catproc


def measure_primary_flux(
        stamp : np.ndarray,
        psf_model : np.ndarray,
        mf_args = {}
) -> float :
    """
    Measure the flux of an unsubtracted PSF

    Parameters
    ----------
    stamp : np.ndarray
      2-D stamp with a star in the middle
    psf_model : np.ndarray
      a model of the PSF to use for flux measurement
    mf_args : {}
      Arguments to pass to mf_utils.apply_matched_filter

    Output
    ------
    flux : float
      the flux of the primary
    """
    # mf = mf_utils.make_matched_filter(model_psf)
    # collect arguments
    kwargs = dict(
        mf_width = mf_args.get("mf_width", min(psf_model.shape)),
        correlate_mode=mf_args.get('correlate_mode', 'valid'),
        throughput_correction=mf_args.get('throughput_correction', True),
        kl_basis=mf_args.get('kl_basis', None),
    )
    star_flux = mf_utils.apply_matched_filter(
        stamp, psf_model,
        **kwargs,
    )
    center = misc.get_stamp_center(star_flux)
    return star_flux[*center[::-1]]


def inject_psf(
        stamp : np.ndarray,
        psf : np.ndarray,
        pos : tuple[int, int],
) -> np.ndarray :
    """
    Add the given PSF to the stamp and return the combined image

    Parameters
    ----------
    stamp : np.ndarray
      2D image array
    psf : np.ndarray
      2D image
    pos : tuple[int, int]
      integer (x, y) / (col, row) coordinates of the injection location
      relative to the center

    Output
    ------
    injected_stamp : np.ndarray
      stamp with a new PSF in it, same shape as the input stamp

    """
    # make sure pos is an array and convert to row, col format
    pos = np.array(pos)[::-1]
    stamp_center = misc.get_stamp_center(stamp)
    # interpolate the PSF to be centered. It will be rescaled anyway so you
    # don't need to worry about conserving flux
    psf_center = misc.get_stamp_center(psf)

    # pad the injection stamp by the PSF shape in each direction.
    # after you add the PSF in, you can cut it back again
    pad = psf.shape
    injected_stamp_padded = np.pad(stamp, pad, 'constant', constant_values=0)

    # compute the lower left corner of the injection
    # the new center
    inj_center = stamp_center + np.array(pad)
    # the place you added the PSF
    injection_pos = inj_center + pos
    # the corner of the injection site
    corner = injection_pos - psf_center
    yrange = corner[0] , corner[0]+pad[0]
    xrange = corner[1] , corner[1]+pad[1]
    injected_stamp_padded[yrange[0]:yrange[1], xrange[0]:xrange[1]] += psf
    # remove the padding
    injected_stamp =  injected_stamp_padded[
        pad[1]:-pad[1], pad[0]:-pad[0]
    ].copy()
    return injected_stamp


def row_inject_psf(
        row,
		star,
		pos,
		contrast,
		kklip : int = -1,
        mf_args={},
) -> np.ndarray:
    """
    inject a PSF into a row
    row : the row of star.cat
    star : the star object with a results dataframe
    pos : the position in x, y relative to center
    contrast : the contrast at which to scale the injection
    kklip : int = -1
      The mode number to use. Use -1 for the largest Kklip
    mf_args = {}
      Arguments to pass to mf_utils.make_normalized_psf -
      width and scale
    """
    result_row = star.results.loc[row.name]
    stamp = row['stamp']
    # kklip is actually used as an index, not a mode number, so subtract 1 
    if kklip != -1:
        kklip -= 1

    psf_model = mf_utils.make_normalized_psf(
        result_row['klip_model'].iloc[kklip].copy(),
        scale = 1.,
    )
    star_flux = measure_primary_flux(stamp, psf_model, mf_args=mf_args)
    # compute the companion flux at the given contrast
    inj_flux = contrast * star_flux
    inj_stamp = inject_psf(stamp, psf_model * inj_flux, pos)

    inj_row = row.copy()
    inj_row['stamp'] = inj_stamp
    return inj_row


def cat_inject_psf(star, pos, contrast, kklip, mf_args={}):
    """
    Apply row_inject_psf to a star. You can use this function with the stars
    pd.Series to inject a fake at the same contrast/position/kklip to all the
    stars.
    Adds the column 'inj_stamp' to the star catalog entries
    """
    inj_cat = star.cat.apply(
        row_inject_psf,
        star=star, pos=pos, contrast=contrast, kklip=kklip, mf_args=mf_args,
        axis=1
    )
    star.cat['inj_stamp'] = inj_cat['stamp']
    return 

def find_injection(img : np.ndarray, pos : np.ndarray):
    """
    Sometimes the brightest pixel in the residual / detection map is 1 off from
    the injection site, especially when convolving with the PSF mode. Find it.
    Parameters
    ----------
    img : np.ndarray
      the 2-D detection map
    pos : np.ndarray
      the injection site in center-origin coordinates i.e. (0, 0) is the center
      of the img.

    Output
    ------
    recovered_pos : np.ndarray
      The recovered position of the detection, in lower-left origin coordinates
    """
    pos_arr = misc.center_to_ll_coords(img.shape[0], pos)[::-1]
    # get the search range
    row_range = np.arange(max(pos_arr[0]-1, 0), min(pos_arr[0]+2, img.shape[0]))
    col_range = np.arange(max(pos_arr[1]-1, 0), min(pos_arr[1]+2, img.shape[1]))
    # assume that the recovered site is the injection site for initialization
    # purposes
    max_pos = pos_arr.copy()
    max_flux = img[*max_pos]
    
    for r in row_range:
        for c in col_range:
            if img[r, c] > max_flux:
                max_flux = img[r, c]
                max_pos[0] = r
                max_pos[1] = c
            else:
                continue
    return max_pos

def calc_snr_from_series(
        snr_series : pd.Series, thresh: float = 5., n_modes : int =3
):
    snr_series = snr_series.squeeze()
    # return the median of the top 3 snr values
    snr = snr_series.sort_values(ascending=False)[:n_modes].min()
    # above_thresh = snr_series[snr_series >= thresh]
    # is_detected = len(above_thresh) >= n_modes
    # if is_detected == False:
    #     return np.nan
    # snr = above_thresh.sort_values(ascending=False)[:n_modes].mean()
    # return above_thresh.median()
    return snr

def inject_subtract_detect(
        star, pos, contrast, use_kklip, n_modes = 3, snr_thresh=5,
) -> pd.Series:
    """
    Inject subtract and measure the SNR for a fake companion
    """
    # first, copy the catalog over
    cat_inject_psf(star, pos, contrast, kklip=use_kklip)

    results = star.run_klip_subtraction(stamp_column='inj_stamp')

    snrmaps = star.run_make_snr_maps(results)
    recovered_snr = snrmaps.apply(
        lambda col: pd.Series({'recovered_snr': col.apply(
            lambda img: img[*find_injection(img, pos)]
        )})
    )
    # aggregate the SNR values into a single value
    snr = recovered_snr.apply(
        calc_snr_from_series,
        thresh=snr_thresh,
        n_modes=n_modes,
        axis=1

    )
    return snr


def row_inject_subtract_detect(
        star,
        row,
        pos,
        contrast,
        sim_thresh,
        min_nref,
        snr_thresh,
        n_modes
):
    """
    Inject a PSF into a star and recover it.
    star : star object
    pos : (x, y) position relative to center
    contrast : flux relative to primary

    Output
    ------
    True if injection is above SNR threshold, false if not
    """
    inj_row = row_inject_psf(row, star=star, pos=pos, contrast=contrast, kklip=-1)
    results = star._row_klip_subtract(
        inj_row,
        **star.subtr_args,
        # sim_thresh=sim_thresh,
        # min_nref=min_nref,
    )
    snrmaps = star._row_make_snr_map(results).squeeze()
    detmap = dutils.flag_candidate_pixels(
        snrmaps,
        thresh=snr_thresh,
        n_modes=n_modes,
    )
    center = np.tile(np.floor((star.stamp_size-1)/2).astype(int), 2)
    # recover the SNR at the injected position
    inj_pos = center + np.array(pos)[::-1]
    inj_snr = np.median(
        np.stack(snrmaps.values)[..., inj_pos[0], inj_pos[1]]
    )
    # get the detection flag at the detected positions
    is_detected = detmap[*inj_pos]
    return inj_snr, is_detected


def make_star_contrast_curves(
        star,
        ub : float = 1.,
        lb : float = 1e-4,
        thresholds : list[float] = [5, 3, 1],
) -> pd.DataFrame:
    """
    Generate the contrast curves for a star

    Parameters
    ----------
    star : starclass.Star
    ub : float = 1.
      The upper bound of the contrasts to search
    lb : float = 1e-4
      Contrast search lower bound
    thresholds : list[float]
      the thresholds at which to report the contrast
    """
    def match_snr_thresh(contrast, row, pos, snr_thresh=5):
        snr = star._row_inject_subtract_detect(
            row, pos, contrast, snr_thresh, n_modes=star.det_args['n_modes']
        )[0]
        return np.abs(snr - snr_thresh)
    center = int(np.floor(star.stamp_size/2))
    # iterate over every pixel
    positions = list(itertools.product(
        np.arange(star.stamp_size) - center,
        np.arange(star.stamp_size) - center
    ))
    # remove places to not teset
    positions.pop(positions.index((0,0)))

    contrast_maps = {}

    for i, results_row in star.results.iterrows():
        row_positions = positions.copy()

        # remove candidates from injection analysis
        for pos in results_row['snr_candidates']['pixel']:
            row_positions.pop(row_positions.index(pos))
        contrast_df = pd.DataFrame(row_positions, columns=['x', 'y'], dtype=int)
        for threshold in thresholds:
            contrast_df[f'{threshold}'] = contrast_df.apply(
                lambda contrast_row: optimize.minimize_scalar(
                    match_snr_thresh, 
                    bounds=(lb, ub),
                    args=(star.cat.loc[i], contrast_row[['x','y']].values.astype(int), threshold)
                ).x,
                axis=1
            )
        contrast_maps[i] = contrast_df
    return contrast_maps

def contrast_map_to_radial(contrast_map, stamp_size):
    """Collapse a contrast map along the radial axis"""
    contrast_map['sep'] = contrast_map[['x','y']].apply(
        np.linalg.norm,
        axis=1
    )
    return contrast_map



def inject_and_recover_snr(
        star,
        row,
        contrast,
        pos,
        snr_thresh=5.,
        n_modes=3,
        kklip=None,
        plot=False
) -> float:
    """
    Inject a PSF into a stamp and recover the SNR, as defined by the residuals
    of the stamp itself.
    """
    inj_row = row_inject_psf(
        row, star=star, pos=pos, contrast=contrast, kklip=-1
    )
    results = star._row_klip_subtract(
        inj_row,
        **star.subtr_args,
    )
    snrmaps = star._row_make_snr_map(results).squeeze()
    inj_snr = snrmaps.apply(
        lambda img: img[*find_injection(img, pos)]
    )
    if kklip is None:
        snr = calc_snr_from_series(
            inj_snr, thresh=5., n_modes=3
        )
    else:
        snr = inj_snr[kklip]

    return snr

def optimize_snr_vs_thresh(
    contrast, star, row, pos, snr_thresh=5., n_modes=3, kklip=None
):
    """
    Compute the difference between the snr and the target threshold.
    Pass this to a minimization function like optimize.minimize_scalar
    """
    snr = inject_and_recover_snr(
        star, row, contrast, pos, snr_thresh, n_modes, kklip
    )
    return np.abs(snr - snr_thresh)
