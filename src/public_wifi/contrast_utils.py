import itertools
import numpy as np
import pandas as pd
from scipy import optimize
from public_wifi import misc
from public_wifi import detection_utils as dutils
from public_wifi import catalog_processing as catproc


def measure_primary_flux(
        stamp : np.ndarray,
        model_psf : np.ndarray,
) -> float :
    """
    Measure the flux of an unsubtracted PSF

    Parameters
    ----------
    stamp : np.ndarray
      2-D stamp with a star in the middle
    model_psf : np.ndarray
      a model of the PSF to use for flux measurement

    Output
    ------
    flux : float
      the flux of the primary
    """
    mf = dutils.make_matched_filter(model_psf)
    star_flux = dutils.apply_matched_filter(
        stamp, mf, correlate_mode='valid'
    ).max()
    return star_flux


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
    stamp_center = np.floor(np.array(stamp.shape)/2).astype(int)
    psf_center = np.floor(np.array(psf.shape)/2).astype(int)

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


def row_inject_psf(row, star, pos, contrast, kklip : int = -1) -> np.ndarray:
    """
    inject a PSF
    row : the row of star.cat
    star : the star object with a results dataframe
    pos : the position in x, y relative to center
    contrast : the contrast at which to scale the injection
    kklip : int = -1
      The mode number to use. Use -1 for the largest Kklip
    """
    result_row = star.results.loc[row.name]
    stamp = row['stamp']
    # kklip is actually used as an index, not a mode number, so subtract 1 
    if kklip != -1:
        kklip -= 1
    psf_model = dutils.make_normalized_psf(
        result_row['klip_model'].iloc[kklip].copy(),
        7, # 7x7 psf, hard-coded
        1.,  # total flux of final PSF
    )
    star_flux = measure_primary_flux(stamp, psf_model)
    # compute the companion flux at the given contrast
    inj_flux = contrast * star_flux
    inj_stamp = inject_psf(stamp, psf_model * inj_flux, pos)

    inj_row = row.copy()
    inj_row['stamp'] = inj_stamp
    return inj_row


def make_injected_cat(star, pos, scale, kklip):
    """Apply row_inject_psf to the whole catalog"""
    inj_cat = star.cat.apply(
        row_inject_psf,
        star=star, pos=pos, scale=scale, kklip=kklip,
        axis=1
    )
    return inj_cat


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
    snrmaps = star.row_make_snr_map(results).squeeze()
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


def build_contrast_curve(
        star,
        row,
        sim_thresh,
        min_nref,
        snr_thresh,
        n_modes
):
    """
    Return value is an array of the flux required for an snr_thresh detection
    at a particular radius.
    """
    stamp_shape = star.stamp_size
    center = np.floor((stamp_shape-1)/2).astype(int)
    positions = itertools.combinations_with_replacement(
        np.arange(stamp_shape)[2:-2] - center, 2
    )
    contrasts = [1., 0.5, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4][:2]
    detections = []
    # detmap = np.zeros((stamp_shape, stamp_shape), dtype=bool)
    pos = list(positions)[0]
    detections = pd.DataFrame(None, columns=['contrast', 'pos', 'is_det'])
    # initialize loop variables
    is_detected, contrast = True, contrasts[0]
    while is_detected:
    # for contrast in contrasts:
        snr, is_detected = row_inject_subtract_detect(
            star,
            star.cat.iloc[-1], 
            pos, contrast,
            sim_thresh,
            min_nref,
            snr_thresh,
            n_modes
        )
        detection = pd.Series({'pos': pos, 'contrast': contrast, 'det': is_detected})
        print(detection)
        print(f"PSF detected with snr {snr:.2f} at {contrast:1.1e} contrast")
        contrast = contrast / 1.1
    return detections


def make_star_contrast_curves(
        star,
        ub : float = 1.,
        lb : float = 1e-4
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
    """
    def match_snr_thresh(contrast, row, pos, snr_thresh=5):
        snr = star.row_inject_subtract_detect(row, pos, contrast, snr_thresh)[0]
        return np.abs(snr - snr_thresh)
    center = int(np.floor(star.stamp_size/2))
    positions = list(itertools.product(
        np.arange(star.stamp_size) - center,
        np.arange(star.stamp_size) - center
    ))
    contrast_maps = {}
    for i, results_row in star.results.iterrows():
        row_positions = positions.copy()
        # remove candidates from injection analysis
        for pos in results_row['snr_candidates']['pixel']:
            row_positions.pop(row_positions.index(pos))
        contrast_df = pd.DataFrame(row_positions, columns=['x', 'y'], dtype=int)
        for sigma in [5, 3, 1]:
            contrast_df[f'{sigma}'] = contrast_df.apply(
                lambda contrast_row: optimize.minimize_scalar(
                    match_snr_thresh, 
                    bounds=(lb, ub),
                    args=(star.cat.loc[i], contrast_row[['x','y']].values.astype(int), sigma)
                ).x,
                axis=1
            )
        contrast_maps[i] = contrast_df
    return contrast_maps
