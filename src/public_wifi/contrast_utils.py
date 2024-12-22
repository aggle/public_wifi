import numpy as np
import pandas as pd
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



def row_inject_psf(row, star, pos, scale, kklip : int = -1) -> np.ndarray:
    """
    inject a PSF 
    """
    result_row = star.results.loc[row.name]
    stamp = row['stamp']
    # kklip is actually an index, not a mode number, so subtract 1 
    if kklip != -1:
        kklip -= 1
    psf_model = dutils.make_normalized_psf(
        result_row['klip_model'].iloc[kklip].copy(),
        7, # 7x7 psf, hard-coded
        1.,  # total flux of final PSF
    )
    star_flux = measure_primary_flux(stamp, psf_model)
    # compute the companion flux at the given contrast
    inj_flux = star_flux * scale
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


def inject_subtract_detect(star, pos, scale, sim_thresh, min_nref):
    """
    Inject a PSF into a star and recover it.
    star : star object
    pos : (x, y) position relative to center
    scale : contrast relative to primary
    """
    inj_cat = make_injected_cat(star, pos, scale, -1)
    results = inj_cat.apply(
        star.row_klip_subtract,
        sim_thresh=sim_thresh,
        min_nref=min_nref,
        axis=1
    )
    snrmaps = results.apply(
        star.row_make_snr_map,
        axis=1
    ).squeeze()
    return snrmaps
    # return stamp, psf, inj_stamp
