import numpy as np
import pandas as pd
from public_wifi import detection_utils as dutils

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
    star_flux = dutils.apply_matched_filter(stamp, mf, correlate_mode='valid').max()
    return star_flux

def inject_subtract_detect(star, pos, scale):
    """
    Inject a PSF into a star and recover it.
    star : star object
    pos : (x, y) position relative to center
    scale : contrast relative to primary
    """
    # filter index
    ind = 0 # F814W
    stamp = star.cat.loc[ind, 'stamp'].copy()
    # make a PSF to inject, but first use it as a matched filter to measure the
    # star flux
    psf = dutils.make_normalized_psf(
        star.results.loc[ind, 'blip_model'].iloc[-1].copy(),
        7, # 7x7 psf
        1.,  # total flux of final PSF
    )
    # measure the primary star flux so you can set the contrast
    star_flux = measure_primary_flux(stamp, psf)
    # compute the companion flux at the given contrast
    inj_flux = star_flux * scale
    inj_stamp = inject_psf(stamp, psf * inj_flux, pos)

    # perform PSF subtraction and detection

    return stamp, psf, inj_stamp
