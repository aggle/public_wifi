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
    injected_stamp = np.pad(stamp, psf.shape, 'constant', constant_values=0)

    # compute the lower left corner of the injection
    # the new center
    inj_center = stamp_center + np.array(psf.shape)
    # the place you added the PSF
    injection_pos = inj_center + pos
    # the corner of the injection site
    corner = injection_pos - psf_center
    yrange = corner[0] , corner[0]+psf.shape[0]
    xrange = corner[1] , corner[1]+psf.shape[1]
    injected_stamp[yrange[0]:yrange[1], xrange[0]:xrange[1]] += psf
    # remove the padding
    return injected_stamp[
        psf.shape[1]:-psf.shape[1], psf.shape[0]:-psf.shape[0]
    ].copy()



def inject_subtract_detect(star, pos, scale):
    """
    Inject a PSF into a star and recover it.
    star : star object
    pos : (x, y) position relative to center
    scale : contrast relative to primary
    """
    ind = 0
    stamp = star.cat.loc[ind, 'stamp'].copy()
    # make a PSF to inject, but first use it as a matched filter to meaure the
    # star flux
    psf = dutils.make_normalized_psf(
        star.results.loc[ind, 'klip_model'].iloc[-1].copy(),
        7,
        1.,
    )
    mf = dutils.make_matched_filter(psf)
    star_flux = dutils.apply_matched_filter(stamp, mf, correlate_mode='valid').max()
    inj_flux = star_flux * scale
    psf *= inj_flux
    print(stamp[5, 5], star_flux, psf[3, 3])

    inj_stamp = inject_psf(stamp, psf, pos)
    return stamp, psf, inj_stamp
