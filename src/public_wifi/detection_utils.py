import numpy as np
from scipy.signal import correlate

from astropy.nddata import Cutout2D

## Detection
def make_matched_filter(stamp, width : int | None = None):
    center = np.floor(np.array(stamp.shape)/2).astype(int)
    if isinstance(width, int):
        stamp = Cutout2D(stamp, center[::-1], width).data
    stamp = np.ma.masked_array(stamp, mask=np.isnan(stamp))
    stamp = stamp - np.nanmin(stamp)
    stamp = stamp/np.nansum(stamp)
    stamp = stamp - np.nanmean(stamp)
    return stamp.data

def apply_matched_filter(
        target_stamp : np.ndarray,
        psf_model : np.ndarray,
) -> np.ndarray:
    """
    Apply the matched filter as a correlation. Normalize by the matched filter norm.
    """
    matched_filter = make_matched_filter(psf_model)
    detmap = correlate(
        target_stamp,
        matched_filter,
        method='direct',
        mode='same')
    detmap = detmap / np.linalg.norm(matched_filter)**2
    return detmap
