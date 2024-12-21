import numpy as np
import pandas as pd
from itertools import combinations

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
        correlate_mode='same',
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
    """
    matched_filter = make_matched_filter(psf_model)
    detmap = correlate(
        target_stamp,
        matched_filter,
        method='direct',
        mode=correlate_mode)
    detmap = detmap / np.linalg.norm(matched_filter)**2
    # detmap = convolve(target_stamp, psf_model-psf_model.min(), normalize_kernel=True)
    return detmap



def make_series_snrmaps(residuals):
    """Make SNR maps out of a series of arrays. Compute the sigma-clipped noise of an image and divide through"""
    std_maps = residuals.apply(lambda img: sigma_clipped_stats(img)[-1])
    snrmaps = residuals/std_maps
    return snrmaps

def detect_snrmap(snrmaps, snr_thresh=5, n_modes=3) -> pd.Series | None:
    """
    Detect sources using the SNR maps
    snrmaps : pd.Series
      A pd.Series where each entry is the SNR map for a Kklip mode
    thresh : float
      the SNR threshold
    n_modes : int
      the threshold on the number of modes in which a candidate must be detected

    A detection is a pixel that is over the threshold in at least three modes
    """
    stack = np.stack(snrmaps)
    center_pixel = np.floor(((np.array(stack.shape[-2:])-1)/2)).astype(int)
    # get all the pixels over threshold
    initial_candidates = pd.DataFrame(
        np.where(stack >= snr_thresh),
        index=['kklip','dy','dx']
    ).T
    # no candidates? Quit early
    if len(initial_candidates) == 0:
        return None
    initial_candidates['dy'] -= center_pixel[1]
    initial_candidates['dx'] -= center_pixel[0]
    # drop the central pixel
    central_pixel_filter = initial_candidates[['dy','dx']].apply(
        lambda row: all(row.values == (0, 0)) == False,
        axis=1
    ) 
    initial_candidates = initial_candidates[central_pixel_filter].copy()
    # group by row, col and find the ones that appear more than n_modes
    candidate_filter = initial_candidates.groupby(['dy', 'dx']).size() >= n_modes
    candidates = candidate_filter[candidate_filter].index.to_frame().reset_index(drop=True)
    candidates = candidates[['dx','dy']].apply(tuple, axis=1)
    # return candidates
    candidates = candidates.reset_index(name='pixel')
    candidates.rename(columns={"index": "cand_id"}, inplace=True)
    return group_candidates(candidates)


def group_candidates(candidates):
    """
    Given a group of candidate positions, group the contiguous pixels together
    """
    positions = candidates['pixel'].apply(np.array)
    if len(positions) == 0:
        return pd.DataFrame(None, columns=[ 'cand_id', 'pixel' ])
    elif len(positions) == 1:
        return pd.DataFrame({'cand_id': [1], 'pixel' : [tuple(positions) ]})
    else:
        pass
    # compute a distance matrix
    dists = positions.apply(lambda pt1: positions.apply(lambda pt2: np.linalg.norm(pt1 - pt2)))
    # flag neighboring pixels
    distflag = dists < 2
    # loop over the candidates and check the distance matrix for neighbors
    # assign the same group ID to all neighbors
    pos_ids = positions.index
    # this will store the group id
    groups = pd.Series(0, index=pos_ids, name = 'cand_id')
    # initialize the counters
    group_id = 0
    pos_index = 0
    while any(groups==0):
        group_id += 1
        # this is an emergency condition in case i get caught in an infinite loop
        if group_id > pos_ids.size+10:
            break
        test_cand_id = pos_ids[pos_index]
        unassigned_ids = groups.index[groups == 0]
        for cand_id in unassigned_ids:
            flag = distflag.loc[cand_id, test_cand_id]
            if flag:
                groups[cand_id] = group_id
        pos_index += 1
    # make sure the group IDs are consecutive
    for i, g in enumerate(groups.unique()):
        groups[groups == g] = i+1
    # # now for each position, you have a group identifier
    # # switch it around so that for each group identifier, you have a list of positions
    # candidates = pd.Series(
    #     {g: positions.loc[groups[groups == g].index].apply(tuple) for g in groups.unique()},
    #     name='candidates'
    # )
    # candidates = pd.concat(
    #     candidates.to_dict(),
    #     names=['cand_id', 'pix_id']
    # ).reset_index(name='pixel').drop("pix_id", axis=1)
    # return candidates
    candidates['cand_id'] = groups
    return candidates

def jackknife_analysis(
        star,
        sim_thresh = 0.5,
        min_nref = 2,
) -> pd.Series :
    """
    Perform a jackknife test on a star by iteratively doing PSF subtraction, removing one reference each time.

    Parameters
    ----------
    star : starclass.Star
      the star to analyze

    Output
    ------
    jackknife_result : pd.Series
      a series with a hierarchical index of (target_name, kklip) that stores the subtracted array

    """
    references = star.references.query("used == True")
    ref_targets = references.index.get_level_values("target")
    ref_iterator = {i: list(ref_targets[ref_targets != i]) for i in ref_targets}
    results = {}
    for r, refs in ref_iterator.items():
        results[r] = star.cat.apply(
            star.row_klip_subtract,
            sim_thresh=sim_thresh,
            min_nref=min_nref,
            jackknife_reference=r,
            axis=1
        )['klip_sub']
    # do two levels of concatenation to turn it onto a proper series
    jackknife = pd.concat(results, names=['target', 'index']).reorder_levels(['index', 'target'])
    jackknife = pd.concat(jackknife.to_dict())
    jackknife.name = 'klip_jackknife'
    # now make it an SNR map
    jackknife = make_series_snrmaps(jackknife)
    return jackknife
