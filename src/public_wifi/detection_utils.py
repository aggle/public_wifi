import numpy as np
import pandas as pd

from astropy.stats import sigma_clipped_stats

from public_wifi import misc

def flag_candidate_pixels(
        maps : np.ndarray | pd.Series,
        thresh : float,
        n_modes : int = 3
) -> np.ndarray[bool]:
    """
    Flag pixels in a detection cube as True if they meet the detection
    criteria, or False if they don't

    Parameters
    ----------
    maps : np.array | pd.Series
      A stack of detection maps, one per Kklip
    """
    stack = np.stack(maps)
    stackmap = (stack >= thresh)
    mode_detections = np.sum(stackmap.astype(int), axis=0)
    detections = (mode_detections >= n_modes)
    return detections

def make_series_snrmaps(residuals):
    """Make SNR maps out of a series of arrays. Compute the sigma-clipped noise of an image and divide through"""
    std_maps = residuals.apply(lambda img: sigma_clipped_stats(img)[-1])
    snrmaps = residuals/std_maps
    return snrmaps


def detect_snrmap(
        snrmaps,
        snr_thresh : float = 5,
        n_modes : int = 3
) -> pd.DataFrame:
    """
    Detect sources using the SNR maps
    snrmaps : pd.Series
      A pd.Series where each entry is the SNR map for a Kklip mode
    thresh : float
      the SNR threshold
    n_modes : int
      the threshold on the number of modes in which a candidate must be detected

    A detection is a pixel that is over the threshold in at least three modes

    Output
    ------
    candidates : pd.DataFrame
      DataFrame with columns 'cand_id', 'pixel' where pixel is a tuple of (x, y)

    """
    cand_flags = flag_candidate_pixels(snrmaps, snr_thresh, n_modes)
    # no candidates? Quit early
    if (cand_flags == False).all():
        return pd.DataFrame(None, columns=['cand_id', 'pixel'])

    initial_candidate_pixels = pd.DataFrame(
        np.where(cand_flags),
        index=['dy','dx']
    ).T

    center_pixel = misc.get_stamp_center(snrmaps)
    initial_candidate_pixels['dy'] -= center_pixel[1]
    initial_candidate_pixels['dx'] -= center_pixel[0]

    initial_candidate_pixels = initial_candidate_pixels[['dx','dy']].apply(tuple, axis=1)
    # drop the central pixel - assume you can't make a detection there
    candidate_pixels = initial_candidate_pixels[initial_candidate_pixels != (0, 0)]
    candidate_pixels = candidate_pixels.reset_index(name='pixel')
    candidate_pixels.rename(columns={"index": "cand_id"}, inplace=True)
    candidates = group_nearest_candidate_pixels(candidate_pixels)
    return candidates


def group_nearest_candidate_pixels(candidates):
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

def make_jackknife_iterator(references):
    """
    Make an interator for jackknife analysis
    Returns a dict whose key is the reference that has been excluded, and whose
    entries are the other references
    """
    used_refs = references.query("used == True")
    ref_targets = used_refs.index.get_level_values("target")
    ref_iterator = {i: list(ref_targets[ref_targets != i]) for i in ref_targets}
    return ref_iterator

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
    # references = star.references.query("used == True")
    # ref_targets = references.index.get_level_values("target")
    # ref_iterator = {i: list(ref_targets[ref_targets != i]) for i in ref_targets}
    ref_iterator = make_jackknife_iterator(star.references)
    results = {}
    for r, refs in ref_iterator.items():
        results[r] = star.run_klip_subtraction(
            sim_thresh=sim_thresh,
            min_nref=min_nref,
            jackknife_reference=r
        )['klip_sub']
    # do two levels of concatenation to turn it onto a proper series
    jackknife = pd.concat(results, names=['target', 'index']).reorder_levels(['index', 'target'])
    jackknife = pd.concat(jackknife.to_dict())
    jackknife.name = 'klip_jackknife'
    # now make it an SNR map
    jackknife = make_series_snrmaps(jackknife)
    return jackknife

