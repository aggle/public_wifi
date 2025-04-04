import numpy as np
import pandas as pd

from scipy import stats
from astropy.stats import sigma_clipped_stats

from public_wifi import misc
from public_wifi import matched_filter_utils as mf_utils

def flag_candidate_pixels(
        maps : np.ndarray | pd.Series,
        thresh : float,
        n_modes : int = 3,
        min_kklip : int = 10,
) -> np.ndarray[bool]:
    """
    Flag pixels in a detection cube as True if they meet the detection
    criteria, or False if they don't

    Parameters
    ----------
    maps : np.array | pd.Series
      A stack of detection maps, one per Kklip
    """
    stack = np.stack(maps)[min_kklip:]
    stackmap = (stack >= thresh)
    mode_detections = np.sum(stackmap.astype(int), axis=0)
    detections = (mode_detections >= n_modes)
    return detections

def make_series_snrmaps(residuals):
    """Make SNR maps out of a series of arrays. Compute the sigma-clipped noise of an image and divide through"""
    std_maps = residuals.apply(lambda img: sigma_clipped_stats(img)[-1])
    snrmaps = residuals/std_maps
    snrmaps.name = 'snrmap'
    return snrmaps


def calc_snr_from_series(
        snr_series : pd.Series,
        thresh: float = 5.,
        n_modes : int =3,
        min_kklip : int = 10,
):
    # assume it's indexed by kklip
    snr_series = snr_series.squeeze()[min_kklip:]
    # return the median of the top 3 snr values
    snr = snr_series.sort_values(ascending=False)[:n_modes].median()
    return snr

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
    ref_iterator = make_jackknife_iterator(star.references)
    results = {}
    for r, refs in ref_iterator.items():
        results[r] = star.run_klip_subtraction(
            sim_thresh=sim_thresh,
            min_nref=min_nref,
            jackknife_reference=r
        )['klip_sub']
    jackknife = pd.concat(results, names=['target', 'cat_row', 'numbasis'])
    # reorder the index to put the catalog row first
    jackknife = jackknife.reorder_levels(['cat_row', 'numbasis', 'target'])
    jackknife.sort_index(level=[0, 1], inplace=True)
    # now make it an SNR map
    jackknife = make_series_snrmaps(jackknife)
    jackknife.name = 'klip_jackknife'
    return jackknife


def compare_stamp_distribution(
    stamp,
    ref_distro : np.ndarray | None = None
) -> float:
    """
    Compare the pixel values in a stamp to a normal distribution (or to a
    reference distribution, if provided)
    """
    stamp = np.sort(np.array(stamp).ravel())
    if ref_distro is None:
        score = stats.shapiro(stamp).statistic
    else:
        ref_distro = np.sort(np.array(ref_distro).ravel())
        score = stats.kstest(stamp, ref_distro).statisic
    return score


def estimate_crossmf_filter_position(
        crossmf_df : pd.DataFrame,
        numbasis : list[int] | None = None,
        base_uncertainty : float = 1,
) -> pd.Series :
    """
    compute the position from a weighted sum of the cross-matched filter
    results. The uncertainty is assumed to be half a pixel and the weight is
    the normalized matched filter response.

    Parameters
    ----------
    crossmf_df : pd.DataFrame
      A dataframe with a detpos and detmap column
    numbasis : list[int] | None = None
      if provided, use these values of numbasis 
    base_uncertainty : float = 1
      uncertainty of the centroid in pixel units

    Output
    ------
    pos : pd.Series
      a series with row, col, d_row, and d_col entries
    """
    def estimate_kklip_position(group):
        """
        Compute the posiiton for many MF responses for a single Kklip
        """
        weights = group.apply(
            lambda row: row['detmap'][*row['detpos']],
            axis=1
        )
        weights = weights/weights.sum()
        weighted_pos = group['detpos'].apply(
            lambda row: pd.Series(row, index=['row','col'])
        ).apply(
            lambda row: row*weights.loc[row.name], axis=1
        ).sum()
        # sigma**2 = sum(w_i**2 * sigma_i**2) w_i -> weight sigma_i -> uncertainty
        weighted_unc =  weights.apply(
            lambda row_wts: pd.Series(row_wts, index=['row','col'])
        ).apply(
            lambda row_wts: np.sqrt((row_wts**2) * (base_uncertainty)**2), axis=1
        ).apply(np.linalg.norm)
        pos = pd.Series({
            'row': weighted_pos['row'],
            'col': weighted_pos['col'],
            'd_row': weighted_unc['row'],
            'd_col': weighted_unc['col']
        })
        return pos

    if numbasis is not None:
        crossmf_df = crossmf_df.query(f"numbasis in {numbasis}")
    pca_positions = crossmf_df.groupby(['numbasis']).apply(lambda group: estimate_kklip_position(group))
    positions = {}
    for i in ['row','col']:
        weights = 1/pca_positions['d_'+i]**2
        pos = np.sum(pca_positions[i] * weights / (weights.sum()))
        unc = np.sqrt(1/weights.sum())
        positions[i] = pos
        positions['d_'+i] = unc
    return pd.Series(positions)

