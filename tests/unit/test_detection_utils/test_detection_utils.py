import pytest

from public_wifi import starclass as sc
from public_wifi import detection_utils as dutils
from public_wifi import misc


def test_snr_map(random_processed_star):
    star = random_processed_star
    print(f"Randomly chosen star: {star.star_id}")
    assert('snrmap' in star.results.columns)
    # assert(len(star.results.snrmap.iloc[0]) == len(star.results['klip_sub'] c)
    assert(
        all(
            star.results.apply(
                lambda row: dutils.np.shape(sc.np.stack(row['klip_sub'])) == sc.np.shape(sc.np.stack(row['snrmap'])),
                axis=1
            )
        )
    )


def test_flag_candidate_pixels(star_with_candidates):
    star = star_with_candidates
    snrmap = star.results['snrmap'].iloc[1]
    flags = dutils.flag_candidate_pixels(snrmap, 3, 3)
    assert(flags.ndim == 2)
    assert(flags.dtype == bool)
    assert(flags.any())

def test_detect_snrmap(star_with_candidates):
    star = star_with_candidates
    snrmap = star.results['snrmap'].iloc[1]
    candidates = dutils.detect_snrmap(snrmap, 5, 3)
    print(candidates)
    assert(isinstance(candidates, dutils.pd.DataFrame))
    assert(all([c in candidates.columns for c in ['cand_id', 'pixel']]))


@pytest.mark.skip("Function not implemented")
@pytest.mark.parametrize('snr_thresh', [100, 5])
def test_detect_snrmap_dev(star_with_candidates, snr_thresh):
    star = star_with_candidates
    snrmap = star.results['snrmap'].iloc[1]
    candidates = dutils.detect_snrmap_dev(snrmap, snr_thresh, 3)
    print(snr_thresh, candidates)
    assert(isinstance(candidates, dutils.pd.DataFrame))
    assert(all([c in candidates.columns for c in ['cand_id', 'pixel']]))
