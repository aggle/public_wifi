"""
Methods for processing the whole catalog of target detections
"""
from pathlib import Path
import pandas as pd
from public_wifi import starclass as sc

from pathlib import Path

def load_catalog(
        catalog_file : str | Path,
        snr_thresh=50
) -> pd.DataFrame:
    """
    Helper function to load the catalog. Users will need to write their own.
    """
    dtypes = {
        'target': str,
        'file': str,
        'filter': str,
        'ra': float,
        'dec': float,
        'x': float,
        'y': float,
        'mag_aper': float,
        'e_mag_aper': float,
        'dist': float,
        'snr': float,
    }
    init_catalog = pd.read_csv(str(catalog_file), dtype=dtypes)
    init_catalog[['x','y']] = init_catalog[['x','y']]-1
    print(f"Filtering out stars with SNR < {snr_thresh}")
    stars = init_catalog['target'].unique()
    snr_thresh = snr_thresh
    above_thresh = init_catalog.groupby("target").apply(
        lambda group: all(group['snr'] >= snr_thresh),
        include_groups=False,
    )
    keep_stars = list(above_thresh[above_thresh].index)
    catalog = init_catalog.query(f"target in {keep_stars}").copy()
    catalog = catalog.sort_values(by='target').reset_index(drop=True)
    return catalog


# This method does all the processing steps. Write your next step below, and
# add it to the execution list
def process_catalog(
        # initalization args
        input_catalog : pd.DataFrame,
        star_id_column : str,
        match_references_on : str | list,
        data_folder : str | Path,
        stamp_size : int = 11,
        bad_references : list = [],
        scale_stamps : bool = False,
        center_stamps : bool = False,
        # psf subtraction args
        min_nref : int = 2,
        sim_thresh : float = 0.5,
        # detection args
        snr_thresh : float = 5.,
        n_modes : int = 3,
        mf_width : int | None = None,
) -> pd.Series :
    """
    Given an input catalog, run the analysis.

    Parameters
    ----------
    input_catalog : pd.DataFrame
      a catalog where each detection is a row
    star_id_column : str,
      this is the column that has the star identifier
    match_references_on : list
      these are the columns that you use for matching references
    stamp_size : int = 15
      what stamp size to use
    scale_stamps : bool = False
      If True, scale all stamps from 0 to 1
    min_nref : int = 2
      Use at least this many reference stamps, regardless of similarity score
    sim_thresh : float = 0.5
      A stamp's similarity score must be at least this value to be included
      If fewer than `min_nref` reference stamps meet this criteria, use the
      `min_nref` ones with the highest similarity scores    
    snr_thresh : float = 5.
      SNR threshold for candidate detections
    n_modes : int = 3
      The number of modes above threshold required to pass candidate checks
    mf_width : int | None = None
      Size of the matched filter. If None, use the stamp size.

    Output
    ------
    stars : pd.Series
      A series where each entry is a Star object with the data and analysis results

    """
    # initialize the catalog
    stars = catalog_initialization(
        input_catalog,
        star_id_column=star_id_column,
        match_references_on=match_references_on,
        data_folder=data_folder,
        stamp_size=stamp_size,
        bad_references=bad_references,
        scale_stamps=scale_stamps,
        center_stamps=center_stamps,
    )
    # perform PSF subtraction
    subtr_args = dict(min_nref=min_nref, sim_thresh=sim_thresh)
    catalog_subtraction(
        stars,
        **subtr_args,
    )
    # perform the detection analysis
    det_args = dict(snr_thresh=snr_thresh, n_modes=n_modes, mf_width=mf_width)
    catalog_detection(
        stars, **det_args
    )

    # perform the candidate checking
    catalog_candidate_validation(
        stars,
        **subtr_args,
    )
    return stars


def catalog_initialization(
        input_catalog : pd.DataFrame,
        star_id_column : str,
        match_references_on : str | list[str],
        data_folder : str | Path,
        stamp_size : int = 15,
        bad_references : str | list[str] = [],
        scale_stamps : bool = False,
        center_stamps : bool = False,
) -> pd.Series :
    """
    initialize the Star objects as a series. This includes creating the
    Star objects from corresponding catalog rows, collecting the corresponding
    references, and computing the stamp similarity scores,.

    Parameters
    ----------
    input_catalog : pd.DataFrame
      a catalog where each detection is a row
    star_id_column : str
      this is the column that has the star identifier
    match_references_on : list
      these are the columns that you use for matching references
    stamp_size : int = 15
      what stamp size to use
    bad_references : str | list[str] = []
      a list of values of the [star_id_column] that should be flagged as not
      suitable for use as PSF references
    min_nrefs : int = 1
      Stars with fewer than this many references in either filter are rejected

    Output
    ------
    stars : pd.Series
      A series where each entry is a Star object with the data and analysis results

    """
    # input format checking
    if isinstance(match_references_on, str):
        match_references_on = [match_references_on]
    if isinstance(bad_references, str):
        bad_references = [bad_references]

    # Create the Star objects from the catalog
    stars = input_catalog.groupby(star_id_column).apply(
        lambda group: sc.Star(
            group.name,
            group,
            data_folder = data_folder,
            stamp_size = stamp_size,
            match_by = match_references_on,
            scale_stamps=scale_stamps,
            center_stamps=center_stamps,
        ),
        include_groups=False
    )
    # flag the bad references
    for br in bad_references:
        if br in stars.index:
            stars[br].is_good_reference = False
    return stars


def catalog_subtraction(
        stars : pd.Series,
        sim_thresh : float = 0.5,
        min_nref : int = 2,
) -> None:
    """
    Perform PSF subtraction on all the stars, setting attributes in-place

    Parameters
    ----------
    stars : pd.Series
      pandas Series where each entry is a starclass.Star object, and the index is the star identifier
    min_nref : int = 2
      Use at least this many reference stamps, regardless of similarity score
    sim_thresh : float = 0.5
      A stamp's similarity score must be at least this value to be included
      If fewer than `min_nref` reference stamps meet this criteria, use the
      `min_nref` ones with the highest similarity scores

    """
    print(f"Subtracting with similarity score threshold: sim >= {sim_thresh}")
    # assign references and compute similarity score
    for star in stars:
        star.set_references(stars, compute_similarity=True)

    for star in stars:
        # KLIP
        star.results = star.run_klip_subtraction(
            sim_thresh=sim_thresh, min_nref=min_nref
        )
    return


def catalog_detection(
        stars : pd.Series,
        snr_thresh : float,
        n_modes : int,
        mf_width : int | None = None,
) -> None:
    """
    Perform MF detection on all the stars

    Parameters
    ----------
    stars : pd.Series
      pandas Series where each entry is a starclass.Star object, and the index
      is the star identifier

    Output
    ------
    updates star.results dataframe in-place. Adds columns for SNR maps,
    detection maps, and candidates
    """
    det_args = dict(snr_thresh=snr_thresh, n_modes=n_modes, mf_width=mf_width)
    for star in stars:
        star.det_args.update(det_args)
        # PSF Convolution
        detmaps = star.apply_matched_filter(contrast=True, throughput_correction=True)
        star.results[detmaps.name] = detmaps
        # SNR
        snrmaps = star.run_make_snr_maps()
        star.results[snrmaps.name] = snrmaps
        # Candidate identification
        candidates = star.results.apply(
            star.row_detect_snrmap_candidates,
            axis=1
        ).squeeze()
        star.results[candidates.name] = candidates
        # flux maps
        fluxmaps = star.run_make_mf_flux_maps()
        star.results[fluxmaps.name] = fluxmaps

        # PCA Results version!
        # star.pca_results = sc.apply_mf_to_pca_results(star.pca_results, mf_width=mf_width)
    return

def catalog_candidate_validation(stars : pd.Series, sim_thresh, min_nref) -> None:
    for star in stars:
        jackknife = star.jackknife_analysis(
            sim_thresh=sim_thresh,
            min_nref=min_nref
        )
        star.results['klip_jackknife'] = jackknife
    return
