import abc
from pathlib import Path

import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

from public_wifi import subtraction_utils as subutils
from public_wifi import detection_utils as detutils

def load_catalog(catalog_file, snr_thresh=50):
    """Helper function to load the catalog"""
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
    all_stars = init_catalog['target'].unique()
    snr_thresh = snr_thresh
    above_thresh = init_catalog.groupby("target").apply(
        lambda group: all(group['snr'] >= snr_thresh),
        include_groups=False,
    )
    keep_stars = list(all_stars[above_thresh])
    catalog = init_catalog.query(f"target in {keep_stars}").copy()
    return catalog


class Star:
    """
    This is the central object. It carries around everything associated with a star:
    - the star's ID
    - the star's stamps, ids and their IDs and metadata
      - metadata : file name and position in the array
    - the star's PSF subtraction products
    - the star's companion detection products
    - if it's good for use as a PSF reference
    - if it has companions detected
    - a list of the detections

    It has the following methods:
    - cut_stamps : cut out the stamps from the fits files
    - klip_subtract : perform PSF subtraction
    - klip_detect : apply detection methods to PSF-subtracted data products
    - klip_contrast : compute the contrast / detection limits
    """
    def __init__(
            self,
            star_id : str,
            group : pd.DataFrame,
            data_folder : str | Path,
            stamp_size : int  = 15,
            match_by : list[str] = ['filter'],
            scale_stamps = False,
    ) -> None:
        """
        Initialize a Star object from a row of the catalog

        Parameters
        ----------
        match_by : list[str] = ['filter']
          a list of columns to use for matching targets with references. e.g. which filter
        """
        self.star_id = star_id
        self.stamp_size = stamp_size
        self.data_folder = data_folder
        self.cat = group#.reset_index(drop=True)
        self.match_by = match_by
        # status flags
        self.is_good_reference = True # assumed True
        self.has_companions = False
        # values that are initialized by methods
        self.cat['cutout'] = self.cat.apply(
            lambda row: self.get_cutout(row, stamp_size),
            axis=1,
        )
        self.cat['stamp'] = self.cat['cutout'].apply(lambda ct: ct.data.copy())
        if scale_stamps:
            self.cat['stamp'] = self.cat['stamp'].apply(self.scale_stamp)
        # measure the background
        self.cat['bgnd'] = self.measure_bgnd(51, 20)
        return

    # has_companions should always be the opposite of is_good_reference
    @property
    def has_companions(self):
        return self._has_companions
    @has_companions.setter
    def has_companions(self, new_val : bool):
        self._has_companions = new_val
        # After the change in state, check and set, if necessary, the reference status
        self.check_reference()

    @property
    def cat(self):
        """The catalog entries corresponding to this star"""
        return self._cat
    @cat.setter
    def cat(self, new_val : pd.DataFrame):
        new_val = new_val.reset_index(names='cat_id')
        self._cat = new_val.copy()

    def generate_match_query(self, row):
        """
        Generate the query string based on self.match_by.
        The columns specified by match_by must be of type str
        """
        match_values = {m: row[m] for m in self.match_by}
        query = "and ".join(f"{m} == '{v}'" for m, v in match_values.items())
        return query

    def check_reference(self):
        """This method checks for all the conditions"""
        is_ok = False
        if self.has_companions == False:
            is_ok = True
        self.is_good_reference = is_ok

    def get_cutout(
            self,
            row,
            stamp_size : int,
    ) -> Cutout2D:
        """
        Cut out the stamp from the data
        """
        filepath = self.data_folder / row['file']
        img = fits.getdata(str(filepath), 'SCI')
        wcs = WCS(fits.getheader(str(filepath), 'SCI'))
        stamp = Cutout2D(
            img,
            (row['x'],row['y']),
            size=stamp_size,
            wcs=wcs,
            mode='trim',
            fill_value = np.nan,
            copy=True
        )
        # recenter on the brightest pixel
        maxpix = np.array(
            np.unravel_index(np.nanargmax(stamp.data), stamp.shape)[::-1]
        ) + np.array(stamp.origin_original)
        stamp = Cutout2D(
            img,
            tuple(maxpix),
            size=stamp_size,
            wcs=wcs,
            mode='trim',
            fill_value = np.nan,
            copy=True
        )

        return stamp

    def scale_stamp(self, stamp):
        # scale a stamp form 0 to 1
        return (stamp - np.nanmin(stamp))/np.ptp(stamp)

    def measure_bgnd(self, stamp_size=51, bgnd_rad=20):
        """
        stamp_size = 51
          the size of the stamp to use for measureing the background
        bgnd_rad : 20
            pixels further from the center than this are used to measure the bgnd 
        """
        bgnd_stamps = self.cat.apply(self.get_cutout, stamp_size=self.stamp_size, axis=1)
        center = int(np.floor(self.stamp_size/2))
        sep_map = np.linalg.norm(np.mgrid[:self.stamp_size, :self.stamp_size] - center, axis=0)
        bgnd_mask = sep_map < bgnd_rad
        bgnd = bgnd_stamps.apply(
            lambda stamp: (np.nanmean(stamp.data[bgnd_mask]), np.nanstd(stamp.data[bgnd_mask]))
        )
        return bgnd

    def set_references(self, other_stars, compute_similarity=True):
        """
        Assemble the references for each stamp. Put "good reference" checks here

        other_stars:
          pd.Series of Star objects
        compute_similarity : bool = True:
          if True, compute the similarity score
        """
        # references = pd.concat(stars[stars.index != self.star_id].apply(lambda s: s.meta.copy()))
        references = {}
        for star in other_stars[other_stars.index != self.star_id]:
            if star.is_good_reference == False:
                pass
            else:
                references[star.star_id] = star.cat.copy()
        references = pd.concat(references, names=['target', 'index'])
        references['used'] = False
        self.references = references
        if compute_similarity:
            self.compute_similarity()

    def compute_similarity(self):
        """Compute the similarity between the target stamps and the references"""
        # initialize an empty column of floats
        self.references['sim'] = np.nan
        # for each row, select the references, compute the similarity, and
        # store it in the column
        for i, row in self.cat.iterrows():
            target_stamp = row['stamp']
            query = self.generate_match_query(row)

            sim = self.references.query(query)['stamp'].apply(
                lambda ref_stamp: ssim(
                    ref_stamp,
                    target_stamp,
                    data_range=np.ptp(np.stack([ref_stamp, target_stamp]))
                )
            )
            self.references.loc[sim.index, 'sim'] = sim

    def row_get_references(self, row, sim_thresh=0.0, min_nref=2):
        """Get the references associated with a row entry"""
        query = self.generate_match_query(row)
        reference_rows = self.references.query(query).sort_values(by='sim', ascending=False)
        # select the refs above threshold, or at least the top 5
        nrefs = len(reference_rows.query(f"sim >= {sim_thresh}"))
        # update nrefs if it is too small
        if nrefs <= min_nref:
            print(f"Warning: {self.star_id} has fewer than {min_nref} references above threshold!")
            nrefs = min_nref
        reference_rows = reference_rows[:nrefs]
        return reference_rows

    def row_klip_subtract(self, row, numbasis=None, sim_thresh=0.0, min_nref=2):
        """Wrapper for KLIP that can be applied on each row of star.cat"""
        target_stamp = row['stamp']
        # select the references
        reference_rows = self.row_get_references(row, sim_thresh, min_nref)
        # reset and then set the list of references used
        # only reset the references that match the query
        self.row_get_references(row, -1)['used'] = False
        self.references.loc[reference_rows.index, 'used'] = True

        # pull out the stamps
        reference_stamps = reference_rows['stamp']

        target_stamp = target_stamp - target_stamp.min()
        reference_stamps = reference_stamps.apply(lambda ref: ref - ref.min())
        scale = target_stamp.max() / reference_stamps.apply(np.max)
        reference_stamps = reference_stamps * scale#.apply(lambda ref: ref / ref.max())
        kl_sub_img, klip_model_img = subutils.klip_subtract(
            target_stamp,
            reference_stamps,
            np.arange(1, reference_stamps.size)
        )
        # return each as an entry in a series. this allows it to be
        # automatically merged with self.cat
        return pd.Series({s.name: s for s in [klip_model_img, kl_sub_img]})

    def row_make_snr_map(self, row):
        resids = row['kl_sub']
        std_maps = row['kl_sub'].apply(lambda img: sigma_clipped_stats(img)[-1])
        snr_maps = resids/std_maps
        return pd.Series({'snrmap': snr_maps})

    def row_convolve_psf(self, row):
        df = pd.DataFrame(row[['klip_model', 'kl_sub']].to_dict())
        detmaps = df.apply(
            lambda dfrow : detutils.apply_matched_filter(dfrow['kl_sub'], dfrow['klip_model']),
            axis=1
        )
        center = int(np.floor(self.stamp_size/2))
        primary_fluxes = df.apply(
            lambda dfrow : detutils.apply_matched_filter(
                row['stamp'],
                dfrow['klip_model']
            )[center, center],
            axis=1
        )
        detmaps = detmaps/primary_fluxes

        return pd.Series({'detmap': detmaps})


def process_stars(
        input_catalog : pd.DataFrame,
        star_id_column : str,
        match_references_on : str | list,
        data_folder : str | Path,
        stamp_size : int = 11,
        bad_references : list = [],
        min_nref : int = 2,
        sim_thresh : float = 0.5,
        scale_stamps : bool = False,
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
    min_nref : int = 2
      Use at least this many reference stamps, regardless of similarity score
    sim_thresh : float = 0.5
      A stamp's similarity score must be at least this value to be included
      If fewer than `min_nref` reference stamps meet this criteria, use the
      `min_nref` ones with the highest similarity scores    
    scale_stamps : bool = False
      If True, scale all stamps from 0 to 1

    Output
    ------
    stars : pd.Series
      A series where each entry is a Star object with the data and analysis results

    """
    # initialize the catalog
    stars = initialize_stars(
        input_catalog,
        star_id_column,
        match_references_on,
        data_folder,
        stamp_size,
        bad_references,
        scale_stamps,
    )
    # perform PSF subtraction
    subtract_all_stars(
        stars,
        sim_thresh=sim_thresh,
        min_nref=min_nref
    )
    # perform the detection analysis
    detect_all_stars(stars)
    return stars


def initialize_stars(
        input_catalog : pd.DataFrame,
        star_id_column : str,
        match_references_on : str | list[str],
        data_folder : str | Path,
        stamp_size : int = 15,
        bad_references : str | list[str] = [],
        scale_stamps : bool = False,
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
        lambda group: Star(
            group.name,
            group,
            data_folder = data_folder,
            stamp_size = stamp_size,
            match_by = match_references_on,
            scale_stamps=scale_stamps,
        ),
        include_groups=False
    )
    # flag the bad references
    for br in bad_references:
        stars[br].is_good_reference = False
    # assign references and compute similarity score
    for star in stars:
        star.set_references(stars, compute_similarity=True)
    return stars


def subtract_all_stars(
        all_stars : pd.Series,
        sim_thresh : float = 0.5,
        min_nref : int = 2,
) -> None:
    """
    Perform PSF subtraction on all the stars, setting attributes in-place

    Parameters
    ----------
    all_stars : pd.Series
      pandas Series where each entry is a starclass.Star object, and the index is the star identifier
    min_nref : int = 2
      Use at least this many reference stamps, regardless of similarity score
    sim_thresh : float = 0.5
      A stamp's similarity score must be at least this value to be included
      If fewer than `min_nref` reference stamps meet this criteria, use the
      `min_nref` ones with the highest similarity scores

    """
    print(f"Subtracting with similarity score threshold: sim >= {sim_thresh}")
    for star in all_stars:
        # gather subtraction results
        star.subtraction = star.cat.apply(
            star.row_klip_subtract,
            sim_thresh=sim_thresh,
            min_nref=min_nref,
            axis=1
        )
        star.results = star.cat.join(star.subtraction)
    return


def detect_all_stars(
        all_stars : pd.Series,
) -> None:
    """
    Perform MF detection on all the stars

    Parameters
    ----------
    all_stars : pd.Series
      pandas Series where each entry is a starclass.Star object, and the index is the star identifier
    """
    for star in all_stars:
        # gather subtraction results
        star.results = star.results.join(
            star.results.apply(
                star.row_convolve_psf,
                axis=1
            )
        )
        star.results = star.results.join(
            star.results.apply(
                star.row_make_snr_map,
                axis=1
            )
        )
    return
