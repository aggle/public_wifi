import abc
from pathlib import Path

import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from pyklip import klip

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
    ) -> None:
        """
        Initialize a Star object from a row of the catalog

        Parameters
        ----------
        match_by : list[str] = ['filter']
          a list of columns to use for matching targets with references. e.g. which filter
        """
        self.star_id = star_id
        self.is_good_reference = True # assumed True
        self.cat = group#.reset_index(drop=True)
        # values that that need setter and getter methods
        self.has_companions = False
        # values that are initialized by methods
        self.cat['stamp'] = self.cat.apply(
            lambda row: self.get_stamp(row, stamp_size, data_folder),
            axis=1,
        )
        self.match_by = match_by
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

    def get_stamp(
            self,
            row,
            stamp_size : int,
            data_folder : str | Path
    ) -> Cutout2D:
        """
        Cut out the stamp from the data
        """
        filepath = data_folder / row['file']
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
            target_stamp = row['stamp'].data
            query = self.generate_match_query(row)

            sim = self.references.query(query)['stamp'].apply(
                lambda ref_stamp: ssim(
                    ref_stamp.data,
                    target_stamp,
                    data_range=np.ptp(np.stack([ref_stamp.data, target_stamp]))
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
        target_stamp = row['stamp'].data
        # select the references
        reference_rows = self.row_get_references(row, sim_thresh, min_nref)
        # reset and then set the list of references used
        # only reset the references that match the query
        self.row_get_references(row, -1)['used'] = False
        self.references.loc[reference_rows.index, 'used'] = True

        # pull out the stamps
        reference_stamps = reference_rows['stamp'].apply(lambda ref: ref.data)

        target_stamp = target_stamp - target_stamp.min()
        reference_stamps = reference_stamps.apply(lambda ref: ref - ref.min())
        scale = target_stamp.max() / reference_stamps.apply(np.max)
        reference_stamps = reference_stamps * scale#.apply(lambda ref: ref / ref.max())
        kl_sub_img, kl_basis_img, psf_model_img = klip_subtract(
            target_stamp,
            reference_stamps,
            np.arange(1, reference_stamps.size)
        )
        # return each as an entry in a series. this allows it to be
        # automatically merged with self.cat
        return pd.Series({s.name: s for s in [kl_basis_img, psf_model_img, kl_sub_img]})

    def make_detection_map(self):
        pass

def process_stars(
        input_catalog : pd.DataFrame,
        star_id_column : str,
        match_references_on : str | list,
        data_folder : str | Path,
        stamp_size : int = 11,
        min_nref : int = 2,
        sim_thresh : float = 0.5,
) -> None :
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

    Output
    ------
    stars : pd.Series
      A series where each entry is a Star object with the data and analysis results

    """
    if isinstance(match_references_on, str):
        match_references_on = [match_references_on]
    # Create the Star objects from the catalog
    stars = input_catalog.groupby(star_id_column).apply(
        lambda group: Star(
            group.name,
            group,
            data_folder = data_folder,
            stamp_size = stamp_size,
            match_by = match_references_on,
        ),
        include_groups=False
    )
    # assign references and compute similarity score
    for star in stars:
        star.set_references(stars, compute_similarity=True)
    subtract_all_stars(stars, sim_thresh=sim_thresh, min_nref=min_nref)
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


### PSF subtraction ###
def klip_subtract(
        target_stamp,
        reference_stamps,
        numbasis = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Perform KLIP subtraction on the target stamp.
    Returns the KL basis vectors, the subtracted images, and the PSF models
    """
    stamp_shape = target_stamp.shape
    if numbasis is None:
        numbasis = np.array([len(reference_stamps)-1])
    targ_stamp_flat = target_stamp.ravel()
    ref_stamps_flat = np.stack([i.ravel() for i in reference_stamps])

    kl_sub, kl_basis = klip.klip_math(
        targ_stamp_flat, ref_stamps_flat,
        numbasis = numbasis,
        return_basis = True,
    )
    # construct the PSF model
    coeffs = np.inner(targ_stamp_flat, kl_basis)
    psf_model = kl_basis * np.expand_dims(coeffs, [i+1 for i in range(kl_basis.ndim-1)])
    psf_model = np.array([np.sum(psf_model[:k], axis=0) for k in numbasis])

    # store as Series objects
    if isinstance(numbasis, int):
        numbasis = [numbasis]
    kl_basis = pd.Series(dict(zip(range(1, len(kl_basis)+1), kl_basis)), name='kl_basis')
    kl_basis.index.name = 'numbasis'
    kl_sub = pd.Series(dict(zip(numbasis, kl_sub.T)), name='kl_sub')
    kl_sub.index.name = 'numbasis'
    psf_model = pd.Series(dict(zip(numbasis, psf_model)), name='psf_model')
    psf_model.index.name = 'numbasis'
    # return the subtracted stamps as images
    kl_sub_img = kl_sub.apply(lambda img: img.reshape(stamp_shape))
    kl_basis_img = kl_basis.apply(lambda img: img.reshape(stamp_shape))
    psf_model_img = psf_model.apply(lambda img: img.reshape(stamp_shape))
    return kl_sub_img, kl_basis_img, psf_model_img


def make_matched_filter(stamp, width : int | None = None):
    center = np.floor(np.array(stamp.shape)/2).astype(int)
    if isinstance(width, int):
        stamp = Cutout2D(stamp, center[::-1], width).data
    stamp = np.ma.masked_array(stamp, mask=np.isnan(stamp))
    stamp = stamp - np.min(stamp)
    stamp = stamp/np.sum(stamp)
    stamp = stamp - np.min(stamp)
    return stamp.data


def apply_matched_filter(
        target_stamp : np.ndarray,
        matched_filter : np.ndarray,
) -> np.ndarray:
    pass
