import abc
from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from pyklip import klip

from public_wifi.utils import detection_utils

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
            stamp_size=15,
    ) -> None:
        """
        Initialize a Star object from a row of the catalog
        parameters
        ----------
        other_stars : pd.Series | None
          if a Series, the index must be the star_id
        """
        self.star_id = star_id
        self.is_good_reference = True # assumed True
        self.meta = group#.reset_index(drop=True)
        # values that that need setter and getter methods
        self.has_companions = False
        # values that are initialized by methods
        self.meta['stamp'] = self.meta.apply(
            lambda row: self.get_stamp(row, stamp_size, data_folder),
            axis=1,
        )
        return

    # has_companions should always be the opposite of is_good_reference
    @property
    def has_companions(self):
        return self._has_companions
    @has_companions.setter
    def has_companions(self, new_val : bool):
        self._has_companions = new_val
        # After the change in state, check the reference status
        self.check_reference()

    @property
    def meta(self):
        return self._meta
    @meta.setter
    def meta(self, new_val : pd.DataFrame):
        new_val = new_val.reset_index(names='stamp_id')
        self._meta = new_val.copy()



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

    def set_references(self, other_stars):
        """
        Assemble the references for each stamp. Put "good reference" checks here
        """
        # references = pd.concat(stars[stars.index != self.star_id].apply(lambda s: s.meta.copy()))
        references = {}
        for star in other_stars[other_stars.index != self.star_id]:
            if star.is_good_reference == False:
                pass
            else:
                references[star.star_id] = star.meta.copy()
        references = pd.concat(references, names=['target', 'index'])
        self.references = references

    def row_klip_subtract(self, row, numbasis=None):
        """Wrapper for KLIP that can be applied on each row of star.meta"""
        filt = row['filter']
        target_stamp = row['stamp'].data
        reference_stamps = self.references.query(f"filter == '{filt}'")['stamp'].apply(lambda ref: ref.data)
        kl_sub_img, kl_basis_img, psf_model_img = klip_subtract(
            target_stamp,
            reference_stamps,
            np.arange(1, reference_stamps.size)
        )
        # return each as an entry in a series. this allows it to be
        # automatically merged with self.meta
        return pd.Series({s.name: s for s in [kl_basis_img, psf_model_img, kl_sub_img]})

    def make_detection_map(self):
        pass


def subtract_all_stars(all_stars : pd.Series):
    """Perform PSF subtraction on all the stars, setting attributes in=place"""
    for star in all_stars:
        star.set_references(all_stars)
        # gather subtraction results
        star.subtraction = star.meta.apply(star.row_klip_subtract, axis=1)
        star.results = star.meta.join(star.subtraction)
        # star.psf_models = star.results.apply(
        #     star.row_build_psf_model,
        #     axis=1
        # )
        # star.results['psf_model'] = star.psf_models
    return


def klip_subtract(
        target_stamp,
        reference_stamps,
        numbasis=None,
) -> tuple[pd.Series, pd.Series]:
    """
    Perform KLIP subtraction on the target stamp.
    Returns the KL basis vectors, the subtracted images, and the PSF models
    """
    stamp_shape = target_stamp.shape
    if numbasis is None:
        numbasis = len(reference_stamps)-1
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
    stamp = np.ma.masked_array(stamp, mask=np.isnan(mask))
    stamp = stamp - np.min(stamp)
    stamp = stamp/np.sum(stamp)
    stamp = stamp - np.min(stamp)
    return stamp.data
