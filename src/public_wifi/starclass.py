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
        self.match_by = match_by
        self.cat = group.sort_values(by=self.match_by)
        # status flags
        self.is_good_reference = True # assumed True
        self.has_candidates = False
        # values that are initialized by methods
        self.cat['cutout'] = self.cat.apply(
            lambda row: self.get_cutout(row, stamp_size),
            axis=1,
        )
        # measure the background
        self.cat['bgnd'] = self.measure_bgnd(51, 20)
        # cut out the stamps and subtract the background
        self.cat['stamp'] = self.cat.apply(
            lambda row: row['cutout'].data.copy() - row['bgnd'][0],
            axis=1
        )
        if scale_stamps:
            self.cat['stamp'] = self.cat['stamp'].apply(self.scale_stamp)
        return

    # has_candidates should always be the opposite of is_good_reference
    @property
    def has_candidates(self):
        return self._has_candidates
    @has_candidates.setter
    def has_candidates(self, new_val : bool):
        self._has_candidates = new_val
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
        if self.has_candidates == False:
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
        self.nrefs = self.references.groupby(
            self.match_by).apply(
                len, include_groups=False,
            )
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
            # print(f"Warning: {self.star_id} has fewer than {min_nref} references above threshold!")
            nrefs = min_nref
        reference_rows = reference_rows[:nrefs]
        return reference_rows

    def update_nrefs(self):
        """Update the number of refs used based on the `used` flag"""
        self.nrefs = self.references.query("used == True").groupby(
            self.match_by).apply(
                len,
                include_groups=False,
            ).reset_index(name='Nrefs')

    def row_set_reference_status(self, row, used_reference_rows):
        """
        Flag the reference rows in the index provided as used; flag all others as false
        """
        self.row_get_references(row, -1)['used'] = False
        self.references.loc[used_reference_rows.index, 'used'] = True
        # update self.nrefs, the number of refs for each set
        self.update_nrefs()

    def row_klip_subtract(self, row, sim_thresh=0.0, min_nref=2, jackknife_reference : str = ''):
        """
        Wrapper for KLIP that can be applied on each row of star.cat
        row : star.cat row
        sim_thresh : float = 0.0
          minimum similarity score to be included
        min_nref : int = 2
          flag at least this many refs as OK to use, in order of similarity score
        jackknife_reference : str = ''
          during jackknife testing, exclude this reference
        """
        target_stamp = row['stamp']
        # select the references
        reference_rows = self.row_get_references(row, sim_thresh, min_nref)
        # reset the list of used references, and then flag the references that
        # are selected. reset the references that match the query
        self.row_set_reference_status(row, reference_rows)
        # if performing a jackknife analysis, remove the reference
        reference_stamps = reference_rows.query(f"target != '{jackknife_reference}'")['stamp']

        target_stamp = target_stamp - target_stamp.min()
        reference_stamps = reference_stamps.apply(lambda ref: ref - ref.min())
        scale = target_stamp.max() / reference_stamps.apply(np.max)
        reference_stamps = reference_stamps * scale#.apply(lambda ref: ref / ref.max())
        klip_basis_img, klip_sub_img, klip_model_img = subutils.klip_subtract(
            target_stamp,
            reference_stamps,
            np.arange(1, reference_stamps.size)
        )
        # return each as an entry in a series. this allows it to be
        # automatically merged with self.cat
        return pd.Series({s.name: s for s in [klip_basis_img, klip_model_img, klip_sub_img]})

    def row_make_snr_map(self, row):
        # resids = row['klip_sub']
        # std_maps = row['klip_sub'].apply(lambda img: sigma_clipped_stats(img)[-1])
        # snr_maps = resids/std_maps
        snr_maps = detutils.make_series_snrmaps(row['klip_sub'])
        return pd.Series({'snrmap': snr_maps})

    def row_convolve_psf(self, row):
        df = pd.DataFrame(row[['klip_model', 'klip_sub']].to_dict())
        detmaps = df.apply(
            lambda dfrow : detutils.apply_matched_filter(dfrow['klip_sub'], dfrow['klip_model']),
            axis=1
        )
        # center = int(np.floor(self.stamp_size/2))
        # primary_fluxes = df.apply(
        #     lambda dfrow : detutils.apply_matched_filter(
        #         row['stamp'],
        #         dfrow['klip_model'],
        #         correlate_mode='valid',
        #     )[0, 0],
        #     axis=1
        # )
        # detmaps = detmaps/primary_fluxes
        return pd.Series({'detmap': detmaps})
    def row_detect_snrmap_candidates(self, row, snr_thresh=3, n_modes=3):
        candidates = detutils.detect_snrmap(
            row['snrmap'],
            snr_thresh=snr_thresh,
            n_modes=n_modes
        )
        if candidates is None:
            candidates = pd.DataFrame(None, columns=['cand_id', 'pixel']
            )
        return pd.Series({'snr_candidates': candidates})

    def jackknife_analysis(self, sim_thresh, min_nref):
        """Perform jackknife analysis"""
        jackknife = detutils.jackknife_analysis(self, sim_thresh, min_nref)
        return jackknife
        

