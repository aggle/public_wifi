from pathlib import Path

import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from public_wifi import misc
from public_wifi import centroid
from public_wifi import subtraction_utils as subtr_utils
from public_wifi import detection_utils as dutils
from public_wifi import matched_filter_utils as mf_utils
from public_wifi import contrast_utils as cutils

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
    def __repr__(self):
        return f"Star {self.star_id}"

    def __init__(
            self,
            star_id : str,
            group : pd.DataFrame,
            data_folder : str | Path,
            stamp_size : int  = 15,
            match_by : list[str] = ['filter'],
            scale_stamps = False,
            subtr_args : dict = {},
            det_args : dict = {},
            center_stamps : bool = False,
    ) -> None:
        """
        Initialize a Star object from a row of the catalog

        Parameters
        ----------
        star_id : str,
          unique star identifier
        group : pd.DataFrame
          the catalog entries for this star
        data_folder : str | Path
          parent folder where the exposure files are stored
        stamp_size : int  = 15
          stamp size to cut out for analysis
        match_by : list[str] = ['filter']
          a list of columns to use for matching targets with references. e.g.
          which filter
        scale_stamps = False
          if True, scale all stamps from 0 to 1
        subtr_args : dict = {}
          arguments passed to the PSF modeling and subtraction algorithm
        det_args : dict = {}
          arguments passed to the candidate detection algorithm
        center_stamps : bool = False
          If True, use scipy.ndimage.shift to shift PSFs to the center

        """
        self.star_id = star_id
        self.stamp_size = stamp_size
        self.data_folder = data_folder
        self.match_by = match_by
        self.cat = group.sort_values(by=self.match_by)
        # status flags
        self.is_good_reference = True # assumed True
        self.has_candidates = False
        # processing function parameters
        self.subtr_args = subtr_args
        self.det_args = det_args
        # values that are initialized by methods
        self._cutout_pad = 2
        self.cat['cutout'] = self.cat.apply(
            lambda row: self._get_cutout(row, stamp_size, pad=self._cutout_pad),
            axis=1,
        )
        # measure the background
        self.cat['bgnd'] = self.measure_bgnd(51, 20)
        # cut out the stamps and subtract the background
        self.cat['stamp'] = self.cat['cutout'].apply(
            self._get_stamp_from_cutout,
            pad=self._cutout_pad
        )
        bgnds = self.cat['bgnd'].apply(lambda bgnd: bgnd[0])
        self.cat['stamp'] = self.cat['stamp'] - bgnds
        # stamp manipulation
        if center_stamps:
            self.cat['stamp'] = self.cat['stamp'].apply(misc.shift_stamp_to_center)
        if scale_stamps:
            self.cat['stamp'] = self.cat['stamp'].apply(misc.scale_stamp)

        # this will hold the analysis results
        self.results = pd.DataFrame()

        return

    # has_candidates should always be the opposite of is_good_reference
    @property
    def has_candidates(self):
        return self._has_candidates
    @has_candidates.setter
    def has_candidates(self, new_val : bool):
        self._has_candidates = new_val
        # After the change in state, check and set, if necessary, the reference status
        self._check_reference()

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

    def _check_reference(self):
        """This method checks for all the conditions"""
        is_ok = False
        if self.has_candidates == False:
            is_ok = True
        self.is_good_reference = is_ok

    def _get_cutout(
            self,
            row,
            stamp_size : int,
            ext : int | str = 'SCI',
            pad = 0
    ) -> Cutout2D:
        """
        Cut out the stamp from the data

        Parameters
        ----------
        pad : int
          How much padding in each dimension to give the cutout, i.e. if the
          stamp size is 11x11, a cutout with padding of 2 will be 15x15
        """
        filepath = self.data_folder / row['file']
        img = fits.getdata(str(filepath), ext)
        wcs = WCS(fits.getheader(str(filepath), ext))
        padded_size = stamp_size + pad*2
        cutout = Cutout2D(
            img,
            (row['x'],row['y']),
            size=padded_size,
            wcs=wcs,
            mode='trim',
            fill_value = np.nan,
            copy=True
        )
        # recenter on the brightest pixel
        maxpix = np.array(
            np.unravel_index(np.nanargmax(cutout.data), cutout.shape)[::-1]
        ) + np.array(cutout.origin_original)
        # # actually, recenter on the centroid
        # # maxpix = misc.compute_psf_center(cutout.data, pad=5) +\
        # center = misc.get_stamp_center(cutout.data)
        # maxpix = centroid.compute_centroid(
        #     cutout.data,
        #     incoord = tuple(center),
        #     silent = True,
        # ) + np.array(cutout.origin_original)
        # print(maxpix, tuple(row[['x','y']]))

        cutout = Cutout2D(
            img,
            tuple(maxpix),
            size=padded_size,
            wcs=wcs,
            mode='trim',
            fill_value = np.nan,
            copy=True
        )

        return cutout
    def _get_stamp_from_cutout(self, cutout : Cutout2D, pad=2) -> np.ndarray:
        data = cutout.data.copy()
        if pad == 0:
            stamp = data.copy()
        else:
            stamp = data[pad:-pad, pad:-pad].copy()
        return stamp

    def measure_bgnd(self, stamp_size=51, bgnd_rad=20) -> float:
        """
        stamp_size = 51
          the size of the stamp to use for measureing the background
        bgnd_rad : 20
            pixels further from the center than this are used to measure the bgnd 
        """
        bgnd_stamps = self.cat.apply(
            self._get_cutout,
            stamp_size=stamp_size,
            pad=0,
            axis=1
        )
        center = int(np.floor(self.stamp_size/2))
        sep_map = np.linalg.norm(np.mgrid[:stamp_size, :stamp_size] - center, axis=0)
        bgnd_mask = sep_map < bgnd_rad
        bgnd = bgnd_stamps.apply(
            lambda stamp: (
                np.nanmean(stamp.data[bgnd_mask]),
                np.nanstd(stamp.data[bgnd_mask])
            )
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
        references = pd.concat(references, names=['target', 'cat_row'])
        references.index = references.index.reorder_levels(['cat_row', 'target'])
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

    def _row_get_references(self, row, sim_thresh=0.0, min_nref=2):
        """Get the references associated with a row entry"""
        query = self.generate_match_query(row)
        reference_rows = self.references.query(query).sort_values(
            by='sim', ascending=False
        )
        # select the refs above threshold, or at least the top 5
        nrefs = len(reference_rows.query(f"sim >= {sim_thresh}"))
        # update nrefs if it is too small
        if nrefs <= min_nref:
            # print(f"Warning: {self.star_id} has fewer than {min_nref}
            # references above threshold!")
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
        self._row_get_references(row, -1)['used'] = False
        self.references.loc[used_reference_rows.index, 'used'] = True
        # update self.nrefs, the number of refs for each set
        self.update_nrefs()

    def run_klip_subtraction(
            self,
            sim_thresh : float | None = None,
            min_nref : int | None = None,
            stamp_column : str = 'stamp',
            jackknife_reference : str = ''
    ):
        """
        sim_thresh : float
          image similarity score threshold
        min_nref : int
          Include at least this many reference PSFs, ordered by similarity
        stamp_column : str = 'stamp'
          which column to pull the stamp from
        """
        if sim_thresh is None:
            sim_thresh = self.subtr_args['sim_thresh']
        else:
            self.subtr_args.update({'sim_thresh': sim_thresh})
        if min_nref is None:
            min_nref = self.subtr_args['min_nref']
        else:
            self.subtr_args.update({'min_nref': min_nref})

        self.subtr_args.update(dict(
            sim_thresh=sim_thresh,
            min_nref=min_nref,
        ))
        # if the subtraction parameters are not provided, read them from the class attr
        sim_thresh = self.subtr_args['sim_thresh']
        min_nref = self.subtr_args['min_nref']
        results = self.cat.apply(
            self._row_klip_subtract,
            sim_thresh = sim_thresh,
            min_nref = min_nref,
            stamp_column = stamp_column,
            jackknife_reference = jackknife_reference,
            axis=1
        )
        results = pd.concat(results.to_dict(), names=['cat_row', 'numbasis'])
        # results = combine_pca_results(results)
        # results = self.cat.join(self.subtraction)
        return results



    def _row_klip_subtract(
            self,
            row,
            sim_thresh,
            min_nref,
            stamp_column : str = 'stamp',
            jackknife_reference : str = '',
    ) -> pd.DataFrame:
        """
        Wrapper for KLIP that can be applied on each row of star.cat
        row : star.cat row
        sim_thresh : float | None
          minimum similarity score to be included
          If None, read from self.subtr_args
        min_nref : int | None
          flag at least this many refs as OK to use, in order of similarity score
          If None, read from self.subtr_args
        stamp_column : str
          Which cat column to use for the stamp (e.g. with fake injections)
        jackknife_reference : str = ''
          during jackknife testing, exclude this reference
        """

        target_stamp = row[stamp_column]
        # select the references
        reference_rows = self._row_get_references(row, sim_thresh, min_nref)
        # reset the list of used references, and then flag the references that
        # are selected. reset the references that match the query
        self.row_set_reference_status(row, reference_rows)
        # if performing a jackknife analysis, remove the reference
        reference_stamps = reference_rows.query(f"target != '{jackknife_reference}'")['stamp']

        target_stamp = target_stamp - target_stamp.min()
        reference_stamps = reference_stamps.apply(lambda ref: ref - ref.min())
        scale = target_stamp.max() / reference_stamps.apply(np.max)
        reference_stamps = reference_stamps * scale#.apply(lambda ref: ref / ref.max())
        results = subtr_utils.klip_subtract(
            target_stamp,
            reference_stamps,
            np.arange(1, reference_stamps.size)
        )
        # return each as an entry in a series. this allows it to be
        # automatically merged with self.cat
        # return pd.Series(
        #     {s.name: s for s in [klip_basis_img, klip_model_img, klip_sub_img]}
        # )
        return results

    def run_make_snr_maps(self, results=None):
        """
        Divide the residual stamps by their standard deviation
        results : None
          You can pass in your own results dataframe. If you don't, it uses the
          star's.
        """
        if results is None:
            results = self.results
        # group by the catalog row and compute the SNR map of the residuals
        gb_cat = self.results.groupby("cat_row", group_keys=False)
        snrmaps = gb_cat['klip_sub'].apply(
            dutils.make_series_snrmaps
        )
        snrmaps.name = 'snrmap'
        return snrmaps

    def _row_apply_matched_filter(
            self, row, mf_width = None, contrast=True, throughput_correction=True
    ):
        """
        Convolve a matched filter against a residual stamp for a single row of the results dataframe

        contrast : bool = False
          If true, convolve the model with the stamp and divide bu the flux
        throughput_correction : bool = True
          If True, correct for the KLIP throughput
        """
        # this dataframe is indexed by Kklip
        df = pd.concat(row[['klip_model', 'klip_sub', 'klip_basis']].to_dict(), axis=1)

        kl_basis = df['klip_basis']
        detmaps = df.apply(
            # each row is a set of corresponding Kklips
            lambda kklip_row : mf_utils.apply_matched_filter_to_stamp(
                kklip_row['klip_sub'],
                kklip_row['klip_model'],
                mf_width = mf_width,
                correlate_mode='same',
                kl_basis = kl_basis[:kklip_row.name] if throughput_correction else None,
            ),
            axis=1
        )
        if contrast:
            primary_fluxes = df.apply(
                lambda dfrow: cutils.measure_primary_flux(
                    self.cat.loc[row.name, 'stamp'],
                    dfrow['klip_model'],
                ),
                axis=1
            )
            detmaps = detmaps/primary_fluxes
        return pd.Series({'detmap': detmaps})

    def apply_matched_filter(
        self, mf_width=None, contrast=True, throughput_correction=True
    ):
        """Wrapper for row_apply_matched_filter for the entire results dataframe"""
        detmaps = self.results.apply(
            self._row_apply_matched_filter,
            mf_width=mf_width,
            contrast=contrast,
            throughput_correction=throughput_correction,
            axis=1
        ).squeeze()
        return detmaps

    def _row_make_mf_flux_map(
            self, row, contrast=True,
    ) -> pd.Series:
        """
        Apply a matched filter to the PSF subtraction residuals and correct for
        the PSF subtraction throughput

        Parameters
        ----------
        row : pd.Series
          A row of the self.results DataFrame
        contrast : bool = True
          If True, apply the MF to the primary star, and divide the flux map by the result.

        Output
        ------
        flux_map : pd.Series
          Residual maps in units of flux or contrast, indexed by mode

        """
        df = pd.DataFrame(row[['klip_model', 'klip_sub', 'klip_basis']].to_dict())
        df['matched_filter'] = df['klip_model'].apply(
            mf_utils.make_matched_filter,
            width=7,
        )
        detmaps = df.apply(
            lambda dfrow : mf_utils.apply_matched_filter_to_stamp(
                dfrow['klip_sub'],
                dfrow['klip_model'],
                mf_width = min(7, self.stamp_size),
                kl_basis = None,
            ),
            axis=1
        )
        thpt = df.apply(
            lambda dfrow: mf_utils.compute_throughput(
                dfrow['matched_filter'],
                df['klip_basis'][:dfrow.name],
            ).iloc[-1],
            axis=1
        )
        fluxmaps = detmaps/thpt
        if contrast:
            center = int(np.floor(self.stamp_size/2))
            primary_fluxes = df.apply(
                lambda dfrow : mf_utils.apply_matched_filter_to_stamp(
                    row['stamp'],
                    dfrow['klip_model'],
                    correlate_mode='same',
                )[center, center],
                axis=1
            )
            fluxmaps = fluxmaps/primary_fluxes
        return pd.Series({'fluxmap': fluxmaps})

    def run_make_mf_flux_maps(self, contrast=True):
        "Wrapper for _row_make_mf_flux_map on self.results"
        fluxmaps = self.results.apply(
            self._row_make_mf_flux_map,
            contrast=contrast,
            axis=1
        ).squeeze()
        return fluxmaps

    def detect_snrmap_candidates(self, results):
        try:
            snr_thresh = self.det_args.get('snr_thresh', 5.0)
            n_modes = self.det_args.get('n_modes', 3)
        except KeyError as e:
            print(f"Error: self.det_args probably not set")
            raise e
        gb_cat = results.groupby("cat_row", group_keys=False)['snrmap']
        candidates = gb_cat.apply(
            dutils.detect_snrmap,
            snr_thresh=snr_thresh,
            n_modes=n_modes,
        )
        candidates.name = 'snr_candidates'
        return candidates

    # def row_detect_snrmap_candidates(
    #         self,
    #         row,
    # ):
    #     # if the subtraction parameters are not provided, read them from the class attr
    #     try:
    #         snr_thresh = self.det_args.get('snr_thresh', 5.0)
    #         n_modes = self.det_args.get('n_modes', 3)
    #     except KeyError as e:
    #         print(f"Error: self.det_args probably not set")
    #         raise e

    #     candidates = dutils.detect_snrmap(
    #         row['snrmap'],
    #         snr_thresh=snr_thresh,
    #         n_modes=n_modes
    #     )
    #     if candidates is None:
    #         candidates = pd.DataFrame(None, columns=['cand_id', 'pixel'])
    #     return pd.Series({'snr_candidates': candidates})

    def jackknife_analysis(
            self,
            sim_thresh : float | None = None,
            min_nref : int | None = None,
    ) -> pd.Series:
        """
        Perform jackknife analysis
        returns a cube of the jackknife analysis
        """
        if sim_thresh is None:
            sim_thresh = self.subtr_args.get('sim_thresh')
        else:
            self.subtr_args['sim_thresh'] = sim_thresh
        if min_nref is None:
            min_nref = self.subtr_args.get('min_nref')
        else:
            self.subtr_args['min_nref'] = min_nref
        jackknife = dutils.jackknife_analysis(self, sim_thresh, min_nref)
        jackknife_name = jackknife.name
        return jackknife



    def _row_inject_psf(self, row, pos, scale, kklip : int = -1) -> np.ndarray:
        """
        inject a PSF 
        """
        result_row = self.results.loc[row.name]
        stamp = row['stamp']
        # kklip is actually an index, not a mode number, so subtract 1 
        if kklip != -1:
            kklip -= 1
        psf_model = mf_utils.make_normalized_psf(
            result_row['klip_model'].iloc[kklip].copy(),
            7, # 7x7 psf, hard-coded
            1.,  # total flux of final PSF
        )
        star_flux = cutils.measure_primary_flux(stamp, psf_model)
        # compute the companion flux at the given contrast
        inj_flux = star_flux * scale
        inj_stamp = cutils.inject_psf(stamp, psf_model * inj_flux, pos)
        inj_row = row.copy()
        inj_row['stamp'] = inj_stamp
        return inj_row

    def _row_inject_subtract_detect(
            self,
            row : pd.Series,
            pos : tuple[int],
            contrast : float,
            snr_thresh : float | None = None,
            n_modes : int | None = None,
    ) -> tuple[float, bool]:
        """
        Inject, subtract, and detect fake PSFs. Uses the attribute argument parameters
        snr_thresh : the SNR threshold to declare a detection.
          If NaN, uses self.det_args. For contrast curves, provide the
          significance level of the detection you wish to report.
        """
        if snr_thresh is None:
            snr_thresh = float(self.det_args['snr_thresh'])
            print(f"Resetting snr_thresh to {snr_thresh}")
        if n_modes is None:
            n_modes = int(self.det_args['n_modes'])
            print(f"Resetting n_modes to {n_modes}")

        inj_row = self._row_inject_psf(row, pos=pos, scale=contrast, kklip=-1)
        results = self._row_klip_subtract(
            inj_row, **self.subtr_args,
        )
        snrmaps = self._row_make_snr_map(results).squeeze()
        # recover the SNR at the injected position
        center = misc.get_stamp_center(self.stamp_size)
        # inj_pos = center + np.array(pos)
        inj_pos = misc.center_to_ll_coords(center, pos)
        # inj_snr = np.stack(snrmaps.values)[..., inj_pos[1], inj_pos[0]]
        # sometimes the recovered position is off from the injection site by 1
        inj_snr = snrmaps.apply(
            lambda img: img[*cutils.find_injection(img, pos)]
        )
        mean_snr = cutils.calc_snr_from_series(
            inj_snr, thresh=snr_thresh, n_modes=n_modes
        )
        # sort the SNR, drop the first 2 Kklips, and take the mean of the
        # highest values
        # mean_snr = np.sort(inj_snr[3:])[-n_modes:].mean()
        # get the detection flag at the detected positions
        detmap = dutils.flag_candidate_pixels(
            snrmaps,
            thresh = snr_thresh,
            n_modes = n_modes,
        )
        is_detected = detmap[*inj_pos[::-1]]
        return mean_snr, is_detected


def apply_mf_to_pca_results(
        init_pca_results : pd.DataFrame,
        mf_width : int | None = None,
        det_pos : tuple[int] | None = None,
):
    """
    Apply matched filtering to the star.pca_results dataframe
    Adds the following columns to the dataframe:
    mf : the matched filter
    mf_prim_flux : the flux of the primary as measured by that matched filter
    mf_norm : the mf norm
    pca_bias : the bias in the MF introduced by the PCA basis vectors
    mf_map : the MF correlation with the subtracted image
    detmap : the mf_map divided by the mf_norm
    fluxmap : the mf_map divided by (mf_norm - pca_bias)
    contrastmap : the fluxmap dividde by mf_prim_flux
    detpos : the brightest pixel in the detmap
    detmap_posflux : the flux of the detpos pixel in the detmap
    fluxmap_posflux : the flux of the detpos pixel in the fluxmap

    Parameters
    ----------
    init_pca_results : pd.DataFrame
      A dataframe with numbasis as the index containing columns for the basis
      vector, model, and residual
    mf_width : int | None = None
      the width of the matched filter. If None, set equal to the stamp size
    det_pos : if provided, use this position for recovering the flux
      useful for fake injection and recovery

    Output
    ------
    pca_results : pd.DataFrame
      a dataframe indexed by Kklip with the above columns added

    """
    pca_results = init_pca_results.copy()
    pca_results['mf'] = pca_results['klip_model'].apply(
        mf_utils.make_matched_filter, width=mf_width
    )
    # primary star flux as measured by the matched filter
    pca_results['mf_prim_flux'] = pca_results.apply( 
        lambda row: cutils.measure_primary_flux(row['klip_model'], row['mf']),
        axis=1
    )
    pca_results['mf_norm'] = pca_results['mf'].apply(
        mf_utils.compute_throughput, klmodes=None
    )
    pca_results['pca_bias'] = pca_results.apply(
        lambda row: mf_utils.compute_pca_bias(
            row['mf'],
            modes=pca_results.loc[:row.name, 'klip_basis']
        ).iloc[-1],
        axis=1
    )
    pca_results['mf_map'] = pca_results.apply(
        lambda row:  mf_utils.correlate(
            row['klip_sub'],
            row['mf'],
            mode='same',
            method='auto'
        ),
        axis=1
    )
    pca_results['detmap'] = pca_results['mf_map']/pca_results['mf_norm']
    thpt = (pca_results['mf_norm'] - pca_results['pca_bias'])
    pca_results['fluxmap'] = pca_results['mf_map']/thpt
    pca_results['contrastmap'] = pca_results['fluxmap']/pca_results['mf_prim_flux']

    if det_pos is None:
        pca_results['detpos'] = pca_results['detmap'].apply(
            lambda detmap: np.unravel_index(detmap.argmax(), detmap.shape)
        )
    else:
        pca_results['detpos'] = pca_results.apply(lambda row: tuple(det_pos), axis=1)

    pca_results['detmap_posflux'] = pca_results.apply(
        lambda row: row['detmap'][*row['detpos']],
        axis=1
    )
    pca_results['fluxmap_posflux'] = pca_results.apply(
        lambda row: row['fluxmap'][*row['detpos']],
        axis=1
    )
    return pca_results


def combine_pca_results(subtr_results) -> pd.DataFrame | None:
    """
    Combine the results of PSF subtraction into a dataframe indexed by the
    ordered modes
    """
    if isinstance(subtr_results, pd.DataFrame):
        pca_results = pd.concat(
            {k: pd.concat(v, axis=1) for k, v in subtr_results.T.to_dict().items()},
            axis=0
        )
    elif isinstance(subtr_results, pd.Series):
        pca_results = pd.concat(subtr_results.to_dict(), axis=1)
    else:
        pca_results = None
    return pca_results
