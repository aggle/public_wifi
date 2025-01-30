"""
analysis_manager.py contains a class that holds the psf-subtracted stars and the detection class that combines stars to find candidates
"""

from pathlib import Path
import pandas as pd

from public_wifi import catalog_processing as catproc
from public_wifi import catalog_detection as catdec

class AnalysisManager:
    """
    This class 
    """
    def __init__(
            self,
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
            snr_thresh = 5.,
            n_modes = 3,
            mf_width = 7,
            cat_det_kklip = 10,
    ) -> None:
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
          SNR threshold for flagging a candidate
        n_modes : int = 3
          Candidates must have SNR>thresh in at least this many modes
        mf_width : int = 7
          The size of the matched filter to use when using matched filter detection
        cat_det_kklip = 10
          Initial Kklip value for the "Catalog Detection" scroller


        Output
        ------
        stars : pd.Series
          A series where each entry is a Star object with the data and analysis
          results
        """
        self._processing_parameters = dict(
            input_catalog = input_catalog,
            star_id_column = star_id_column,
            match_references_on = match_references_on,
            data_folder = data_folder,
            bad_references = bad_references,
            stamp_size = stamp_size,
            scale_stamps = scale_stamps,
            center_stamps = center_stamps,
            # psf subtraction args
            min_nref = min_nref,
            sim_thresh = sim_thresh,
            # detection args
            snr_thresh = snr_thresh,
            n_modes = n_modes,
        )
        self._detection_parameters = dict(
            mf_width = mf_width,
            kklip = cat_det_kklip,
        )

        print("Processing catalog.")
        self.stars = catproc.process_catalog(
            **self._processing_parameters,
        )

        print("Generating catalog detection map.")
        self.det = catdec.CatDet(
            self.stars, stamp_size, **self._detection_parameters
        )
        print("Finished generating detection maps.")
        return

    def _reprocess(self):
        print("Reprocessing with new parameters.")
        self.stars = catproc.process_catalog(
            **self._processing_parameters,
        )
        self.det = catdec.CatDet(
            stars=self.stars,
            stamp_size=self._processing_parameters['stamp_size'],
            **self._detection_parameters,
        )
        print("Reprocessing complete.")
        return

    def _reprocess_detection(self):
        print("Rerunning detection with new parameters")
        self.det = catdec.CatDet(
            stars=self.stars,
            stamp_size=self._processing_parameters['stamp_size'],
            **self._detection_parameters,
        )
        print("Reprocessing complete.")
        return

    # use these methods to automatically trigger reprocessing when updating
    # parameters
    def _update_processing_parameters(self, **params):
        self._processing_parameters.update(**params)
        self._reprocess()
        return

    def _update_detection_parameters(self, **params):
        self._detection_parameters.update(**params)
        self._reprocess_detection()
        return

