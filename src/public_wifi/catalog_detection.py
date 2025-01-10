"""
These tools use the whole catalog to measure detections
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore, shapiro
from astropy.stats import sigma_clipped_stats

from public_wifi import contrast_utils as cutils
from public_wifi import matched_filter_utils as mf_utils

def make_cdf(x):
    """Shorthand for making a CDF/EDF of all the values in an n-dimensional array"""
    return np.sort(x.ravel())


class CatDet:
    def __init__(self, stars : pd.Series, stamp_size, kklip, mf_width):
        """A class that uses the whole catalog to measure detections"""
        self.stars = stars
        self.residuals = stars.apply(lambda star: star.results['klip_sub'])
        self.stamp_size = stamp_size
        # setting these variables will trigger recalculation of everything, so
        # set the internal ones
        self._kklip = kklip
        self._mf_width = mf_width
        # the following should all change when you update kklip
        self.kklip_resid = self.residuals.map(lambda r: r.loc[self.kklip])
        self.kklip_resid_norm = self.kklip_resid.map(self.normalize_array)
        self.detection_maps = self.generate_mf_detection_maps()
        self.contrast_maps = self.generate_contrast_maps()


    @property
    def kklip(self):
        return self._kklip
    @kklip.setter
    def kklip(self, new_val : int):
        self._kklip = new_val
        self.recompute()
    @property
    def mf_width(self):
        return self._mf_width
    @mf_width.setter
    def mf_width(self, new_val : int):
        self._mf_width = new_val
        self.recompute()
    
    def recompute(self):
        self.kklip_resid = self.residuals.map(lambda r: r.loc[self.kklip])
        self.kklip_resid_norm = self.kklip_resid.map(self.normalize_array)
        self.detection_maps = self.generate_mf_detection_maps()
        self.contrast_maps = self.generate_contrast_maps()
    def select_resids_kklip(self):
        self.residuals.map(lambda r: r.loc[self.kklip])
    
    def normalize_array(self, x):
        """Normalize an array with sigma-clipped stats"""
        mean, _, std = sigma_clipped_stats(x)
        return (x-mean)/std

    def generate_matched_filters(self):
        matched_filters = pd.concat({
                col: self.kklip_resid_norm.apply(
                    lambda row: mf_utils.make_matched_filter(
                        self.stars[row.name].results.loc[col, 'klip_model'].loc[self.kklip], 
                        self.mf_width
                    ),
                    axis=1
                )
                for col in self.kklip_resid_norm.to_dict()
            }, axis=1)
        return matched_filters

    def generate_mf_detection_maps(self):
        # apply a matched filter, and then normalize by pixel
        mf_detect = pd.concat({
            col: self.kklip_resid_norm.apply(
                lambda row: mf_utils.apply_matched_filter(
                    row[col],
                    self.stars[row.name].results.loc[col, 'klip_model'].loc[self.kklip],
                    mf_width=self.mf_width,
                    throughput_correction=True,
                    kl_basis=None
                ),
                axis=1
            )
            for col in self.kklip_resid_norm.to_dict()
        }, axis=1)
        mf_detect_norm = pd.concat({
            col: mf_detect[col].apply(
                lambda r: pd.Series(r.ravel())
            ).apply(
                self.normalize_array
            ).apply(
                lambda row: np.reshape(row, (self.stamp_size, self.stamp_size)),
                axis=1
            )
            for col in mf_detect.to_dict()
        }, axis=1)
        return mf_detect_norm

    def generate_contrast_maps(self):
        primary_flux = self.stars.apply(
            lambda star: star.cat.apply(
                lambda row: cutils.measure_primary_flux(
                    row['stamp'],
                    star.results.loc[row.name, 'klip_model'].loc[self.kklip],
                    dict(mf_width=self.mf_width),
                ),
                axis=1
            )
        )
        mf_thpt = pd.concat({
            col: self.kklip_resid.apply(
                lambda row: mf_utils.apply_matched_filter(
                    row[col],
                    self.stars[row.name].results.loc[col, 'klip_model'].loc[self.kklip],
                    mf_width=self.mf_width,
                    throughput_correction=True,
                    kl_basis=self.stars[row.name].results.loc[col, 'klip_basis'].loc[:self.kklip]
                ),
                axis=1
            )
            for col in self.kklip_resid.to_dict()
        }, axis=1)
        mf_contrast = mf_thpt / primary_flux
        return mf_contrast
