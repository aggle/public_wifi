"""
These tools use the whole catalog to measure detections
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore, shapiro
from astropy.stats import sigma_clipped_stats

from public_wifi import starclass
from public_wifi import contrast_utils as cutils
from public_wifi import detection_utils as dutils
from public_wifi import matched_filter_utils as mf_utils

def make_cdf(x):
    """Shorthand for making a CDF/EDF of all the values in an n-dimensional array"""
    return np.sort(x.ravel())

def gaussian_cdf(mu=0, sigma=1, n=1000):
    return make_cdf(np.random.normal(mu, sigma, n))


def normalize_array(x):
    """Normalize an array with sigma-clipped stats"""
    mean, _, std = sigma_clipped_stats(x)
    return (x-mean)/std

def compute_pixelwise_norm(stack: pd.Series):
    """Normalize a series of images pixelwise"""
    stamp_shape = np.stack(stack.values).shape[-2:]
    normed_stack = stack.apply(
        # this turns the series into a dataframe, where each column is one pixel
        lambda r: pd.Series(r.ravel())
    ).apply(
        # normalize down the columns
        normalize_array
    ).apply(
        # reshape the images
        lambda row: np.reshape(row, stamp_shape),
        axis=1
    )
    return normed_stack


class CatDet:
    def __init__(self, stars : pd.Series, stamp_size, kklip, mf_width):
        """A class that uses the whole catalog to measure detections"""
        # set some constants
        self.stamp_size = stamp_size
        # setting these variables will trigger recalculation of everything, so
        # just set the internal ones
        self._kklip = kklip
        self._mf_width = mf_width

        # load the processing results
        self.stars = stars
        self.all_results = pd.concat(
            stars.apply(lambda star: star.results).to_dict(),
            names=['target', 'cat_row', 'numbasis']
        ).reorder_levels(['cat_row', 'target', 'numbasis'])
        # add the analysis columns
        self.all_results = self.normalize_residuals_and_apply_matched_filter(self.all_results)
        # filter down to just a kklip of interest, for convenience and plotting
        self.results = self.filter_kklip(self.all_results, self.kklip)
        # the following should all change when you update kklip
        # self.kklip_resid = self.residuals.map(lambda r: r.loc[self.kklip])
        # self.kklip_resid = self.filter_kklip(self.results, self.kklip)
        # self.kklip_resid_norm = self.kklip_resid.map(normalize_array)
        self.detection_maps = self.generate_pixelwise_snrmap('detmap_norm')
        self.residual_snr_maps = self.generate_pixelwise_snrmap('klip_sub_norm')

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
    def filter_kklip(self, df, kklip):
        return df.query(f"numbasis == {kklip}")

    def recompute(self):
        self.results = self.filter_kklip(self.all_results, self.kklip)
        # self.kklip_resid_norm = self.kklip_resid.map(normalize_array)
        self.detection_maps = self.generate_pixelwise_snrmap('detmap_norm')
        self.residual_snr_maps = self.generate_pixelwise_snrmap('klip_sub_norm')

    def normalize_residuals_and_apply_matched_filter(self, results):
        # normalize the arrays so we can compare them against each other
        results['klip_sub_norm'] = results['klip_sub'].map(normalize_array)
        # apply the matched filters to the normalized residual stamps
        results['mf_map_norm'] = results.apply(
            lambda row:  mf_utils.correlate(
                    row['klip_sub_norm'],
                    row['mf'],
                    mode='same',
                    method='auto'
                ),
                axis=1
        )
        # apply the proper normalization to the matched filter so the scales
        # can be compared
        results['detmap_norm'] = results['mf_map_norm']/results['mf_norm']
        return results

    # def generate_matched_filters(self):
    #     """Generate matched filters for each residual stamp. Not currently used."""
    #     matched_filters = pd.concat({
    #             col: self.kklip_resid_norm.apply(
    #                 lambda row: mf_utils.make_matched_filter(
    #                     self.stars[row.name].results.loc[col, 'klip_model'].loc[self.kklip],
    #                     self.mf_width
    #                 ),
    #                 axis=1
    #             )
    #             for col in self.kklip_resid_norm.to_dict()
    #         }, axis=1)
    #     return matched_filters

    # def apply_mf(self):
    #     """Apply matched filters to the normalized klip residuals"""
    #     mf_detect = pd.concat({
    #         col: self.kklip_resid_norm.apply(
    #             lambda row: mf_utils.apply_matched_filter_to_stamp(
    #                 row[col],
    #                 self.stars[row.name].results.loc[col, 'klip_model'].loc[self.kklip],
    #                 mf_width=self.mf_width,
    #                 kl_basis=None
    #             ),
    #             axis=1
    #         )
    #         for col in self.kklip_resid_norm.to_dict()
    #     }, axis=1)
    #     return mf_detect

    def generate_pixelwise_snrmap(self, results_column='klip_sub_norm'):
        """
        Compute the pixelwise SNR by comparing the pixels at a single coordinate from different stamps,
        then transform back to the stamp for the target.
        In the final image, the value at each pixel represents the pixel's SNR
        in a different distribution from all the other pixels.
        """
        snrmap = self.results[results_column].groupby(['cat_row', 'numbasis'], group_keys=False).apply(
            compute_pixelwise_norm
        )
        # Kklip is redundant so let's remove it from the index
        snrmap.index = snrmap.index.droplevel("numbasis")
        return snrmap



def snr_vs_catalog(
        star : starclass.Star,
        catalog : pd.Series,
        kklip : int = 10,
        col : str = 'detmap'
) -> np.ndarray:
    """
    Compute the pixel-wise SNR of a stamp against the catalog.

    Parameters
    ----------
    star : starclass.Star
      The test star
    catalog : pd.Series
      the other stars (if `star` is in this series, it will be excluded from the analysis)
    kklip : int = 10
      The Kklip index to compare
    col : str = 'detmap'
      which column of the star.results dataframe to use

    Output
    ------
    snr_map : np.array
      A map of each pixel's SNR measured against that pixel in the catalog
    """
    pass # insert body here

