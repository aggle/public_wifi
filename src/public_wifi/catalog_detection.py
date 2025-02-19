"""
These tools use the whole catalog to measure detections
"""

import numpy as np
import pandas as pd
import warnings
from public_wifi.misc import shift_stamp_to_center
from scipy.stats import zscore, shapiro
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources

from public_wifi import misc
from public_wifi import starclass
from public_wifi import contrast_utils as cutils
from public_wifi import detection_utils as dutils
from public_wifi import matched_filter_utils as mf_utils

def make_cdf(x):
    """Shorthand for making a CDF/EDF of all the values in an n-dimensional array"""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    return np.sort(x.ravel())

def gaussian_cdf(mu=0, sigma=1, n=1000):
    return make_cdf(np.random.normal(mu, sigma, n))


# def compute_pixelwise_norm(stack: pd.Series):
#     """Normalize a series of images pixelwise"""
#     stamp_shape = np.stack(stack.values).shape[-2:]
#     normed_stack = stack.apply(
#         # this turns the series into a dataframe, where each column is one pixel
#         lambda r: pd.Series(r.ravel())
#     ).apply(
#         # normalize down the columns
#         normalize_array
#     ).apply(
#         # reshape the images
#         lambda row: np.reshape(row, stamp_shape),
#         axis=1
#     )
#     return normed_stack
def compute_pixelwise_norm(stack : pd.Series) -> pd.Series:
    """
    Take a stack of images, and normalize each pixel against the other pixels at the same position. Return it back as images.
    Index must have the following levels:
        - cat_row
        - target
        - numbasis
    """
    try:
        assert(all([i in stack.index.names for i in ['cat_row', 'target', 'numbasis']]))
    except AssertionError as e:
        print("Error: Index does not have required levels.")
        raise(e)
    stamp_shape = np.stack(stack.values).shape[-2:]
    pixel_df = pd.concat(
        stack.map(lambda img: pd.Series(np.ravel(img))).to_dict(),
        names=list(stack.index.names) + ['pixel_id']
    )
    # compute the sigma-clipped mean and std, for scaling
    pixel_df_mean = pixel_df.groupby(["cat_row","numbasis","pixel_id"]).apply(
        lambda pixels: dutils.sigma_clipped_stats(pixels)[0]
    )
    pixel_df_std = pixel_df.groupby(["cat_row","numbasis","pixel_id"]).apply(
        lambda pixels: dutils.sigma_clipped_stats(pixels)[-1]
    )
    # apply the scaling
    pixel_df_norm = pixel_df.groupby(["cat_row","numbasis","pixel_id"], group_keys=False).apply(
       lambda group: (group - pixel_df_mean.loc[group.name])/pixel_df_std.loc[group.name]
    )
    # convert it back into images for each target
    pixel_df_norm_stamps = pixel_df_norm.groupby(["cat_row","numbasis","target"], group_keys=False).apply(
        lambda pixels: np.reshape(pixels, stamp_shape)
    )
    return pixel_df_norm_stamps



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
        ).reorder_levels(
            ['cat_row', 'target', 'numbasis']
        ).sort_index()

        # Add the analysis columns
        self.all_results = self.normalize_residuals_and_apply_matched_filter(self.all_results)
        # filter down to just a kklip of interest, for convenience and plotting
        # call a function that updates everything you need to update when Kklip or mf_width change
        self.all_results['pixelwise_snrmap'] = self.generate_pixelwise_snrmap(self.all_results, 'klip_sub_norm')
        self.all_results['pixelwise_detmap'] = self.generate_pixelwise_snrmap(self.all_results, 'detmap_norm')
        # self.all_results['candidates'] = self.find_sources(self.all_results['pixelwise_detmap'])
        self.results = self.filter_kklip(self.all_results, self.kklip)
        self.filter_maps()

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

    def filter_maps(self):
        self.results = self.filter_kklip(self.all_results, self.kklip)
        self.detection_maps = self.results['pixelwise_detmap']
        self.residual_snr_maps = self.results['pixelwise_snrmap']
        self.contrast_maps = self.results['contrastmap']#.reset_index("numbasis").drop(columns="numbasis").squeeze()
        # # Kklip is redundant so let's remove it from the index
        for df in [self.results, self.detection_maps, self.residual_snr_maps, self.contrast_maps]:
            df.index = df.index.droplevel("numbasis")

    def recompute(self):
        self.filter_maps()


    def normalize_residuals_and_apply_matched_filter(self, results):
        # normalize the arrays so we can compare them against each other
        results['klip_sub_std'] = results['klip_sub'].apply(
            lambda img: sigma_clipped_stats(img, sigma=1)[-1]
        )
        results['klip_sub_norm'] = results['klip_sub'].map(mf_utils.normalize_array_sigmaclip)
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


    def generate_pixelwise_snrmap(self, results_df, results_column='klip_sub_norm'):
        """
        Compute the pixelwise SNR by comparing the pixels at a single coordinate from different stamps,
        then transform back to the stamp for the target.
        In the final image, the value at each pixel represents the pixel's SNR
        in a different distribution from all the other pixels.
        """
        column = results_df[results_column]
        snrmap = column.groupby(['cat_row', 'numbasis'], group_keys=False).apply(
            compute_pixelwise_norm
        )
        # make sure the indices are aligned with the original dataframe
        return snrmap.reorder_levels(results_df.index.names).loc[results_df.index]

    def find_sources(self, snrmaps : pd.Series, threshold=3, npixels=1):
        "Run photutils' source detection algo on some SNR maps"
        # ignore warnings that get thrown when no sources are detected
        with warnings.catch_warnings(action="ignore"):
            sources = snrmaps.apply(
                lambda detmap: detect_sources(
                    detmap,
                    threshold=threshold,
                    npixels=npixels
                )
            );
        return sources


def snr_vs_catalog(
        results_stack : pd.Series,
        target_id : str,
        drop_stars : list[str] = []
) -> np.ndarray:
    """
    Compute the pixel-wise SNR of a stamp against the catalog.
    You may want to first apply the cross-catalog matched filter and the pick the brightest value across the matched filters

    Parameters
    ----------
    results_stack : pd.Series
        A series of images indexed by: target_star, cat_row, numbasis
    target_id : str
        The identifier of the target star
    drop_stars : list[str]
        Any stars to exclude from the reference set

    Output
    ------
    snr_map : np.array
      A map of each pixel's SNR measured against that pixel in the catalog
    """
    target_pix = results_stack.loc[target_id].apply(mf_utils.normalize_array_sigmaclip)
    stamp_shape = np.stack(target_pix).shape[-2:]
    reference_pixels = results_stack.drop([target_id] + drop_stars, level='target_star').apply(
        lambda img: pd.Series(np.ravel(img), name='pixel_id')
    ).apply(
        # normalize each stamp to sigma=1
        lambda pix: mf_utils.normalize_array_sigmaclip(pix),
        axis=1
    )
    reference_mean = reference_pixels.groupby(["cat_row", "numbasis"]).apply(
        # compute the sigma-clipped STD for each pixel
        lambda group: group.apply(lambda pix: dutils.sigma_clipped_stats(pix)[0])
    ).apply(
        # reshape into images
        np.reshape,
        shape=stamp_shape,
        axis=1
    )
    reference_std = reference_pixels.groupby(["cat_row", "numbasis"]).apply(
        # compute the sigma-clipped STD for each pixel
        lambda group: group.apply(lambda pix: dutils.sigma_clipped_stats(pix)[-1])
    ).apply(
        # reshape into images
        np.reshape,
        shape=stamp_shape,
        axis=1
    )
    snr_stack = (target_pix - reference_mean)/reference_std
    return snr_stack


