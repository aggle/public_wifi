"""
Detecting point sources and analyzing results
"""
import numpy as np
import pandas as pd

from astropy import units
from astropy import nddata
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

import reproject
from reproject import mosaicking


from .utils import table_utils
from .utils import image_utils
from .utils import detection_utils




#############################
# Mosaicking and projection #
#############################
def get_base_cutouts(db, stamp_size):
    """
    Get Cutout2D objects of all the stamps in a database, with the appropriate
    WCS information. Do it in a way that minizes File IO, which can take a long
    time for fits files (in my experience, and depending on the files).

    Parameters
    ----------
    db : db_manager.DBManager object

    Output
    ------
    cutouts : pd.Series of cutouts with the stamp ID as the index
    """
    table = db.join_all_tables()
    # group by exposure ID so you only read each file once
    gb_exp = table.groupby('ps_exp_id')
    # put the whole exposure into an NDData object
    def wcs_func(x):
        img = nddata.NDData(data=table_utils.get_img_from_exp_id(x.name, 'SCI',
                                                                 db.db_path, db.config_file),
                            wcs=table_utils.get_wcs_from_exp_id(x.name,
                                                                db.lookup_dict['lookup_files'],
                                                                db.config_file))
        return img
    parent_nddata = gb_exp.apply(wcs_func)
    row_cutout = lambda row: nddata.Cutout2D(parent_nddata[row['ps_exp_id']].data,
                                             position=tuple(row[['ps_x_exp', 'ps_y_exp']]-1),
                                             wcs=parent_nddata[row['ps_exp_id']].wcs, 
                                             size=stamp_size,
                                             mode='partial', copy=True)
    cutouts = table.set_index('stamp_id').apply(row_cutout, axis=1)
    return cutouts


def subtr2cutout(subtr_results, wcs_series):
    """
    Convert subtraction stamps to an nddata.NDData object (i.e. find the right WCS and add it)
    Dataframe must not have NaNs

    Parameters
    ----------
    sutbr_results : pd.DataFrame
      a subtraction results table (index is star+stamp, column is Ncomponent)
    wcs_series : pd.Seriees
      a series of WCS coordinates with the WCS of the original target stamp cutout
      index must be the original stamp ID
      e.g. the result of cutouts.apply(lambda x: x.wcs)

    Output
    ------
    subtr_cutouts : pd.DataFrame of identical shape to the input df except with
      NDData objects for entries instead of arrays

    """
    def row_func(row):
        return row.apply(lambda arr: nddata.NDData(data=arr, wcs=wcs_series[row.name[1]]))
    subtr_cutouts = subtr_results.apply(row_func, axis=1)
    return subtr_cutouts


def get_mosaic(
        cutouts : pd.Series,
        resolution : units.Quantity | None = None,
        verbose : bool =False
) -> pd.Series:
    """
    Create a mosaic from a series of nddata images.
    Run using pd.Series.apply to a series of NDData objects.

    Parameters
    ----------
    imgs : pd.Series
      Series whose elements are nddata.NDData (or nddata.Cutout2D)
    resolution : [None] astropy.units.Quantity (arcsec)
      the desired pixel scale for the final mosaic
    verbose : [False] if True, notifiy when finished (useful when running on groupby)

    Output
    ------
    mosaic : pd.Series with the footprint and mosaic

    """
    "input: a column of each residual stamp at the same Component"
    # first, drop NaN elements of the NDData-type cutouts
    drop = cutouts.apply(lambda el: np.isnan(el.data).all())
    col = cutouts[~drop]
    data = list(cutouts.apply(lambda x: (x.data, x.wcs)))
    wcs_opt, shape_opt = mosaicking.find_optimal_celestial_wcs(data,
                                                               auto_rotate=False,
                                                               resolution=resolution)
    mosaic, footprint = mosaicking.reproject_and_coadd(data,
                                                       wcs_opt, shape_out=shape_opt,
                                                       reproject_function=reproject.reproject_adaptive,
                                                       return_footprint=True)
    if verbose == True:
        print(f"Finished {cutouts.name}")
    return [mosaic]
    #return pd.Series([mosaic, footprint], index=['mosaic', 'footprint'])


#############
# Detection #
#############

"""
This definition says that any pixel must have an SNR above a threshold value T
for at least N number of modes to be declared a detection
"""
def count_thresh_pixels(snr_df):
    """
    Docstring goes here

    Parameters
    ----------
    snr_df : a dataframe of SNR stamps

    Output
    ------
    pixel indices that have detections

    """
    # future: do this as a groupby
    pixelwise_df = snr_df.map(image_utils.flatten_image_axes)

##################
# Analysis class #
##################
class AnaManager:
    """
    Class to handle various analysis tasks like:
    - rotation
    - detection
    - photometry
    - fake injection and recovery

    Parameters
    ----------
    sub_mgr : public_wifi.SubtrManager instance
    instrument : [None] public_wifi.instruments.Instrument class
    stamp_size : int [0]
      the size of stamps to cut out of the original images. Not sure why i'm doing this tbh
    compute_snr : str = 'pixel'
      If pixel, compute pixelwise-SNR. Other choices: '' (do nothing), 'stamp' - stamp-wise SNR
    """

    def __init__(
            self,
            sub_mgr,
            instrument = None,
            stamp_size : int = 0,
            compute_snr : str = 'pixel',
    ) -> None:
        # start with a subtraction manager object
        self.sm = sub_mgr
        self.db = sub_mgr.db
        # the instrument matters for stuff like pixel scale
        # self.instr = instrument
        self.stamp_size = stamp_size
        # port over the results
        self.results_stamps = {
            "references": sub_mgr.subtr_results.references,
            "residuals": sub_mgr.subtr_results.residuals,
            "models": sub_mgr.subtr_results.models,
        }
        # compute SNR maps
        if compute_snr != '':
            # compute the noise
            self.results_stamps["std"] = detection_utils.compute_noise_map(
                sub_mgr.subtr_results,
                mode=compute_snr,
                clip_thresh = 3.0,
                normalize=True,
            )
            # now normalize the residuals and divide them by the noise
            snr_maps = detection_utils.normalize_stamps(
                sub_mgr.subtr_results.residuals
            ) / self.results_stamps["std"]
            self.results_stamps["snr"] = snr_maps
            # now, find candidate detections
            self.results_stamps['detections'] = detection_utils.get_candidates(self.results_stamps['snr'])
        # make the target stamp cutouts
        self.stamp_cutouts = self.stamps2cutout()
        # this holds the wcs data
        self.stamp_wcs = self.stamp_cutouts.apply(lambda x: x.wcs)
        # turn the results into nddata objects
        self.results_cutouts = self.subtr2cutout()

    def stamps2cutout(self):
        """
        Take the initial set of stamps and turn them into NDData.cutout objects
        that carry WCS information.
        """
        cutouts = get_base_cutouts(self.db, self.stamp_size)
        return cutouts

    def subtr2cutout(self):
        """
        Turn all the subtraction stamps into NDData objects.
        If self.results_cutouts exists, sets that; otherwise, returns dict.
        Returned objects have same keys and shape as results_stamps objects
        """
        results_dict = {}
        for k in ['models', 'residuals', 'snr']:
            try:
                v = self.results_stamps[k]
            except KeyError:
                continue
            results_dict[k] = subtr2cutout(v, self.stamp_wcs)
        if hasattr(self, "results_cutouts"):
            self.results_cutouts.update(results_dict)
        else:
            return results_dict

    # this function needs to be a dummy for now
    def make_star_result_mosaics(
            self,
            # star_id,
            # key='residuals',
            # res_factor=1
    ) -> None:
        return None
    #     """
    #     Make a mosaic out of the star residuals.

    #     Parameters
    #     ----------
    #     star_id : the star you want a mosaic of
    #     key : valid key to self.results_cutouts
    #     res_factor : factor by which to multiply the pixel scale

    #     Output
    #     ------
    #     A single mosaicked image from the combined images of the star

    #     """
    #     if self.instr is not None:
    #         resolution = np.mean(self.instr.pix_scale) * units.pixel * res_factor
    #     else:
    #         resolution = None
    #     star_resids = self.results_cutouts[key].query("star_id == @star_id")
    #     # return star_resids, resolution
    #     mosaic = star_resids.apply(lambda row: get_mosaic(row, resolution), axis=1)
    #     return mosaic
