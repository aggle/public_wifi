"""
Detecting point sources and analyzing results
"""
import numpy as np
import pandas as pd

from astropy import units
from astropy import nddata
from astropy.wcs import WCS

import reproject
from reproject import mosaicking

from .utils import table_utils
from .utils import image_utils

######################
# SNR and detections #
######################
def normalize_stamp(stamp):
    """
    Normalize the stmap to the unit normal distribution.
    This only works with KLIP residuals, which are whitened.
    """
    # shift to mean 0
    normed_stamp = stamp - np.nanmean(stamp)
    # scale to std = 1
    normed_stamp = normed_stamp/np.nanstd(normed_stamp)
    return normed_stamp

def create_std_map(subtr_results, mode='pixel', normalize=False):
    """
    Parameters
    ----------
    subtr_results
    mode: 'pixel' or 'stamp'
    normalize: False
      transform residual stamps to N(0, 1)

    Output
    ------
    std_map : pd.DataFrame, same form as subtr_results.residuals,
      with the standard deviations
    """
    resids = subtr_results.residuals.copy()
    refs = subtr_results.references.copy()
    if normalize == True:
        resids = resids.map(normalize_stamp)
    if mode == 'pixel':
        # compute the SNR map by comparing the target stamp pixel to that same
        # pixel in the residual stamps of all the other stars' residuals
        def calc_pixelwise_std(row, subtr_results):
            # apply row-wise
            star, stamp = row.name
            # pull out the references used for this stamp
            targ_refs = refs.loc[(star, stamp)].dropna()
            # these are the residuals for the reductions of those references
            ref_residuals = resids.query("stamp_id in @targ_refs")
            # now for each array of reference residuals, compute the std
            def _calc_pixelwise_std(col):
                col = col.dropna()
                try:
                    arr = np.stack(col.values)
                except ValueError:
                    return np.nan
                # compute the std and encapsulate it to make pandas happy
                std = [np.nanstd(arr, axis=0)]
                return std
            stds = ref_residuals.apply(_calc_pixelwise_std)
            return stds.squeeze()
        std_map = resids.apply(lambda x: calc_pixelwise_std(x, subtr_results), axis=1)
    elif mode == 'stamp':
        # calculate the standard deviation stampwise
        std_map = subtr_results.residuals.map(np.nanstd)
    else:
        print("mode not recognized, please choose `pixel` or `stamp`")
        std_map = None
    return std_map


def create_snr_map(subtr_results, mode='pixel', normalize=True):
    """
    Create an SNR map from a subtraction results tuple

    Parameters
    ----------
    subtr_results : a Results named tuple generated by a subtraction manager object

    Output
    ------
    same as the residuals dataframe, but with SNR maps instead of residuals

    """
    std_map = subtr_results.residuals.apply(lambda x: create_std_map(subtr_results, mode, normalize),
                                            axis=1)
    snr_map = subtr_results.residuals/std_map
    return snr_map


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


def get_mosaic(cutouts, resolution=None, verbose=False):
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
    compute_snr : [True] if True, compute pixel-wise SNR. If False, don't. Takes time.
    """

    def __init__(self, sub_mgr, instrument=None, compute_snr=True):
        # start with a subtraction manager object
        self.sm = sub_mgr
        self.db = sub_mgr.db
        # the instrument matters for stuff like pixel scale
        self.instr = instrument
        # port over the results
        self.results_stamps = {
            "references": sub_mgr.subtr_results.references,
            "residuals": sub_mgr.subtr_results.residuals,
            "models": sub_mgr.subtr_results.models,
        }
        # compute SNR maps
        if compute_snr == True:
            self.results_stamps["std"] = self.compute_std(mode='pixel')
            self.results_stamps["snr"] = self.results_stamps['residuals']/self.results_stamps['std']
        # make the target stamp cutouts
        self.stamp_cutouts = self.stamps2cutout()
        # this holds the wcs data
        self.stamp_wcs = self.stamp_cutouts.apply(lambda x: x.wcs)
        # turn the results into nddata objects
        self.results_cutouts = self.subtr2cutout()

    def compute_std(self, mode='pixel'):
        """Compute noise maps for the residuals"""
        std_map = create_std_map(self.sm.subtr_results, mode, normalize=True)
        return std_map
    def compute_snr(self, mode='pixel'):
        """Compute SNR maps for the residuals """
        snr_map = create_snr_map(self.sm.subtr_results, mode, normalize=True)
        return snr_map


    def stamps2cutout(self):
        """get WCS for the original stamps"""
        cutouts = get_base_cutouts(self.db, self.instr.stamp_size)
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

    def make_star_result_mosaics(self, star_id, key='residuals', res_factor=1):
        """
        Make a mosaic out of the star.
        Parameters
        ----------
        star_id : the star you want a mosaic of
        key : valid key to self.results_cutouts
        res_factor : factor by which to multiply the pixel scale
        """
        if self.instr is not None:
            resolution = np.mean(instr.pix_scale) * units.pixel * res_factor
        else:
            resolution = None
        star_resids = self.results_cutouts[key].query("star_id == @star_id")
        mosaic = star_resids.apply(lambda col: get_mosaic(col, resolution))
