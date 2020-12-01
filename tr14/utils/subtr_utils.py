"""
Utilities for PSF subtraction
"""

import numpy as np
import pandas as pd

from . import table_utils
from . import shared_utils

# RDI
import sys
sys.path.append(shared_utils.load_config_path('rdi_path'))
import rdi
from rdi import RDIklip as RK
from rdi.utils import utils as rutils


def calc_corr_mat(stamps, corr_func, corr_func_args={}):
    """
    Calculate a correlation matrix between the stamps.

    Parameters
    ----------
    stamps : pd.Series (or castable to pd.Series)
      the stamp images whose correlations will be calculated.
    corr_func : python function
      a correlation function that accepts arguments as (targ, refs, **kwargs)
    corr_func_args : dict [{}]
      dictionary of keyword arguments to pass to the correlation function

    Output
    ------
    corr_mat : pd.DataFrame
      the final correlation matrix. The index and columns are whatever the
      stamps index is
    """
    if not isinstance(stamps, pd.Series):
        stamps = pd.Series(stamps)
    # initialize the correlation matrix and use it for looping
    corr_mat = pd.DataFrame(np.nan,
                            index=stamps.index, columns=stamps.index,
                            dtype=np.float)
    # rescale the stamps to max value = 1
    stamps = stamps.apply(lambda x: x/np.nanmax(x))
    # calculate an upper triangular matrix of correlations
    other_stamps = list(stamps.index)
    for i, stamp_id in enumerate(corr_mat.index[:-1]):
        targ = stamps[stamp_id]
        # remove target stamp from the references
        #other_stamps = list(corr_mat.index)
        other_stamps.pop(0)
        corr = corr_func(targ, np.stack(stamps[other_stamps], axis=0),
                         **corr_func_args)
        corr_mat.loc[stamp_id, other_stamps] = corr

    # now fill in the empty half of the matrix - beware of NaN's
    corr_mat = corr_mat.add(corr_mat.T, fill_value=0)

    return corr_mat


def klip_subtr_wrapper(target_stamp, refs_table, restore_scale=False, klip_args={}):
    """
    Wrapper for RDI klip subtraction so you can pass in a table without extra processing

    Parameters
    ----------
    target_stamp: MxN np.array
      stamp to be subtracted, as an image
    refs_table: pd.DataFrame
      dataframe in the stamps_table format
    restore_scale: bool [False]
      if True, put back the original flux scale of the target stamp. If False,
      then leave the KL-subtracted result as-is after the target and references
      have been scaled to max_val = 1

    Output
    ------
    kl_sub : np.array
      an array of klip-subtracted images. the shape of the returned array is
      whatever is appropriate for the passed KLIP parameters, but the final two
      dimensions are always the row, col image coordinates

    """
    targ_stamp_flat = rutils.flatten_image_axes(target_stamp)
    ref_stamps_flat = rutils.flatten_image_axes(np.stack(refs_table['stamp_array']))
    # rescale target and references
    target_scale = np.nanmax(targ_stamp_flat, axis=-1, keepdims=True)
    ref_stamps_scale = np.nanmax(ref_stamps_flat, axis=-1, keepdims=True)
    targ_stamp_flat = targ_stamp_flat / target_scale
    ref_stamps_flat = ref_stamps_flat / ref_stamps_scale
    # apply KLIP
    kl_max = len(ref_stamps_flat)-1
    kl_basis = RK.generate_kl_basis(ref_stamps_flat,
                                    kl_max=klip_args.get('kl_max', kl_max))
    kl_sub = RK.klip_subtract_with_basis(targ_stamp_flat, ref_stamps_flat,
                                         n_bases=klip_args.get('n_bases', kl_max))
    # return the subtracted stamps as images
    kl_sub_img = rutils.make_image_from_flat(kl_sub)
    if restore_scale == True:
        kl_sub_img = kl_sub_img * target_scale
    return kl_sub_img





class StarTarget:
    """
    Collect all the stamps and PSF references for a unique star
    """

    def __init__(self, star_id, stamp_df, psf_corr_mat=None):
        """
        Initialize the object with a list of targets and references
        """
        self.target_stamps = stamp_df.set_index('stamp_id').query('stamp_star_id == @star_id')['stamp_array']
        self.ref_stamps = stamp_df.set_index('stamp_id').query('stamp_star_id != @star_id')['stamp_array']
        self.psf_corr_mat = psf_corr_mat.dropna( axis=1, how='all')
