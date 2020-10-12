"""
Utilities for PSF subtraction
"""

import numpy as np
import pandas as pd

from . import table_utils


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
