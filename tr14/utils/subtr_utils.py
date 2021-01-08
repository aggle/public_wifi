"""
Utilities for PSF subtraction:
- PSF correlation metrics
- KLIP utilities and wrappers
- KLIP PSF model reconstruction
"""

import numpy as np
import pandas as pd

# RDI imports
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import sys

# local imports
from . import table_utils
from . import shared_utils
from . import image_utils

# pyklip
sys.path.append(shared_utils.load_config_path('pyklip_path', as_str=True))
from pyklip import klip


###################
# PSF correlation #
###################
# these metrics are described in Ruane et al., 2019

# Mean squared error (MSE)
def calc_refcube_mse(targ, references):
    """
    Calculate the pixel-wise mean squared error (MSE) for each reference
    compared to the target. Returns -log10(MSE) so that the largest values
    indicate the most similarity 

    Parameters
    ----------
      targ: the target image to subtract
      references: the cube of references [Nimg x Npix] (can also be 3-d)

    Output
    ------
      mse: Nref array of MSE values
    """
    npix = targ.size
    targ = targ.ravel() # 1-d
    #references = references.reshape(references.shape[0],
    #                                reduce(lambda x,y: x*y, references.shape[1:]))
    references = image_utils.flatten_image_axes(references)

    mse = np.squeeze(np.nansum((targ - references)**2, axis=-1) / npix)
    #mse = (np.linalg.norm(targ-references, axis=-1)) / npix
    mse = -np.log10(mse)

    return mse

# Pearson correlation coefficient (PCC)
def calc_refcube_pcc(targ, references):
    """
    Compute the Pearson correlation coefficient (PCC) between the target and the refs
    PCC(a, b) = cov(a, b)/(std(a) std(b))
    does higher or lower mean more similar?
    Parameters
    ----------
      targ: 1- or 2-d target image
      references: 2- or 3-D cube of references
    Output
    ------
      pcc: Nref array of PCC values
    """
    npix = targ.size
    targ = targ.ravel() # 1-d
    #references = references.reshape(references.shape[0],
    #                                reduce(lambda x,y: x*y, references.shape[1:]))
    references = image_utils.flatten_image_axes(references).copy()

    # stats.pearsonr can't handle nan's, which are present in edge stamps
    pcc = np.array([stats.pearsonr(targ, r)[0] for r in references])
    return pcc

# Structural similarity index metric (SSIM), with helper functions
def calc_refcube_ssim(targ, references, win_size=3., kw_args={}):
    """
    Use the scikit-image library function to calculate the SSIM

    Parameters
    ----------
    targ : np.array
      2-D target image
    references : np.array
      3-D [Nref, Nx, Ny] reference image stack
    win_size : int
      window size for calculating SSM (must be odd)
    kw_args : {}
      any additional keyword arguments to pass to
      skimage.metrics.structural_similarity()

    Output
    ------
    ssim_vals : np.array
      Nref array of SSIM values between the target and the indicated reference image

    """
    targ = targ.ravel()
    references = image_utils.flatten_image_axes(references)
    ssim_vals = np.array([ssim(targ, r, win_size=win_size,
                               use_sample_covariance=True,
                               **kw_args)
                          for r in references])
    return ssim_vals

# this function applies the correlation functions to a stamp dataframe
def calc_corr_mat(stamps, corr_func, corr_func_args={}, rescale=True):
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
    if rescale == True:
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


####################
# KLIP subtraction #
####################

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
    klip_args : dict [{}]
      dictionary of optional arguments passed to pyklip.klip.klip_math

    Output
    ------
    kl_sub_img : np.array
      an array of klip-subtracted images. the shape of the returned array is
      whatever is appropriate for the passed KLIP parameters, but the final two
      dimensions are always the row, col image coordinates
    kl_basis_img : np.array
      array of the klip basis, as images.

    """
    targ_stamp_flat = image_utils.flatten_image_axes(target_stamp)
    ref_stamps_flat = image_utils.flatten_image_axes(np.stack(refs_table['stamp_array']))
    # rescale target and references
    target_scale = np.nanmax(targ_stamp_flat, axis=-1, keepdims=True)
    ref_stamps_scale = np.nanmax(ref_stamps_flat, axis=-1, keepdims=True)
    targ_stamp_flat = targ_stamp_flat #/ target_scale
    ref_stamps_flat = ref_stamps_flat #* (target_scale / ref_stamps_scale)
    # apply KLIP
    kl_max = np.array([len(ref_stamps_flat)-1])
    numbasis = klip_args.get('numbasis', kl_max)
    # kl_sub, kl_basis = klip.klip_math(targ_stamp_flat, ref_stamps_flat,
    #                                   numbasis = klip_args.get('numbasis', kl_max),
    #                                  return_basis = klip_args.get('return_basis', True))
    return_basis = klip_args.get('return_basis', True)
    klip_results = klip.klip_math(targ_stamp_flat, ref_stamps_flat,
                                  numbasis = numbasis,
                                  return_basis = return_basis)
    if return_basis == True:
        kl_sub, kl_basis = klip_results
    else:
        kl_sub = klip_results

    # convert to series, indexed by the numbasis
    if isinstance(numbasis, int):
        numbasis = [numbasis]
    kl_sub = pd.Series(dict(zip(numbasis, kl_sub.T)))
    kl_sub.index.name = 'numbasis'
    if return_basis == True:
        kl_basis = pd.Series(dict(zip(range(1, len(kl_basis)+1), kl_basis)))
        kl_basis.index.name = 'kklip'


    # return the subtracted stamps as images
    kl_sub_img = kl_sub.apply(image_utils.make_image_from_flat)
    if return_basis == True:
        kl_basis_img = kl_basis.apply(image_utils.make_image_from_flat)
    if restore_scale == True:
        kl_sub_img = kl_sub_img * target_scale
        if return_basis == True:
            kl_basis_img = kl_basis_img * target_scale

    if return_basis == True:
        return kl_sub_img, kl_basis_img
    else:
        return kl_sub_img


def klip_subtr_table(targ_row, stamp_table, restore_scale=False, klip_args={}):
    """
    Designed to use stamps_tab.apply to do klip subtraction to a whole table of stamps
    Assumes return_basis is True; otherwise, fails
    """
    target_star_id = targ_row['stamp_star_id']
    target_stamp = targ_row['stamp_array']
    refs_table = stamp_table.query('stamp_star_id != @target_star_id')

    results =  klip_subtr_wrapper(target_stamp, refs_table,
                                  restore_scale, klip_args)
    return results


def psf_model_from_basis(target, kl_basis, numbasis=None):
    """
    Generate a model PSF from the KLIP basis vectors. See Soummer et al 2012.

    Parameters
    ----------
    target : np.array
      the target PSF, 2-D
    kl_basis : np.array
      Nklip x Nrows x Ncols array
    numbasis : int or np.array [None]
      number of Kklips to use. Default is None, meaning use all the KL vectors

    Output
    ------
    psf_model : np.array
      Nklip x Nrows x Ncols array of the model PSFs. Dimensionality depends on the value
      of num_basis
    """
    # make sure numbasis is an integer array
    if numbasis is None:
        numbasis = len(kl_basis)
    if isinstance(numbasis, int):
        numbasis = np.array([numbasis])
    numbasis = numbasis.astype(np.int)

    coeffs = np.inner(target.ravel(), image_utils.flatten_image_axes(kl_basis))
    psf_model = kl_basis * np.expand_dims(coeffs, [i+1 for i in range(kl_basis.ndim-1)])
    psf_model = np.array([np.sum(psf_model[:k], axis=0) for k in numbasis])

    return np.squeeze(psf_model)


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


class SubtrManager:
    """
    Class to manage PSF subtraction for a self-contained set of stamps.
    Initializes with a DBManager instance as argument. Optionally, calculate the PSF correlations
    """

    def __init__(self, db_manager, calc_corr_flag=True):
        # calculate all three correlation matrices
        self.db = db_manager
        if calc_corr_flag == True:
            self.calc_psf_corr()

    def calc_psf_corr(self):
        """
        Compute the correlation matrices
        """
        # set the stamp ID as the index
        stamps = self.db.stamps_tab.set_index('stamp_id')['stamp_array']
        self.corr_mse = calc_corr_mat(stamps, calc_refcube_mse)
        self.corr_pcc = calc_corr_mat(stamps, calc_refcube_pcc)
        self.corr_ssim = calc_corr_mat(stamps, calc_refcube_ssim)

    def subtr_table(self, numbasis=None):
        """
        Do PSF subtraction on the whole table

        Parameters
        ----------
        None

        Output
        ------
        sets self.psf_subtr and self.psf_model

        """
        if numbasis is None:
            numbasis = np.arange(1, self.db.stamps_tab.shape[0]-1, 20)
        subtr_mapper = lambda x: klip_subtr_table(x, self.db.stamps_tab,
                                                  klip_args={'numbasis': numbasis,
                                                             'return_basis': True})
        results = self.db.stamps_tab.set_index('stamp_id').apply(subtr_mapper, axis=1)
        self.psf_subtr = results.apply(lambda x: x[0])
        self.psf_model = results.apply(lambda x: x[1])



