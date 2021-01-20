"""
Utilities for PSF subtraction:
- PSF correlation metrics
- KLIP utilities and wrappers
- KLIP PSF model reconstruction
"""

import numpy as np
import pandas as pd
from collections import namedtuple

# RDI imports
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import sys

# local imports
from . import table_utils
from . import shared_utils
from . import image_utils

# NMF
from sklearn.decomposition import NMF
# pyklip
sys.path.append(shared_utils.load_config_path('pyklip_path', as_str=True))
from pyklip import klip


###################
# PSF correlation #
###################
# these metrics are described in Ruane et al., 2019

# Mean squared error (MSE)
def calc_refcube_mse(targ, references, kw_args={}):
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

# sometimes this can be helpful for plotting
def rescale_mse(mse_scores):
    """Remap linearly from 0 to 1"""
    print("Warning: performing this operation breaks the symmetry of the "\
          "correlation matrix. From now on be careful to only use *columns*.")
    mse_scores = mse_scores - mse_scores.min()
    mse_scores = mse_scores / mse_scores.max()
    return mse_scores

# Pearson correlation coefficient (PCC)
def calc_refcube_pcc(targ, references, kw_args={}):
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
def calc_refcube_ssim(targ, references, kw_args={}):
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
    kw_args.setdefault('win_size', 5.)
    targ = targ.ravel()
    references = image_utils.flatten_image_axes(references)
    ssim_vals = np.array([ssim(targ, r,
                               use_sample_covariance=True,
                               **kw_args)
                          for r in references])
    return ssim_vals

# this function applies the correlation functions to a stamp dataframe
def calc_corr_mat(stamps, corr_func, corr_func_args={}, rescale=False):
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
    rescale : bool [False]
      if True, rescale all the stamps so the max flux is 1 before computing
      their correlation

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
                         kw_args=corr_func_args)
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
    refs_table: pd.Series
      pandas series containing the reference stamps (index can be the reference IDs)
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
    ref_stamps_flat = image_utils.flatten_image_axes(np.stack(refs_table))
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
        kl_basis.index.name = 'numbasis'


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
    # replace this with the self.get_references() function that handles more casese
    refs_table = stamp_table.query('stamp_star_id != @target_star_id')
    # get the ref IDs, which may be the index
    try:
        ref_ids = stamp_table['stamp_id']
    except:
        print("`stamp_id` not in the columns, cannot return references list")
        ref_ids = None
    results =  klip_subtr_wrapper(target_stamp, refs_table,
                                  restore_scale, klip_args)
    return results, ref_ids


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


class SubtrManager:
    """
    Class to manage PSF subtraction for a self-contained set of stamps.
    Initializes with a DBManager instance as argument. Optionally, calculate the PSF correlations
    """

    def __init__(self, db_manager, calc_corr_flag=True):
        # calculate all three correlation matrices
        self.db = db_manager

        # table for tracking references used
        cols = pd.Index(self.db.stamps_tab['stamp_id'], name='targ_id')
        indx = pd.Index(self.db.stamps_tab['stamp_id'], name='ref_id')
        self.reference_table = pd.DataFrame(None, columns=columns, index=index, dtype=bool)
        # correlation function arguments
        self.corr_func_args_dict = {'mse': {},
                                    'pcc': {},
                                    'ssim': {'win_size': 5.}}
        # klip arguments
        self.klip_args_dict = {'return_numbasis': True,
                               'numbasis': None}
        self.nmf_args_dict = {'init': 'random',
                              'random_state': 0}

        if calc_corr_flag == True:
            self.calc_psf_corr()


    def calc_psf_corr(self):
        """
        Compute the correlation matrices. Sets self.corr_mats, a namedtuple
        with all the matrices as attributes
        """
        # initialize the contained to hold the correlation matrices
        corr_mats = namedtuple('corr_mats', 'mse','pcc','ssim')
        # set the stamp ID as the index
        stamps = self.db.stamps_tab.set_index('stamp_id')['stamp_array']
        corr_mats.mse = calc_corr_mats.mat(stamps, calc_refcube_mse,
                                      self.corr_mats.func_args_dict['mse'])
        corr_mats.pcc = calc_corr_mats.mat(stamps, calc_refcube_pcc,
                                      self.corr_mats.func_args_dict['pcc'])
        corr_mats.ssim = calc_corr_mats.mat(stamps, calc_refcube_ssim,
                                       self.corr_mats.func_args_dict['ssim'])
        # finally, assign to the object
        self.corr_mats = corr_mats


    def remove_nans_from_psf_subtr(self):
        """
        Sometimes the PSF subtraction table has columns that are entirely NaN. Remove them.
        modifies self.psf_subtr in place
        """
        if not hasattr(self, 'psf_subtr'):
            print("self.psf_subtr does not exist (yet?)")
            raise AttributeError
        filtfunc = lambda x: np.all(np.isnan(x))
        drop_cols = self.psf_subtr.applymap(filtfunc).all(axis=0)
        drop_cols = drop_cols[drop_cols].index
        self.psf_subtr.drop(drop_cols, axis=1, inplace=True)


    def _get_reference_stamps(self, targ_stamp_id):
        """
        Given a target stamp, find the list of appropriate reference stamps
        using e.g. the star_id and the stamp_ref_flag value

        Parameters
        ----------
        targ_stamp_id : str
          the stamp_id of the PSF targeted for subtraction

        Output
        ------
        ref_stamps : pd.Series
          pd.Series of the reference stamp arrays, where the index is the stamp id

        """
        # first, make sure the input is a stamp id
        if not isinstance(targ_stamp_id, str):
            print(f"Error: input is not a string")
            raise ValueError
        if targ_stamp_id[0] != 'T':
            print(f"Error: passed value of targ_stamp_id is not a valid "\
                  f"stamp ID ({targ_stamp_id})")
        # reject all stamps with a bad reference flag
        ref_query = "stamp_ref_flag == True"
        # reject all references that correspond to the target's parent star
        targ_star_id = self.db.find_matching_id(targ_stamp_id, 'star')
        ref_query += " and stamp_star_id != @targ_star_id"
        ref_stamps = self.db.stamps_tab.query(ref_query)
        return ref_stamps.set_index('stamp_id')['stamp_array']


    def perform_table_subtraction(self, func, func_args={}):
        """
        This seems like the perfect candidate for a decorator
        Apply the subtraction function to the whole table using table.apply()

        Parameters
        ----------
        func : PSF subtraction function
        func_args : {}
          dictionary of arguments

        Output
        ------
        sets self.subtr_results, which contains the residuals, models, and references

        """
        subtr_mapper = lambda x: self._table_apply_wrapper(x, func, func_args)
        agg_results = self.db.stamps_tab.set_index('stamp_id', drop=False).apply(subtr_mapper, axis=1)
        results = namedtuple('subtr_results', ('residuals','models','references'))
        results.residuals = results.apply(lambda x: x.residuals)
        results.models = results.apply(lambda x: x.residuals)
        results.references = results.apply(lambda x: x.residuals)
        self.subtr_results = results
        #self.psf_model = results.apply(lambda x: x.models)
        #self.psf_subtr = results.apply(lambda x: x.residuals)
        #self.subtr_refs = results.apply(lambda x: x.ref_ids).T
        

    def perform_table_subtraction_klip(self, numbasis=None):
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
            # by default, use a log-spaced numbasis
            numbasis = np.logspace(0, np.log10(self.db.stamps_tab.shape[0]-1), 20)
            numbasis = np.unique(numbasis.astype(np.int))
            self.klip_args_dict['numbasis'] = numbasis
            #numbasis = np.arange(1, self.db.stamps_tab.shape[0]-1, 20)
        subtr_mapper = lambda x: self.subtr_klip(x,
                                                 kwargs=self.klip_args_dict)
        results = self.db.stamps_tab.set_index('stamp_id', drop=False).apply(subtr_mapper, axis=1)
        self.psf_subtr = results.apply(lambda x: x.residuals)
        self.psf_model = results.apply(lambda x: x.models)
        self.subtr_refs = results.apply(lambda x: x.ref_ids).T#.apply(lambda x: pd.Series(x)).T



    def _table_apply_wrapper(self, targ_row, func, func_args={}):
        """
        Wrapper for all the subtraction functions, so that output is unified?
        This is only used in the context of self.db.stamps_tab.apply

        Parameters
        ----------
        func : class method [nmf]
          the function that performs PSF subtraction, given the target, references,
          and other arguments
        targ_row : pd.DataFrame row / pd.Series
          the row of the stamps_table to be subtracted
        func_args : {}
          extra arguments to pass to the subtraction function

        Output
        ------
        results : namedtuple
          namedtuple with three attributes:
          1. ref_ids - list of reference stamps used
          2. models - the PSF models, by component
          3. residuals - PSF subtraction residuals (targ - models)
        """
        targ_id = targ_row['stamp_id']
        targ_stamp = targ_row['stamp_array']
        # get the reference stamps
        ref_stamps = self._get_reference_stamps(targ_row['stamp_id'])
        #ref_stamps = ref_stamps.apply(np.ravel)
        #shared_utils.debug_print(ref_ids)
        # excellent use of namedtuple here, pat yourself on the back!
        results = namedtuple('results', ('ref_ids', 'residuals', 'models'))
        results.ref_ids = pd.Series(ref_stamps.index)
        # psf_subtraction
        results.residuals, results.models = func(targ_stamp, ref_stamps, func_args)
        # update the reference table
        self.reference_table[targ_id] = False
        self.reference_table.loc[results.ref_ids, targ_id] = True
        return results


    def subtr_klip(self, targ_row, kwargs={}):
        """
        Designed to use stamps_tab.apply to do klip subtraction to a whole table of stamps
        Assumes return_basis is True; otherwise, fails
        Returns a named tuple with fields residuals, models, and ref_ids
        """
        target_stamp = targ_row['stamp_array']
        # get the reference stamps
        ref_stamps = self._get_reference_stamps(targ_row['stamp_id'])
        #shared_utils.debug_print(ref_ids)
        # excellent use of namedtuple here, pat yourself on the back!
        results = namedtuple('klip_results', ('residuals', 'models'))
        results.ref_ids = pd.Series(ref_stamps.index)
        kwargs['restore_scale'] = kwargs.get('restore_scale', False)
        restore_scale = kwargs.pop('restore_scale')
        results.residuals, results.models =  klip_subtr_wrapper(target_stamp, ref_stamps,
                                                                restore_scale,
                                                                kwargs)
        return results


    def subtr_nfm(self, targ, refs, kwargs={}):
        """
        Perform NMF subtraction on one target and its references

        Parameters
        ----------
        targ : target stamp
        refs : pd.Series of reference stamps
        kwargs : {}
          other arguments to pass to sklearn's NMF

        Output
        ------
        residuals, psf_models
        """
        # set default arguments
        for k, default in self.nmf_args_dict.items():
            kwargs[k] = kwargs.get(k, default)
        # prep reference images
        # drop refs with negative values
        neg_stamps = refs[refs.apply(lambda x: np.any(x < 0))].index
        #shared_utils.debug_print(f"Neg stamps: {len(neg_stamps)}")
        refs = refs.drop(neg_stamps, inplace=False)
        # flatten
        refs_flat = np.stack(refs.apply(np.ravel))
        # generate the PSF model from the transformed data and components
        nmf_model = NMF(n_components=targ.size, **kwargs)
        trans_data = nmf_model.fit_transform(refs_flat)
        nmf_components = nmf_model.components_
        psf_models = image_utils.make_image_from_flat(np.dot(trans_data, nmf_components))
        residuals = targ - psf_models
        # add an index
        residuals = pd.Series({i+1: r for i, r in enumerate(residuals)})
        psf_models = pd.Series({i+1: r for i, r in enumerate(psf_models)})

        return residuals, psf_models


