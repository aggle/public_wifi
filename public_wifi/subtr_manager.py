"""
Utilities for PSF subtraction:
- PSF correlation metrics
- KLIP utilities and wrappers
- KLIP PSF model reconstruction
"""

import numpy as np
import pandas as pd
from collections import namedtuple
import functools
from pathlib import Path

# RDI imports
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import sys

# local imports
from .utils import shared_utils
from .utils import image_utils

# PSF Subtraction Modules
# pw_config = (Path(__file__).parent.absolute() / "./config-public_wifi.cfg").resolve()
# # pyklip
# sys.path.append(shared_utils.load_config_path("extern", "pyklip_path",
#                                               as_str=True,
#                                               config_file=pw_config))
# try:
# from pyklip import klip
# except ModuleNotFoundError:
    # print("Error loading pyklip: did you forget to run sys.append with the pyklip path?")
# NMF
# sys.path.append(shared_utils.load_config_path("extern", "nmf_path",
#                                               as_str=True,
#                                               config_file=pw_config))
# try:
# from NonnegMFPy import nmf as NMFPy
# except ModuleNotFoundError:
#     print("Error loading NonnegMFPy: did you forget to run sys.append with the NMF path?")

from pyklip import klip
from NonnegMFPy import nmf as NMFPy

# Results object
Results = namedtuple('Results', ('residuals', 'models', 'references'))


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
                               data_range = np.ptp(np.stack([targ, references]))
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
                            dtype=float)
    # rescale the stamps to max value = 1
    if rescale == True:
        stamps = stamps.apply(lambda x: x/np.nanmax(x))
    # # calculate an upper triangular matrix of correlations
    # other_stamps = list(stamps.index)
    # for i, stamp_id in enumerate(corr_mat.index[:-1]):
    #     targ = stamps[stamp_id]
    #     # remove target stamp from the references
    #     #other_stamps = list(corr_mat.index)
    #     other_stamps.pop(0)
    #     corr = corr_func(targ, np.stack(stamps[other_stamps], axis=0),
    #                      kw_args=corr_func_args)
    #     corr_mat.loc[stamp_id, other_stamps] = corr

    # # now fill in the empty half of the matrix - beware of NaN's
    # corr_mat = corr_mat.add(corr_mat.T, fill_value=0)

    corr_mat = stamps.apply(
        lambda targ: stamps.apply(
            lambda ref: corr_func(targ, ref, **corr_func_args)
        )
    )
    corr_mat.columns.name = 'reference_' + corr_mat.columns.name 
    corr_mat.index.name = 'target_' + corr_mat.index.name
    return corr_mat


####################
# KLIP subtraction #
####################

def klip_subtr_wrapper(target_stamp, refs_table, klip_args={}):
    """
    Wrapper for RDI klip subtraction so you can pass in a table without extra processing

    Parameters
    ----------
    target_stamp: MxN np.array
      stamp to be subtracted, as an image
    refs_table: pd.Series
      pandas series containing the reference stamps (index can be the reference IDs)
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
    targ_stamp_flat = targ_stamp_flat #/ target_scale
    ref_stamps_flat = ref_stamps_flat #* (target_scale / ref_stamps_scale)
    # apply KLIP
    kl_max = np.array([len(ref_stamps_flat)-1])
    #numbasis = klip_args.get('numbasis', kl_max)
    numbasis = np.arange(1, len(ref_stamps_flat))
    shared_utils.debug_print(False, f"{kl_max}, {numbasis}")
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

    if return_basis == True:
        return kl_sub_img, kl_basis_img
    else:
        return kl_sub_img


def psf_model_from_basis(target, kl_basis, numbasis=None):
    """
    Generate a model PSF from the KLIP basis vectors. See Soummer et al 2012.

    Parameters
    ----------
    target : np.array
      the target PSF, 1-D
    kl_basis : np.array
      Nklip x Npix array
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
    numbasis = numbasis.astype(int)

    coeffs = np.inner(target, kl_basis)
    psf_model = kl_basis * np.expand_dims(coeffs, [i+1 for i in range(kl_basis.ndim-1)])
    psf_model = np.array([np.sum(psf_model[:k], axis=0) for k in numbasis])

    return np.squeeze(psf_model)


class ZeroRefsError(Exception):
    """Raised when there are zero references found"""
    # Constructor method
    def __init__(self, value):
        self.value = value
        # __str__ display function
    def __str__(self):
        return(repr(self.value))

class SubtrManager:
    """
    Class to manage PSF subtraction for a self-contained set of stamps.
    Initializes with a DBManager instance as argument. Optionally, calculate the PSF correlations
    """

    def __init__(self, db_manager, calc_corr_flag=False, instrument=None):
        # calculate all three correlation matrices
        self.db = db_manager
        self.instr = instrument
        # table for tracking references used
        cols = pd.Index(self.db.stamps_tab['stamp_id'], name='targ_id')
        indx = pd.Index(self.db.stamps_tab['stamp_id'], name='ref_id')
        self.reference_table = pd.DataFrame(None, columns=cols, index=indx, dtype=bool)
        # correlation function arguments
        self.corr_func_args_dict = {'mse': {},
                                    'pcc': {},
                                    'ssim': {'win_size': 5.}}
        # klip arguments
        self.klip_args_dict = {'return_numbasis': True,
                               'numbasis': None}
        self.nmf_args_dict = {'verbose': False,
                              'n_components': 10,
                              'ordered': True}

        if calc_corr_flag == True:
            self.calc_psf_corr()

        if self.instr is not None and hasattr(self.instr, 'stamp_size'):
            self._stamp_mask = np.ones((self.instr.stamp_size, self.instr.stamp_size)) # should not be hard-coded
    ###############
    # stamp masks #
    ###############
    @property
    def stamp_mask(self):
        try:
            return self._stamp_mask
        except AttributeError:
            return None
    @stamp_mask.setter
    def stamp_mask(self, newval):
        """
        Set a new mask but do not apply it to the stamps
        """
        self._stamp_mask = newval
        self._stamp_mask_ind = np.where(newval == 1)

    @property
    def stamp_mask_ind(self):
        try:
            return self._stamp_mask_ind
        except AttributeError:
            return None

    ###########
    # methods #
    ###########
    def calc_psf_corr(self):
        """
        Compute the correlation matrices. Sets self.corr_mats, a namedtuple
        with all the matrices as attributes
        """
        # initialize the contained to hold the correlation matrices
        corr_mats = namedtuple('corr_mats', ('mse','pcc','ssim'))
        # set the stamp ID as the index
        stamps = self.db.stamps_tab.set_index('stamp_id')['stamp_array'].squeeze()
        corr_mats.mse = calc_corr_mat(stamps, calc_refcube_mse,
                                      self.corr_func_args_dict['mse'])
        corr_mats.pcc = calc_corr_mat(stamps, calc_refcube_pcc,
                                      self.corr_func_args_dict['pcc'])
        corr_mats.ssim = calc_corr_mat(stamps, calc_refcube_ssim,
                                       self.corr_func_args_dict['ssim'])
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
        drop_cols = self.psf_subtr.map(filtfunc).all(axis=0)
        drop_cols = drop_cols[drop_cols].index
        self.psf_subtr.drop(drop_cols, axis=1, inplace=True)


    def get_reference_stamps(self, targ_stamp_id, ids_only=False, dmag_max=1):
        """
        Given a target stamp, find the list of appropriate reference stamps
        using e.g. the star_id and the stamp_ref_flag value
        Raises an exception if no references are found

        Parameters
        ----------
        targ_stamp_id : str
          the stamp_id of the PSF targeted for subtraction
        ids_only : bool [False]
          if True, return only the stamp labels, not the arrays
        dmag_max : int [1]
          absolute delta magnitude limit on references

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

        # construct a table with all the necessary query information
        # start with the standard stamps table
        query_table = self.db.stamps_tab.copy()
        # merge it with the ps_table
        query_table = query_table.merge(self.db.find_matching_id(query_table['stamp_id'], 'ps'),
                                        on='stamp_id')
        query_table = query_table.merge(self.db.ps_tab, on='ps_id')

        # apply the dmag cut
        targ_mag = self.db.find_stamp_mag(targ_stamp_id)
        dmag_bool = query_table['ps_mag'].apply(lambda x: np.abs(x - targ_mag) <= dmag_max)
        query_table = query_table.loc[dmag_bool]
        # reject all stamps with a bad reference flag
        ref_query = "stamp_ref_flag == True"

        # reject all references that correspond to the target's parent star
        # add the stars table
        query_table = query_table.merge(self.db.find_matching_id(query_table['stamp_id'], 'star'),
                                        on='stamp_id')
        query_table = query_table.merge(self.db.stars_tab, on='star_id')
        targ_star_id = self.db.find_matching_id(targ_stamp_id, 'star')
        ref_query += " and stamp_star_id != @targ_star_id"
        # finally, apply the query!
        ref_stamps = query_table.query(ref_query)
        if ids_only == True:
            return ref_stamps['stamp_id']
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
        results = namedtuple('subtr_results', ('residuals','models','references', 'failed'))
        subtr_mapper = lambda x: self._table_apply_wrapper(x, func, func_args)
        agg_results = self.db.stamps_tab.set_index('stamp_id', drop=False).apply(subtr_mapper, axis=1)
        results.residuals = agg_results.apply(lambda x: x.residuals)
        results.models = agg_results.apply(lambda x: x.models)
        results.references = agg_results.apply(lambda x: x.ref_ids)
        
        # subtractions that failed are NaN across the entire row
        failed_references = list(results.references[results.references.isna().all(axis=1)].index)
        failed_models = list(results.models[results.references.isna().all(axis=1)].index)
        failed_residuals = list(results.residuals[results.residuals.isna().all(axis=1)].index)
        # uniquify
        results.failed = list(set(failed_residuals + failed_models + failed_references))
        
        # drop the failed subtractions
        results.references.dropna(axis=0, how='all', inplace=True)
        results.models.dropna(axis=0, how='all', inplace=True)
        results.residuals.dropna(axis=0, how='all', inplace=True)

        # return results
        self.subtr_results = results
        

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
            numbasis = np.unique(numbasis.astype(int))
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
        results = namedtuple('results', ('ref_ids', 'residuals', 'models'))

        targ_id = targ_row['stamp_id']
        targ_stamp = targ_row['stamp_array']
        # get the reference stamps
        ref_stamps = self.get_reference_stamps(targ_row['stamp_id'], dmag_max=1)
        # proceed to PSF subtraction
        # excellent use of namedtuple here, pat yourself on the back!
        results.ref_ids = pd.Series(ref_stamps.index)
        # psf_subtraction
        try:
            results.residuals, results.models = func(targ_stamp, ref_stamps, func_args)
        except ZeroRefsError as e:
            results.residuals, results.models = (pd.Series(np.nan),)*2
        # update the reference table
        self.reference_table[targ_id] = False # reset column to False
        self.reference_table.loc[results.ref_ids, targ_id] = True
        return results


    def subtr_klip(self, targ, refs, kwargs={}):
        """
        Designed to use stamps_tab.apply to do klip subtraction to a whole table of stamps
        Assumes return_basis is True; otherwise, fails
        Returns a named tuple with fields residuals, models, and ref_ids
        """
        try:
            assert(len(refs) > 0)
        except AssertionError:
            raise ZeroRefsError("Error: No references given!")
        #target_stamp = targ_row['stamp_array']
        ## get the reference stamps
        #ref_stamps = self.get_reference_stamps(targ_row['stamp_id'], dmag_max=1)
        
        # excellent use of namedtuple here, pat yourself on the back!
        results = namedtuple('klip_results', ('residuals', 'models'))
        if kwargs == {}:
            kwargs = self.klip_args_dict
        residuals, models =  klip_subtr_wrapper(targ, refs,
                                                kwargs)
        return residuals, models


    def subtr_nmf(self, targ, refs, kwargs={}):
        """
        Perform NMF subtraction on one target and its references

        Parameters
        ----------
        targ : 2-D target stamp
        refs : pd.Series of reference stamps
        kwargs : {}
          other arguments, some to pass to NonnegNMFPy's NMFPy.SolveNMF

        Output
        ------
        tuple with residuals and psf_models
        """
        # prep reference images
        try:
            assert(len(refs) > 0)
        except AssertionError:
            raise ZeroRefsError("Error: No references given!")
        shared_utils.debug_print(False, f'Nrefs = {len(refs)}, continuing...')

        # synchronize the global argument dictionary and the passed one
        # self.nmf_args_dict serves as a record; kwargs is used here
        self.nmf_args_dict.update(kwargs)
        kwargs.update(self.nmf_args_dict)

        # flatten
        refs_flat = np.stack(refs.apply(np.ravel))
        # generate the PSF model from the transformed data and components
        nrefs, npix = refs_flat.shape


        # get the number of free parameters
        if kwargs.get('n_components', None) is None:
            kwargs['n_components'] = nrefs
        n_components = kwargs.pop('n_components')


        try:
            ordered = kwargs.pop('ordered')
        except KeyError:
            ordered = False
        if ordered == True:
            # this bit copied from Bin's nmf_imaging (https://github.com/seawander/nmf_imaging)
            # initialize
            W_ini = np.random.rand(nrefs, nrefs)
            H_ini = np.random.rand(nrefs, npix)
            g_refs = NMFPy.NMF(refs_flat, n_components=1)
            W_ini[:, :1] = g_refs.W[:]
            H_ini[:1, :] = g_refs.H[:]
            for n in range(1, n_components+1):
                if verbose == True:
                    print("\t" + str(n) + " of " + str(n_components))
                W_ini[:, :(n-1)] = np.copy(g_refs.W)
                W_ini = np.array(W_ini, order = 'F') #Fortran ordering
                H_ini[:(n-1), :] = np.copy(g_refs.H)
                H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
                g_refs = NMFPy.NMF(refs_flat, W=W_ini[:, :n], H=H_ini[:n, :], n_components=n)
                chi2 = g_refs.SolveNMF(**kwargs)
        else:
            g_refs = NMFPy.NMF(refs_flat, n_components=n_components)
            g_refs.SolveNMF(**kwargs)
        # now you have to find the coefficients to scale the components to your target
        g_targ = NMFPy.NMF(targ.ravel()[None, :], H=g_refs.H, n_components=n_components)
        g_targ.SolveNMF(W_only=True)
        # create the models by component using some linalg tricks
        W = np.tile(g_targ.W, g_targ.W.shape[::-1])
        psf_models = np.dot(np.tril(W), g_targ.H)
        psf_models = image_utils.make_image_from_flat(psf_models)
        residuals = targ - psf_models
        # add an index
        residuals = pd.Series({i+1: r for i, r in enumerate(residuals)})
        psf_models = pd.Series({i+1: r for i, r in enumerate(psf_models)})

        return residuals, psf_models

    def get_targets_references_by_star(self, targ_star, dmag=None):
        """
        Used for when you want to do the subtraction by star, not by target

        targ_star : str
          star_id of the target
        dmag : [None] | float
          dmag range cut on the references. If none, no magnitude cut applied
        """

        full_table = self.db.join_all_tables()
        targ_group = full_table.query('star_id == @targ_star')
        targ_stamps = targ_group.set_index('stamp_id')['stamp_array']
        # select all the *other* stars
        ref_stars = full_table.query('star_id != @targ_star')

        # drop bad references
        ref_stars = ref_stars.query('stamp_ref_flag == True')

        # select by magnitude
        if dmag is not None:
            mag_llim = targ_group['ps_mag'].min() - dmag
            mag_ulim = targ_group['ps_mag'].max() + dmag
            ref_stars = ref_stars.query(f"ps_mag <= {mag_ulim} and ps_mag >= {mag_llim}")

        ref_stamps = ref_stars.set_index('stamp_id')['stamp_array']
        return targ_stamps, ref_stamps


    def subtr_nmf_one_star(self, targ_star, kwargs={}, ref_args={}):
        """
        Same as subtr_nmf except it operates grouping by star ID, since
        all the detections of the same star will in principle use the same
        reference stamps
        Speeds up by almost an order of magnitude
        Still needs a wrapper in the end to organize the output into a single table

        Parameters
        ----------
        targ_star : star_id of the target star
        ref_args : dict of arguments to pass to get_targets_references_by_star
        kwargs : dict of NMF arguments

        Output
        ------
        results : named tuple
          tuple has attributes residuals, models, and references. Each has a dataframe
          of the results
        """
        targ_stamps, ref_stamps = self.get_targets_references_by_star(targ_star)

        # build sequenial NMF model
        try:
            assert(len(ref_stamps) > 0)
        except AssertionError:
            raise ZeroRefsError("Error: No references given!")
        shared_utils.debug_print(False, f'Nrefs = {len(ref_stamps)}, continuing...')

        # synchronize the global argument dictionary and the passed one
        # self.nmf_args_dict serves as a record; kwargs is used here
        self.nmf_args_dict.update(kwargs)
        kwargs.update(self.nmf_args_dict)

        # flatten
        refs_flat = np.stack(ref_stamps.apply(np.ravel))
        # generate the PSF model from the transformed data and components
        nrefs, npix = refs_flat.shape

        # verbosity
        verbose = kwargs.get('verbose', False)

        # get the number of free parameters
        if kwargs.get('n_components', None) is None:
            kwargs['n_components'] = nrefs
        n_components = kwargs.pop('n_components')

        try:
            ordered = kwargs.pop('ordered')
        except KeyError:
            ordered = False
        if ordered == True:
            # this bit copied from Bin's nmf_imaging (https://github.com/seawander/nmf_imaging)
            # initialize
            W_ini = np.random.rand(nrefs, nrefs)
            H_ini = np.random.rand(nrefs, npix)
            g_refs = NMFPy.NMF(refs_flat, n_components=1)
            W_ini[:, :1] = g_refs.W[:]
            H_ini[:1, :] = g_refs.H[:]
            for n in range(1, n_components+1):
                if verbose == True:
                    print("\t" + str(n) + " of " + str(n_components))
                W_ini[:, :(n-1)] = np.copy(g_refs.W)
                W_ini = np.array(W_ini, order = 'F') #Fortran ordering
                H_ini[:(n-1), :] = np.copy(g_refs.H)
                H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
                g_refs = NMFPy.NMF(refs_flat, W=W_ini[:, :n], H=H_ini[:n, :], n_components=n)
                chi2 = g_refs.SolveNMF(**kwargs)
        else:
            g_refs = NMFPy.NMF(refs_flat, n_components=n_components)
            g_refs.SolveNMF(**kwargs)

        # now you have to find the coefficients to scale the components to your targets
        g_targ = NMFPy.NMF(np.stack(targ_stamps.apply(np.ravel)), H=g_refs.H, n_components=n_components)
        g_targ.SolveNMF(W_only=True)
        # create the models by component using some linalg tricks
        W = np.array([np.tile(i[None], i[None].shape[::-1]) for i in g_targ.W])
        psf_models = np.dot(np.tril(W), g_targ.H)
        psf_models = image_utils.make_image_from_flat(psf_models)

        # convert to series and compute residuals
        psf_models = pd.Series({i: j for i, j in zip(targ_stamps.index, psf_models)})
        residuals = targ_stamps - psf_models
        # now convert to dataframes with the stamp on the column and the component on the index
        dict_func = lambda x: dict(zip(np.arange(1, n_components+1), x))
        psf_models = pd.DataFrame.from_dict(psf_models.apply(dict_func).to_dict(), orient='index')
        residuals = pd.DataFrame.from_dict(residuals.apply(dict_func).to_dict(), orient='index')
        references = pd.DataFrame.from_dict({targ: ref_stamps.index for targ in targ_stamps.index},
                                            orient='index')
        results = Results(residuals=residuals,
                          models=psf_models,
                          references=references)
        return results

    def subtr_nmf_by_star(self, nmf_args={}, return_results=False):
        """
        This function aggregates the results from subtr_nmf_one_star
        Sets self.subtr_results
        """
        # list of stars
        stars = self.db.stars_tab['star_id'].unique()
        #print(stars)
        residuals, models, references = {}, {}, {}
        for i, star in enumerate(stars):
            result = self.subtr_nmf_one_star(star, kwargs=nmf_args, return_results=return_results)
            residuals[star] = result.residuals
            models[star] = result.models
            references[star] = result.references 

        all_results = Results(residuals=pd.concat(residuals, names=['star_id', 'stamp_id']),
                              models=pd.concat(models, names=['star_id', 'stamp_id']),
                              references=pd.concat(references, names=['star_id', 'stamp_id']))
        self.subtr_results = all_results
        if return_results == True:
            return all_results

    def subtr_by_star(self, subtr_func, arg_dict, return_results=False, verbose=False):
        """
        Aggregator for results for the subtraction functions

        Parameters
        ----------
        func : the subtraction function, takes only the name of a star (and other arguments)
        arg_dict : dictionary of keyword arguments
        return_results : bool [False]
          if True, return the results tuple. Otherwise, just sets self.subtr_results

        Output
        ------
        named tuple of results with attributes residuals, models, references

        """
        # list of stars
        stars = self.db.stars_tab['star_id'].unique()
        #print(stars)
        residuals, models, references = {}, {}, {}
        for i, star in enumerate(stars):
            try:
                result = subtr_func(star, kwargs=arg_dict)
            except ZeroRefsError:
                # if there are no references, create an empty dataframe
                # for this star
                stamp_ids = self.db.stamps_tab.query("stamp_star_id == @star")['stamp_id']
                empty_df = pd.DataFrame(index=stamp_ids)
                result = Results(residuals=empty_df,
                                 models=empty_df,
                                 references=empty_df)
            residuals[star] = result.residuals
            models[star] = result.models
            references[star] = result.references 
            if verbose == True:
                print(f"{i+1}/{len(stars)} stars complete")

        all_results = Results(residuals=pd.concat(residuals, names=['star_id', 'stamp_id']),
                              models=pd.concat(models, names=['star_id', 'stamp_id']),
                              references=pd.concat(references, names=['star_id', 'stamp_id']))
        self.subtr_results = all_results
        if return_results == True:
            return all_results


    def subtr_klip_one_star(self, targ_star, kwargs={}, ref_args={}):
        """
        Perform KLIP subtraction on all the stamps for one star. Since stamps
        of the same star share the same references, this computes the Z_k's for
        a star and then projects all the star stamps onto them.

        Parameters
        ----------
        targ_star : star ID of the target (S000000)
        kwargs : {}
         dict of args to pass to KLIP subtraction. Default: self.klip_args_dict
        ref_args : {}
         dict of args to pass to get_targets_references_by_star, like dmag cut.

        Output
        ------
        results : named tuple
          tuple has attributes residuals, models, and references. Each has a dataframe
          of the results

        """
        targ_stamps, ref_stamps = self.get_targets_references_by_star(targ_star)

        # build sequenial NMF model
        try:
            assert(len(ref_stamps) > 0)
        except AssertionError:
            raise ZeroRefsError("Error: No references given!")
        shared_utils.debug_print(False, f'Nrefs = {len(ref_stamps)}, continuing...')

        # synchronize the global argument dictionary and the passed one
        # self.klip_args_dict serves as a record; kwargs is used here
        self.klip_args_dict.update(kwargs)
        kwargs.update(self.klip_args_dict)

        # flatten
        targ_stamps_flat = targ_stamps.apply(image_utils.flatten_image_axes)
        ref_stamps_flat = image_utils.flatten_image_axes(np.stack(ref_stamps))

        # apply KLIP
        kl_max = np.array([len(ref_stamps_flat)-1])
        #numbasis = klip_args.get('numbasis', kl_max)
        numbasis = np.arange(1, len(ref_stamps_flat)-1)
        shared_utils.debug_print(False, f"{kl_max}, {numbasis}")
        return_basis = True
        klip_results = targ_stamps_flat.apply(lambda x: klip.klip_math(x,
                                                                       ref_stamps_flat,
                                                                       numbasis = numbasis,
                                                                       return_basis = return_basis))
        # subtraction results
        residuals = klip_results.apply(lambda x: pd.Series(dict(zip(numbasis, x[0].T))))
        residuals = residuals.map(image_utils.make_image_from_flat)
        # generate PSF models and store in dataframe
        # klip basis
        klip_basis = klip_results.apply(lambda x: pd.Series(dict(zip(numbasis, x[1]))))
        model_gen_df = pd.merge(targ_stamps_flat, klip_basis, left_index=True, right_index=True)
        psf_models = model_gen_df.apply(lambda x: psf_model_from_basis(x['stamp_array'],
                                                                       np.stack(x[numbasis]),
                                                                       numbasis=numbasis),
                                        axis=1)
        psf_models = psf_models.apply(lambda x: pd.Series(dict(zip(numbasis, x))))
        psf_models = psf_models.map(image_utils.make_image_from_flat)

        # references
        references = pd.DataFrame.from_dict({targ: ref_stamps.index for targ in targ_stamps.index},
                                            orient='index')

        results = Results(residuals=residuals,
                          models=psf_models,
                          references=references)
        return results


    def subtr_klip_by_star(self, klip_args={}, return_results=False):
        """
        This function aggregates the results from subtr_klip_one_star
        Sets self.subtr_results
        """
        # list of stars
        stars = self.db.stars_tab['star_id'].unique()
        #print(stars)
        residuals, models, references = {}, {}, {}
        for star in stars:
            result = self.subtr_klip_one_star(star, kwargs=klip_args)
            residuals[star] = result.residuals
            models[star] = result.models
            references[star] = result.references 

        
        all_results = Results(residuals=pd.concat(residuals, names=['star_id', 'stamp_id']),
                              models=pd.concat(models, names=['star_id', 'stamp_id']),
                              references=pd.concat(references, names=['star_id', 'stamp_id']))
        self.subtr_results = all_results
        if return_results == True:
            return all_results

def subtr_nmf_sklearn(targ, refs, kwargs={}):
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
    # prep reference images
    # drop refs with negative values
    neg_stamps = refs[refs.apply(lambda x: np.any(x < 0))].index
    
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


