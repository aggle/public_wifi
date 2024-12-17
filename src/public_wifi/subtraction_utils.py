"""
Tools for performing PSF subtraction
"""

import numpy as np
import pandas as pd

from pyklip import klip
from sklearn.decomposition import NMF
from NonnegMFPy import nmf as NMFPy

def normalized_psf(stamp):
    stamp = stamp - np.nanmin(stamp)
    stamp = stamp/np.nansum(stamp)
    return stamp



def klip_subtract(
        target_stamp,
        reference_stamps,
        numbasis = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Perform KLIP subtraction on the target stamp.
    Returns the subtracted images, and the PSF models
    """
    stamp_shape = target_stamp.shape
    if numbasis is None:
        numbasis = np.array([len(reference_stamps)-1])
    targ_stamp_flat = target_stamp.ravel()
    ref_stamps_flat = np.stack([i.ravel() for i in reference_stamps])

    kl_sub, kl_basis = klip.klip_math(
        targ_stamp_flat, ref_stamps_flat,
        numbasis = numbasis,
        return_basis = True,
    )
    # construct the PSF model
    coeffs = np.inner(targ_stamp_flat, kl_basis)
    klip_model = kl_basis * np.expand_dims(coeffs, [i+1 for i in range(kl_basis.ndim-1)])
    klip_model = np.array([np.sum(klip_model[:k], axis=0) for k in numbasis])

    # store as Series objects
    if isinstance(numbasis, int):
        numbasis = [numbasis]
    kl_basis = pd.Series(dict(zip(range(1, len(kl_basis)+1), kl_basis)), name='kl_basis')
    kl_basis.index.name = 'numbasis'
    kl_sub = pd.Series(dict(zip(numbasis, kl_sub.T)), name='kl_sub')
    kl_sub.index.name = 'numbasis'
    klip_model = pd.Series(dict(zip(numbasis, klip_model)), name='klip_model')
    klip_model.index.name = 'numbasis'
    # return the subtracted stamps as images
    kl_basis_img = kl_basis.apply(lambda img: img.reshape(stamp_shape))
    kl_sub_img = kl_sub.apply(lambda img: img.reshape(stamp_shape))
    klip_model_img = klip_model.apply(lambda img: img.reshape(stamp_shape))
    return kl_basis_img, kl_sub_img, klip_model_img

def nmf_subtract(
        target_stamp : np.ndarray,
        reference_stamps : pd.Series,
        numbasis : int | np.ndarray | None = None,
        n_components : int | None = None,
        verbose=False,
):
    """
    Perform NMF subtraction on one target and its references

    Parameters
    ----------
    target_stamp : np.ndarray
      2-D target stamp
    reference_stamps : pd.Series[np.ndarray]
      the reference PSFs
    kwargs : {}
      other arguments, some to pass to NonnegNMFPy's NMFPy.SolveNMF

    Output
    ------
    tuple with residuals and psf_models
    """
    stamp_shape = target_stamp.shape
    if numbasis is None:
        numbasis = len(reference_stamps)-1
    if isinstance(numbasis, int):
        numbasis = np.array([int])
    # flatten the stamps
    targ_stamp_flat = target_stamp.ravel()
    ref_stamps_flat = np.stack([i.ravel() for i in reference_stamps])

    nrefs, npix = ref_stamps_flat.shape


    # get the number of free parameters
    if n_components is None:
        n_components = int(nrefs)

    # # this bit copied from Bin's nmf_imaging
    # (https://github.com/seawander/nmf_imaging) initialize

    W_ini = np.random.rand(nrefs, nrefs)
    H_ini = np.random.rand(nrefs, npix)
    g_refs = NMFPy.NMF(ref_stamps_flat, n_components=1)
    W_ini[:, :1] = g_refs.W[:]
    H_ini[:1, :] = g_refs.H[:]
    for n in range(1, n_components+1):
        if verbose == True:
            print("\t" + str(n) + " of " + str(n_components))
        W_ini[:, :(n-1)] = np.copy(g_refs.W)
        W_ini = np.array(W_ini, order = 'F') #Fortran ordering
        H_ini[:(n-1), :] = np.copy(g_refs.H)
        H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
        g_refs = NMFPy.NMF(ref_stamps_flat, W=W_ini[:, :n], H=H_ini[:n, :], n_components=n)
        # chi2 = g_refs.SolveNMF(**kwargs)

    # # now you have to find the coefficients to scale the components to your target
    # g_targ = NMFPy.NMF(target_stamp.ravel()[None, :], H=g_refs.H, n_components=n_components)
    # g_targ.SolveNMF(W_only=True)
    # # create the models by component using some linalg tricks
    # W = np.tile(g_targ.W, g_targ.W.shape[::-1])
    # psf_models = np.dot(np.tril(W), g_targ.H)
    # psf_models = image_utils.make_image_from_flat(psf_models)
    # residuals = target_stamp - psf_models
    # # add an index
    # residuals = pd.Series({i+1: r for i, r in enumerate(residuals)})
    # psf_models = pd.Series({i+1: r for i, r in enumerate(psf_models)})
    # return residuals, psf_models
