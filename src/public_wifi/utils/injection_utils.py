"""
Methods for finding sources and establishing detection limits.
"""

import numpy as np
import pandas as pd

"""
Architecture:
Find something
"""
def get_radial_coordinate(shape, center=None):
    """
    Transform a set of x-y coordinates into a set of radius-phi coordinates
    Args:
      shape: the image shape
      center: the center; if None, then do shape/2
    Returns:
      Nx x Ny grid of radii
      Nx x Ny grid of angles (0-2*pi)
    """
    if center is None:
        center = np.array(shape)/2.
    grid = np.mgrid[:float(shape[0]), :float(shape[1])]
    grid = np.rollaxis(np.rollaxis(grid, 0, grid.ndim) - center + 0.5, -1, 0)
    rad2D = np.linalg.norm(grid, axis=0)
    phi2D = np.arctan2(grid[0], grid[1]) + np.pi  # 0 to 2*pi
    return rad2D, phi2D

def get_stamp_coordinates(center, drow, dcol, imshape, nanpad=False):
    """
    get pixel coordinates for a stamp with width dcol, height drow, and center `center` embedded
    in an image of dimensions imshape.
    This is a helper function - use get_stamp_from_image or get_stamp_from_cube
    Arguments:
        center: (row, col) center of the stamp
        drow: height of stamp
        dcol: width of stamp
        imshape: total size of image the stamp is a part of
    Returns:
        img_coords: the stamp indices for the full image array (i.e. stamp = img[img_coords[0], img_coords[1]xo])
        stamp_coords: the stamp indices for selecting the part of the stamp
                      that goes in the image (i.e. stamp[stamp_coords[0], stamp_coords[1]]).
                      this is relevant for stamps on the edge of the images -
                      e.g. if you want to matched filter the image edge, you should
                      plug these coordinates into your matched filter stamp to
                      select only the relevant part of the matched filter
    """
    # handle odd and even: 1 if odd, 0 if even
    oddflag = np.array((dcol%2, drow%2))
    colrad = int(np.floor(dcol))/2
    rowrad = int(np.floor(drow))/2

    rads = np.array([rowrad, colrad], dtype=int)
    center = np.array([center[0],center[1]],dtype=int) #+ oddflag
    img = np.zeros(imshape)
    stamp = np.ones((drow,dcol))
    full_stamp_coord = np.indices(stamp.shape) + center[:,None,None]  - rads[:,None,None]
    # check for out-of-bounds values
    # boundaries
    row_lb,col_lb = (0, 0)
    row_hb,col_hb = imshape

    rowcheck_lo, colcheck_lo = (center - rads)
    rowcheck_hi, colcheck_hi = ((imshape-center) - rads) - oddflag[::-1]

    row_start, col_start = 0,0
    row_end, col_end = stamp.shape

    if rowcheck_lo < 0:
        row_start = -1*rowcheck_lo
    if colcheck_lo < 0:
        col_start = -1*colcheck_lo
    if rowcheck_hi < 0:
        row_end = rowcheck_hi
    if colcheck_hi < 0:
        col_end = colcheck_hi

    # pull out the selections
    img_coords = full_stamp_coord[:,row_start:row_end,col_start:col_end]
    stamp_coords = np.indices(stamp.shape)[:,row_start:row_end,col_start:col_end]
    return (img_coords, stamp_coords)




#################
# PSF INJECTION #
#################
def _inject_psf(img, psf, center, scale_flux=None, subtract_mean=False, return_flat=False):
    """
    !!! use inject_psf() as a wrapper, do not call this function directly !!!
    Inject a PSF into an image at a location given by center. Optional: scale PSF.
    Input:
        img: 2-D img or 3-D cube. Last two dimensions define an img (i.e. [(Nimg,)Nx,Ny])
        psf: 2-D img or 3-D cube, smaller than or equal to img in size. If cube, 
             must have same 1st dimension as img 
        center: center of the injection in the image
        scale_flux: multiply the PSF by this number. If this is an array,
             img and psf will be tiled to match its length
             if scale_flux is None, don't scale psf
        subtract_mean: (False) mean-subtract before returning
        return_flat: (False) flatten the array along the pixels axis before returning
        hpf [None]: pass a high-pass filter width for high-pass filtering.
                    If used, the psf must be the *unfiltered* version
    Returns:
       injected_img: 2-D image or 3D cube with the injected PSF(s)
       injection_psf: (if return_psf=True) 2-D normalized PSF full image
    """

    if scale_flux is None:
        scale_flux = np.array([1])
    elif np.ndim(scale_flux) == 0:
        scale_flux = np.array([scale_flux])
    scale_flux = np.array(scale_flux)

    # get the right dimensions
    img_tiled = np.tile(img, (np.size(scale_flux),1,1))
    psf_tiled = np.tile(psf, (np.size(scale_flux),1,1))

    # get the injection pixels
    injection_pix, psf_pix = get_stamp_coordinates(center, psf.shape[0], psf.shape[1], img.shape)

    # normalized full-image PSF in case you want it later
    #injection_psf = np.zeros(img.shape)
    #cut_psf = psf[psf_pix[0],psf_pix[1]]
    #injection_psf[injection_pix[0], injection_pix[1]] += cut_psf/np.nansum(psf)

    # add the scaled psfs
    injection_img = np.zeros(img_tiled.shape)
    #injection_img[:,injection_pix[0], injection_pix[1]] += (psf_tiled.T*scale_flux).T
    injection_img[:,injection_pix[0], injection_pix[1]] += psf_tiled[:,psf_pix[0], psf_pix[1]]*scale_flux[:,None,None]
    full_injection = injection_img + img_tiled
    if subtract_mean is True:
        full_injection = full_injection - np.nanmean(np.nanmean(full_injection, axis=-1),axis=-1)[:,None,None]
    if return_flat is True:
        shape = full_injection.shape
        if full_injection.ndim == 2:
            full_injection = np.ravel(full_injection)
        else:
            full_injection = np.reshape(full_injection, (shape[0],reduce(lambda x,y: x*y, shape[1:])))
    #if return_psf is True:
    #    return full_injection, injection_psf
    return np.squeeze(full_injection)


def inject_psf(img, psf, center, scale_flux=None, subtract_mean=False, return_flat=False):
    """
    Inject a PSF into an image at a location given by center. Optional: scale PSF
    The PSF is injected by *adding* it to the provided image, not by replacing the pixels
    Input:
        img: 2-D img or 3-D cube. Last two dimensions define an img (i.e. [(Nimg,)Nx,Ny])
        psf: 2-D img or 3-D cube, smaller than or equal to img in size. If cube, 
             must have same 1st dimension as img 
        center: center (row, col) of the injection in the image (can be more than one location)
        scale_flux: multiply the PSF by this number. If this is an array,
             img and psf will be tiled to match its length
             if scale_flux is None, don't scale psf
        subtract_mean: (False) mean-subtract before returning
        return_flat: (False) flatten the array along the pixels axis before returning
        hpf [None]: pass a high-pass filter width for high-pass filtering.
                    If used, the psf must be the *unfiltered* version
    Returns:
       injected_img: 2-D image or 3D cube with the injected PSF(s)
       injection_psf: (if return_psf=True) 2-D normalized PSF full image
    """
    injected_psf=None
    if np.ndim(center) == 1:
        injected_psf  = _inject_psf(img, psf, center, scale_flux, subtract_mean, return_flat)
    elif np.ndim(center) > 1:
        injected_psf = np.sum(np.array([_inject_psf(img, psf, c, scale_flux,
                                                    subtract_mean, return_flat)
                                        for c in center]), axis=0)
    return injected_psf


def inject_region(flat_img, flat_psf, scaling=1, subtract_mean=False):
    """
    Inject a flattened psf into a flattened region with some scaling.
    Input:
        flat_img: 1-d array of the region of interest
        flat_psf: 1-d array of the psf in the region at the correct location
        scaling: (1) multiply the PSF by this number. If this is an array,
                 img and psf will be tiled to match its length.
        subtract_mean: (False) mean-subtract images before returning
    """
    scaling = np.array(scaling)
    # get the right dimensions
    flat_img_tiled = np.tile(flat_img, (np.size(scaling), 1))
    flat_psf_tiled = np.tile(flat_psf, (np.size(scaling), 1))

    # assume the PSF is already properly aligned
    scaled_psf_tiled = (flat_psf_tiled.T*scaling).T

    injected_flat_img = np.squeeze(flat_img_tiled + scaled_psf_tiled)
    if subtract_mean is True:
        injected_flat_img = (injected_flat_img.T - np.nanmean(injected_flat_img, axis=-1)).T
    return injected_flat_img


def mean_subtracted_fake_injections(flat_img, flat_psf, scaling=1):
    """
    flat_img: 2-D image to inject into
    flat_psf: 2-D psf
    """
    scaling = np.array(scaling)
    # get the right dimensions
    flat_img_tiled = np.tile(flat_img, (np.size(scaling), 1))
    flat_psf_tiled = np.tile(flat_psf, (np.size(scaling), 1))

    # assume the PSF is already properly aligned
    scaled_psf_tiled = (flat_psf_tiled.T*scaling).T

    injected_flat_img = np.squeeze(flat_img_tiled + scaled_psf_tiled)
    injected_flat_img = (injected_flat_img.T - np.nanmean(injected_flat_img, axis=-1)).T
