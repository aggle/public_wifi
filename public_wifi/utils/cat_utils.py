"""
Helper functions for catalog matching
"""

import numpy as np
import pandas as pd
from pathlib import Path
from stwcs import wcsutil
from astropy.io import fits

from . import table_utils as tutils

cat_fit_columns = [
    'x_ref', #     Column 1: X (reference)
    'y_ref', #     Column 2: Y (reference)
    'x_input', #     Column 3: X (input)
    'y_input', #     Column 4: Y (input)
    'x_fit', #     Column 5: X (fit)
    'y_fit', #     Column 6: Y (fit)
    'x_resid', #     Column 7: X (residual)
    'y_resid', #     Column 8: Y (residual)
    'x_orig_ref', #     Column 9: Original X (reference)
    'y_orig_ref', #     Column 10: Original Y (reference)
    'x_orig_input', #     Column 11: Original X (input)
    'y_orig_input', #     Column 12: Original Y (input)
    'ref_id', #     Column 13: Ref ID
    'input_id', #     Column 14: Input ID
    'input_extver_id', #     Column 15: Input EXTVER ID 
    'ra', #     Column 16: RA (fit)
    'dec', #     Column 17: Dec (fit)
    'ref_source' #     Column 18: Ref source provenience
]


def catalog2pandas(filename):
    """
    Takes a *catalog_fit.match file and returns it as a pandas dataframe
    Parameters
    ----------
    filename : None
      full path to the *catalog_fit.match file

    Returns
    -------
    df : dataframe with the catalog information
    """
    nheader = 28 # number of lines to skip
    df = pd.read_csv(filename, sep=' *',
                     names=cat_fit_columns,
                     skiprows=nheader,
                     engine='python'
    )
    df['file_id'] = Path(filename).name.split('_')[0]
    return df


def pix2sky(filename, x=None, y=None):
    """
    For the HDUList from an flt file, map the pixel indices to RA and Dec
    I hacked out the piececs of drizzlepac.pix2sky for this.
    Store the results in a table
    THIS NEEDS TESTING TO MAKE SURE ALL THE ARRAY RESHAPING WORKS

    Parameters
    ----------
    filename : pathlib.Path
      The path to and name of a fits file with the appropriate headers
      Assumes the WCS information is stored in the 'sci' extension, EXTNUM=1
    x : number or np.array (Default: None)
      x (column) indices or positions
    y : number or np.array (Default: None)
    Returns
    -------
    radec : np.array [2 x Ny x Nx]
      numpy array of RA and Dec coordinates where the indices correspond to
      the output of np.indices(hdulist[1].data) (i.e. the image shape)
    """
    # If x and y are not provided, do the whole image
    if x is None and y is None:
        data = fits.getdata(filename, 'sci')
        ind = np.indices(data.shape)
        y, x = ind[0].ravel(), ind[1].ravel()
        shape = ind.shape
    else:
        shape = np.array([y, x]).shape
    # handle single numbers
    if np.ndim(x) == 0:
        x = np.array([x])
    if np.ndim(y) == 0:
        y = np.array([y])

    # this handles the WCS transformation
    inwcs = wcsutil.HSTWCS(filename.as_posix()+"[sci,1]")
    # pass in an Nx2 array of the raveled indices for each axis
    # this should also work if x and y are just numbers
    radec = inwcs.all_pix2world(np.array([x, y], dtype=float).T,
                                0) # 0 is the origin in python (vs 1 for matlab/DS9)
    radec = np.reshape(radec.T, shape) 
    return radec


def compile_radec(list_of_file_ids, verbose=False):
    """
    Does not work yet
    For all the files, get the xy-rd maps and combine them in a table
    Then write the table to file

    Parameters
    ----------
    list_of_fileids : list
      a list of file_id values for the images (*not* the actual file names)
    verbose : bool (Default: False)
      if True, print progress updates

    Returns
    -------
    radec_series : pd.Series
      a pandas Series indexed by [file_id, [ra,dec]] that stores the coordinates for every file

    """
    # create a dictionary to hold the data
    index = pd.MultiIndex.from_product([list_of_file_ids, ['ra','dec']],
                                       names=['file_id','coord'])
    radec_df = pd.DataFrame(np.nan,
                            index=index,
                            columns=np.arange(1014**2),#radec.shape[-2]*radec.shape[-1]),
                            dtype=float)
    # loop through the file_ids, updating the series each time
    for i, file_id in enumerate(list_of_file_ids):
        filename = tutils.get_file_from_file_id(file_id)
        radec = pix2sky(filename)
        #radec_series[file_id] = *radec # the * turns it into a tuple for assignment
        radec_df[file_id, 'ra'] = radec[0].ravel()
        radec_df[file_id, 'dec'] = radec[1].ravel()
        if verbose:
            if (i+1)%20 == 0: print(f"{i+1}/{len(list_of_file_ids)} processed")
    if verbose: print("Finished")
    return radec_df


def get_gaia_designation(ref_id, cat_file=shared_utils.align_path / "gaia.cat"):
    """
    For the sources used to align WFC3 with Gaia, get their Gaia designations
    Parameters
    ----------
    ref_id : int (or list-like of ints)
      The value of the ref_id field from the TweakReg output
    cat_file : filename (default {0}/gaia.cat)
      File path to the downloaded Gaia catalog

    Returns
    -------
    gaia_desig: int or list of ints of the Gaia designation

    """.format(shared_utils.align_path.as_posix())
    gaia_cat = pd.read_csv("../data/align_catalog/gaia.cat", sep=' ')
    gaia_ids = gaia_cat.loc[ref_id]['designation']
    return gaia_ids



if __name__ == "__main__":
    pass
