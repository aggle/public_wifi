"""
Helper functions for catalog matching
"""

import pandas as pd
from pathlib import Path
from stwcs import wcsutil

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


def pix2sky(filename):
    """
    For the HDUList from an flt file, map the pixel indices to RA and Dec
    I hacked out the piececs of drizzlepac.pix2sky for this.
    Store the results in a table
    THIS NEEDS TESTING TO MAKE SURE ALL THE ARRAY RESHAPING WORKS

    Parameters
    ----------
    filename : str
      The path to and name of a fits file with the appropriate headers
      Assumes the WCS information is stored in the 'sci' extension, EXTNUM=1

    Returns
    -------
    radec : np.array [2 x Ny x Nx]
      numpy array of RA and Dec coordinates where the indices correspond to
      the output of np.indices(hdulist[1].data) (i.e. the image shape)
    """
    # get the image - you need the shape and indices
    data = fits.getdata(filename, 'sci')
    ind = np.indices(data.shape)
    # this handles the WCS transformation
    inwcs = wcsutil.HSTWCS(filename.as_posix()+"[sci,1]")
    # pass in an Nx2 array of the raveled indices for each axis
    # remember that x is row (1) and y is col (0)
    radec = inwcs.all_pix2world(np.array([ind[1].ravel(), ind[0].ravel()]).T,
                                0) # 0 is the origin in python (vs 1 for matlab/DS9)
    radec = np.reshape(radec.T, ind.shape) 
    return radec


def compile_radec(list_of_files):
    """
    For all the files, get the xy-rd maps and combine them in a table
    Then write the table to file
    """
    pass


if __name__ == "__main__":
    pass
