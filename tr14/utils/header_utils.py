"""
Simple module to write the fits headers to file, and read them, or do other manipulations
For now, the filenames and directories will be hard-coded in
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.io import fits

from . import shared_utils as sutils
from . import table_utils as tutils

extnames = ['SCI','ERR','DQ','SAMP','TIME']
all_headers = ['PRI'] + extnames
header_path = sutils.table_path


def clean_header_dict(hdr_dict):
    """
    Remove unnecessary keywords from a header dictionary
    Keywords to remove: "''", "HISTORY","COMMENT"
    Returns nothing; changes made in-place
    """
    bad_kwds = ['','HISTORY','COMMENT']
    for k in bad_kwds:
        try:
            hdr_dict.pop(k)
        except KeyError:
            # keyword not found; this is fine
            pass



def header2dict(header):
    """
    Convert header to a dict and do cleanup
    Accepts a FITS header
    Returns a dict of the header keywords and values, plus whatever cleanup 
    """
    hdr = dict(header)
    clean_header_dict(hdr)
    return hdr


def headers2dict(filenames):
    """
    Each file has the same keywords in the headers
    Put them *all* into dictionaries (loop header2dict over all the headers in all the files)
    """
    extdict = {k: [] for k in [i.lower() for i in all_headers]}
    for ff in filenames:
        hdulist = fits.open(ff)
        # primary header
        extdict['pri'].append(header2dict(hdulist[0].header))
        # extension headers
        for hdu in hdulist[1:]:
            name = hdu.header['EXTNAME']
            if name.lower() in extdict.keys():
                extdict[name.lower()].append(header2dict(hdu.header))
        hdulist.close()
    return extdict


def headerdict2dataframe(header_dicts):
    """
    Convert a header dictionary into a pandas dataframe

    Parameters
    ----------
      header_dict: the dictionary made from a fits header

    Returns
    -------
      header_df: a single pandas dataframe containing information from all the headers
    """
    numbered_dict = dict(zip(range(len(header_dicts)), header_dicts))
    header_df = pd.DataFrame(numbered_dict).T
    header_df.columns = [i.lower() for i in header_df.columns]
    return header_df


def get_header_dfs(fits_files):
    """
    From the given fits file names, group all the headers into dataframes, with
    dataframe for each type of header and one row for each file

    Parameters
    ----------
    fits_files : list-like
      a list of filenames for FITS images you want to collect together

    Output
    ------
    df_dict : dict
      a dictionary of the dataframes. the dict key is the name of the header type
      (PRI, SCI, ERR, etc)

    """
    header_dicts = headers2dict(fits_files)
    header_dfs = {k: headerdict2dataframe(header_dicts[k]) for k in header_dicts}
    return header_dfs 


def write_headers(filenames, verbose=False):
    """
    Save the headers, as csv files that can be read into dataframes, in the folder designated at the top of the file ({0}). The file name is the extension header EXTNAME value, and 'pri' refers to the primary header. Does all the work to compile and write headers. No return value; writes a file instead

    Parameters
    ----------
    filename : None
     a list of filenames for FITS images you want to treat as a single dataset
    verbose : False
     if True, print the name of the file written

    Returns
    -------
    Nothing
    """.format(header_path)

    header_dicts = headers2dict(filenames)
    header_dfs = {k: headerdict2dataframe(header_dicts[k]) for k in header_dicts}
    for k, v in header_dfs.items():
        out_name = f'{k}_hdrs'  # header_path / f'{k}_hdrs.csv'
        if verbose == True:
            print(f'Writing {tutils.table_path / (out_name + ".csv")}')
        tutils.write_table(v,
                           out_name,
                           f"Compilation of {k.upper()} headers from HST image files")
        #v.to_csv(out_name)
    if verbose == True:
        print("\nFinished.")


def load_headers(extname='pri'):
    """
    Helper function to load the right header file

    Parameters
    ----------
    extname : str [pri]
      shorthand name for the extension whose dataframe you want
      options are: pri, sci, err, dq, samp, and time
    Returns
    -------
    df : pd.DataFrame
      a dataframe with the right header selected
    """
    filepath = header_path / f'{extname.lower()}_hdrs.csv'
    read_args = {'index_col':0, 'header':2}
    # special case
    if extname == 'all':
        # return all the headers
        print("Returning all headers.")
        dfs = {e.lower(): pd.read_csv(header_path / f'{e.lower()}_hdrs.csv',
                                      **read_args)
               for e in all_headers}
        return dfs
    # otherwise, check that the input is OK
    try:
        assert(extname.upper() in all_headers)
    except AssertionError:
        print(f"{extname} not one of {all_headers}, please try again.")
        return None
    df = pd.read_csv(filepath, **read_args)
    return df


def print_columns(df):
    """
    Helper function to print the long list of columns in a header (in alphabetical order)

    Parameters
    ----------
    df : None
      dataframe whose columns you want to print

    Returns
    -------
    Nothing
    """
    columns = sorted(df.columns)
    for i in columns:
        print(i)

########################
# Data Quality parsers #
########################

# compile this regex once
dq_find1 = re.compile('1')
def parse_dq_binword(pix_val):
    """
    This function is designed to be applied to a pandas Series. It returns the
    flags found at each pixel
    See https://hst-docs.stsci.edu/wfc3dhb/chapter-2-wfc3-data-structure/2-2-wfc3-file-structure

    Parameters
    ----------
    dq_array : np.array of type int16

    Output
    ------
    pix_flags: list of ints, or 0
      A list of all the non-zero flags present for a pixel (e.g. [1, 16, 32, 256])
      If pixel value is 0, return 0
    """
    # first, quick check to see if you have to do any work
    if pix_val == 0:
        return 0
    pix_bin = bin(pix_val)[2:] # convert to bin string and drop the 0b prefix
    #pix_flags = [2**i.span()[0] for i in dq_find1.finditer(pix_bin[::-1])]
    pix_flags = [2**i for i, j in enumerate(list(pix_bin[::-1])) if j == '1']
    return pix_flags


def parse_dq_array(dq_array):
    """
    This should return a dictionary, where each key is a flag value, and each
    value is the y,x coordinates of the pixels that have that value

    Parameters
    ----------
    dq_array : np.array
      array of DQ binary words

    Output
    ------
    dq_flag_dict : dict or None
      a dictionary whose keys are the flags and whose entries are the 2xN
      row, col pixel coordinates of pixels that have those flags
      If there are no error flags, return None
    """
    # first, check if there are no errors
    # convert the stamp to a series. the index holds the raveled pixel order
    dq_series = pd.Series(dq_array.ravel())
    # but you only care about the non-zero pixels
    dq_series = dq_series[dq_series != 0]

    # If there are no error flags, return None
    if dq_series.empty == True:
        return None
    # otherwise, get the flags for each remaining pixel
    dq_pix_flags = dq_series.apply(parse_dq_binword)

    # convert this to a dataframe, and then a dictionary for each flag
    dq_columns = [2**i for i in range(15)]
    # set up a dataframe to store the value for each flag
    dq_flags_full = pd.DataFrame(False,
                                 index=dq_pix_flags.index,
                                 columns=dq_columns)
    # set found flags to True
    for i in dq_pix_flags.index:
        dq_flags_full.loc[i, dq_pix_flags[i]] = True

    # cut the df down to only the flags that actually appear
    dq_flags = dq_flags_full.loc[:, dq_flags_full.any(axis=0)]
    # finally, use boolean indexing to get the valid flags in each pixel
    dq_flag_dict = {col: dq_flags[dq_flags[col]][col].index.values 
                    for col in dq_flags.columns}
    # finally, unravel the coordinates
    for k, v in dq_flag_dict.items():
        dq_flag_dict[k] = np.vstack(np.unravel_index(v, dq_array.shape)).T
    return dq_flag_dict


if __name__ == "__main__":
    """
    Run as a stand-alone script to regenerate the header dataframes
    """
    datapath = Path("../../data/sabbi_data/")
    files = datapath.glob("i*flt.fits")
    write_headers(files, verbose=True)
