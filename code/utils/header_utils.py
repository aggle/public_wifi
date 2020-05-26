"""
Simple module to write the fits headers to file, and read them, or do other manipulations
For now, the filenames and directories will be hard-coded in
"""

from pathlib import Path
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
    Put them all into dictionaries
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
    Convert a list of header dictionaries into a pandas dataframe

    Parameters
    ----------
      header_dicts: a list of header dicts

    Returns
    -------
      header_df: a single pandas dataframe containing information from all the headers
    """
    numbered_dict = dict(zip(range(len(header_dicts)), header_dicts))
    header_df = pd.DataFrame(numbered_dict).T
    header_df.columns = [i.lower() for i in header_df.columns]
    return header_df


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
    extname : pri
      shorthand name for the extension whose dataframe you want

    Returns
    -------
    df : a dataframe with the right header selected
    """
    filepath = header_path / f'{extname.lower()}_hdrs.csv'
    read_args = {'index_col':0, 'header':2}
    # special case
    if extname == 'all':
        # return all the headers
        dfs = {e.lower(): pd.read_csv(filepath, **read_args)
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


if __name__ == "__main__":
    """
    Run as a stand-alone script to regenerate the header dataframes
    """
    datapath = Path("../../data/sabbi_data/")
    files = datapath.glob("i*flt.fits")
    write_headers(files, verbose=True)
