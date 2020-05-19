"""
This file contains utilities for handling the KS2 point source catalog
"""

import pandas as pd
from pathlib import Path
import re
from . import shared_utils


"""
You only care about a few of the files:
INPUT.KS2 contains the instructions used to rnun the photometry, and it shows which flt files correspond to the file numbers in the LOGR files
LOGR.XYVIQ1 gives the average position for each source on the master frame (cols 1 and 2), the average flux (cols 5 and 11), the flux sigma (cols 6 and 12), and fit quality (cols 7 and 13) in each filter)
LOGR.FIND_NIMFO gives you the coordinates and fluxes of each star in each exposure. Cols 14 and 15 contain the x and y coordinates in the flt images (i.e. *before* geometric distortion correction). col 36 is the ID number for each star (starts with R). col 39 is the ID for the image (starts with G). col 40 (starts with F) is the ID for the filter.
"""
ks2_files = [shared_utils.ks2_path / i for i in ["LOGR.XYVIQ1", "LOGR.FIND_NIMFO", "INPUT.KS2"]]


"""
The following block parses INPUT.KS2
It creates maps for:
- file ID  -> file name
- filter ID -> filter name
These only apply to this particular KS2 output.
At the end, you have two dataframes:
- filemapper_df (file ID -> file name)
- filtermapper_df (filter ID -> filter name)
"""
def get_file_mapper(ks2_input_file=ks2_files[2]):
    """
    Given a KS2 INPUT.KS2 file, parse the file to get a dataframe that maps between the file numbers and file names.

    Parameters
    ----------
    ks2_input_file : pathlib.Path or str
      full path to KS2 file

    Returns
    -------
    filemapper_df : pd.DataFrame
      pandas dataframe with two columns: file_id, and file_name
    """
    filemapper_df = pd.DataFrame(columns=['file_id','file_name'])
    
    with open(ks2_input_file, 'r') as f:
        for line in f:
            # file ID
            if line.find("PIX=") >= 0:
                file_id = 'G'+line.split(' ')[0]
                file_name = Path(re.search('PIX=".*"', line).group().split('"')[1]).name
                new_data = {'file_id': file_id,
                            'file_name': file_name}
                filemapper_df = filemapper_df.append(new_data, ignore_index=True)
    return filemapper_df


def get_filter_mapper(ks2_input_file=ks2_files[2]):
    """
    Given a KS2 INPUT.KS2 file, parse the file to get a dataframe that maps between the filter numbers and filter names.

    Parameters
    ----------
    ks2_input_file : pathlib.Path or str
      full path to KS2 file

    Returns
    -------
    filtermapper_df : pd.DataFrame
      pandas dataframe with two columns: filter_id, and filter_name
    """
    with open(ks2_files[2], 'r') as f:
        input = f.read() # read in the entire file
        # regex magic
        keys = '|'.join(['F_SLOT *= *[0-9]+?','FILTER * = *"[A-Z0-9]*"'])
        results = re.findall(keys, input)
        # do a little parsing and rearranging
        results = [i.split(' ')[-1] for i in results]
        results = list(zip(results[::2], results[1::2]))
    filtermapper_df = pd.DataFrame(results, columns=['filter_id','filter_name'])
    filtermapper_df['filter_id'] = filtermapper_df['filter_id'].apply(lambda x: "F"+x)
    filtermapper_df['filter_name'] = filtermapper_df['filter_name'].apply(lambda x: x.replace('"',''))
    return filtermapper_df


"""
The following code block parses the master info for each *star* - that is, each astrophysical object.
Reminder: LOGR.XYVIQ1 gives the average position for each source on the master frame (cols 1 and 2), the average flux (cols 5 and 11), the flux sigma (cols 6 and 12), and fit quality (cols 7 and 13) in each filter)
It also renames the columns of interest with more intuitive names.
"""
# this dict is used to change the column names into something more readable
# the zmast, szmast, and q parameters are followed by an integer indicating the filter
column_name_mapper = {
    'umast0': 'master_x',           # x position [pix]
    'vmast0': 'master_y',           # y position [pix]
    'NMAST': 'astro_obj_id',        # object ID number
    'zmast': 'master_counts_f',     # mean photometry [counts/sec]
    'szmast': 'master_e_counts_f',  # sigma photometry [counts/sec]
    'q': 'master_quality_f',        # PSF fit quality
    'o': 'master_crowding_f',       # crowding parameter
}

def get_master_catalog(ks2_master_file=ks2_files[0]):
    """
    From LOGR.XYVIQ1, pull out the master catalog of information about astrophysical objects

    Parameters
    ----------
    ks2_master_file : str or pathlib.Path
     full path to the file

    Returns
    -------
    master_catalog_df : pd.DataFrame
      pandas dataframe with the following columns:
        astro_obj_id, # ID for the astrophysical object
        master_x, master_y, # average x and y position [pix]
        master_counts_f1, e_master_counts_f1, # filter 1 counts and count sigma,
        master_quality_f1, master_crowding_f1, # filter 1 fit quality and crowding
        master_counts_f2, e_master_counts_f2, # filter 2 counts and count sigma,
        master_quality_f2, master_crowding_f2 # filter 2 fit quality and crowding
    """
    # get the columns
    with open(ks2_master_file) as f:
        columns = f.readlines(2)[1] # get line #2
        columns = re.findall('[A-Za-z0-9]+', columns)
    # next step, need to fix the filter indexes!
    # find duplicate entries and increment the suffix by 1 for the duplicates
    col_unique = list(set(columns))
    col_ind = {c:[] for c in col_unique}
    for i in col_unique:
        for j, k in enumerate(columns):
            if i == k:
                col_ind[i].append(j)
    for k,v in col_ind.items():
        if len(v) > 1:
            for i in range(1, len(v)):
                columns[v[i]] = columns[v[i]][:-1] + str(i+1)
    # ok, now you are ready to read in the file
    master_catalog_df = pd.read_csv(ks2_master_file,
                                    names=columns,
                                    sep=' ',
                                    skiprows=[0, 1, 2, 3],
                                    engine='python',
                                    skipinitialspace=True,
                                    index_col=False)
    # ok, now pick the columns you care about and rename them
    # this dictionary maps between the KS2 column names and mine,
    # and accounts for the filter ID appended to the end of some
    # columns
    new_columns = {}
    for i in master_catalog_df.columns:
        for j in column_name_mapper.keys():
            if i.find(j) >= 0:
                new_columns[i] = i.replace(j, column_name_mapper[j])
    master_catalog_df.rename(columns=new_columns, inplace=True)
    # now, remove all the columns that you don't want to keep
    for col in master_catalog_df.columns:
        if col not in new_columns.values():
            master_catalog_df.drop(col, axis=1, inplace=True)

    return master_catalog_df


"""
Finally, this code block parses LOGR.FIND_NIMFO. Notes from ES: 
LOGR.FIND_NIMFO gives you the coordinates and fluxes of each star in each exposure. Cols 14 and 15 contain the x and y coordinates in the flt images (i.e. *before* geometric distortion correction). col 36 is the ID number for each star (starts with R). col 39 is the ID for the image (starts with G). col 40 (starts with F) is the ID for the filter.
n.b. the column numbers start from 1
So, the column names in the file don't match up with the actual number of columns. I'm just going to have to trust Elena on this one and go by her column numbers.
"""
def get_point_source_catalog(ps_file=ks2_files[1]):
    """

    Parameters
    ----------

    Returns
    -------
    point_sources_df : pd.DataFrame
      catalog of point sources with the following columns:
        flt_x, flt_y : position in the FLT images, before geometric distortion correction
        astro_obj_id : identifier for the astronomical object associated with the point source
        file_id : identifier for the file that stores the image the point source comes from
        filter_id : identifier for the filter
        file_ext_id : number that identifies the HDU in the file HDUList
    """
    # use only these columns
    col_names = {
        0: 'x_master',      # x-position in the master frame
        1: 'y_master',      # y-position in the master frame
        13: 'x_flt',        # x-position in the FLT
        14: 'y_flt',        # y-position in the FLT
        15: 'flux_f1',      # counts in filt 1?
        16: 'e_flux_f1',    # counts in filt 2?
        17: 'psf_fit_f1',   # psf fit quality in filt 1
        18: 'crowding_f1',  # crowding in filt 1
        23: 'flux_f2',      # counts in filt 2?
        24: 'e_flux_f2',    # counts in filt 2?
        25: 'psf_fit_f2',   # psf fit quality in filt 2
        26: 'crowding_f2',  # crowding in filt 2
        35: 'astro_obj_id', # astrophysical object ID number
        38: 'file_id',      # file ID number
        39: 'filter_id'     # filter ID number
    }
    point_sources_df = pd.read_csv(ks2_files[1],
                                   sep=' ',
                                   skipinitialspace=True, 
                                   index_col=False, 
                                   skiprows=5, 
                                   header=None,
                                   usecols=col_names.keys())
    point_sources_df.rename(columns=col_names, inplace=True)
    # split the file identifier into the file number and extension number
    point_sources_df['file_ext_id'] = point_sources_df['file_id'].apply(lambda x: int(x.split('.')[1]))
    point_sources_df['file_id'] = point_sources_df['file_id'].apply(lambda x: x.split('.')[0])
    return point_sources_df



if __name__ == "__main__":
    # run it in script mode to get all the dataframes
    ks2_filemapper = get_file_mapper(ks2_files[2])
    ks2_filtermapper = get_filter_mapper(ks2_files[2])
    ks2_mastercat = get_master_catalog(ks2files[0])
    ks2_allsources = get_point_source_catalog(ks2files[1])
