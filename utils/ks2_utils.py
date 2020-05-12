"""
This file contains utilities for handling the KS2 point source catalog
"""

import pandas as pd
from pathlib import Path
import re


# trying some literate programming
"""
You only care about a few of the files:
INPUT.KS2 contains the instructions used to rnun the photometry, and it shows which flt files correspond to the file numbers in the LOGR files
LOGR.XYVIQ1 gives the average position for each source on the master frame (cols 1 and 2), the average flux (cols 5 and 11), the flux sigma (cols 6 and 12), and fit quality (cols 7 and 13) in each filter)
LOGR.FIND_NIMFO gives you the coordinates and fluxes of each star in each exposure. Cols 14 and 15 contain the x and y coordinates in the flt images (i.e. *before* geometric distortion correction). col 36 is the ID number for each star (starts with R). col 39 is the ID for the image (starts with G). col 40 (starts with F) is the ID for the filter.
"""
ks2path = Path("../data/ks2/")
ks2_files = [ks2path / i for i in ["LOGR.XYVIQ1", "LOGR.FIND_NIMFO", "INPUT.KS2"]]


"""
The following section parses INPUT.KS2 (really, a cutout of INPUT.KS2)
It creates maps for:
- file ID  -> file name
- filter ID -> filter name
These only apply to this particular KS2 output.
At the end, you have two dataframes:
- fileid2file_df (file ID -> file name)
- filterid2filter_df (filter ID -> filter name)
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
                file_id = 'G'+line.split(' ')[0] + '.1' # .1 for the hdu index, I believe
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
The following section parses the master info for each *star* - that is, each astrophysical object.
Reminder: LOGR.XYVIQ1 gives the average position for each source on the master frame (cols 1 and 2), the average flux (cols 5 and 11), the flux sigma (cols 6 and 12), and fit quality (cols 7 and 13) in each filter)
I think the output is a little buggy because all the filters are numbered 1 when the number should be incrementing. I have fixed this
"""
def get_master_catalog(ks2_master_file=ks2_files[0]):
    """
    From LOGR.XYVIQ1, pull out the master catalog of information about astrophysical objects
    Parameters
    ----------
    ks2_master_file : str or pathlib.Path
     full path to the file
    Returns
    -------
    master_catalog : pd.DataFrame
      pandas dataframe with the following columns: [astro_obj_id, # ID for the astrophysical object
                                                    x_master, y_master, # average x and y position [pix]
                                                    counts_filt1, e_counts_filt1, quality_filt1 # counts, count sigma, and fit quality in filter 1
                                                    counts_filt2, e_counts_filt2, quality_filt2] # counts, count sigma, and fit quality in filter 2
    """
    # get the columns
    with open(ks2_master_file) as f:
        columns = f.readlines(2)[1] # get line #2
        columns = re.findall('[a-z0-9]+', columns)
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

    master_catalog = pd.DataFrame(ks2_master_file,
                                  names=columns,
                                  sep=' ',
                                  skiprows=[0, 1, 2, 3],
                                  engine='python',
                                  skipinitialspace=True,
                                  index_col=False)
    # ok, now pick the columns you care about and rename them
    # this dictionary maps between the KS2 column names and mine
    column_mapper = {'umast0': 'x_master',
                     'vmast0': 'y_master',
                     'zmast1': 'astro_obj_id',
                     'q1': 'quality_f1'}


if __name__ == "__main__":
    ks2_filemapper = get_file_mapper(ks2_files[2])
    ks2_filtermapper = get_filter_mapper(ks2_files[2])
