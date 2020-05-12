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
filenames = [ks2path / i for i in ["LOGR.XYVIQ1", "LOGR.FIND_NIMFO", "INPUT.KS2"]]

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

"""
filterid2filter_df = pd.DataFrame(columns=['filter_id','filter_name'])
with open(filenames[2], 'r') as f:
    for line in f:
        # filter ID
        # if you find the filter key, then...
        if line.find("F_SLOT") >= 0:
            # assign the filter id
            filt_id = int(re.search("[0-9]+?", line).group())
        # ... the next instance of the "FILTER" kw is the corresponding filter ID
        if line.find("FILTER") >= 0:
            filt_name = line.strip().split(' ')[-1][1:-1]
            filterid2filter_df = filterid2filter_df.append({'filter_id': filt_id,
                                                            'filter_name': filt_name},
                                                           ignore_index=True)
        # file ID
        if line.find("PIX=") >= 0:
            file_id = 'G'+line.split(' ')[0]
            file_name = Path(re.search('PIX=".*"', line).group().split('"')[1]).name
            new_data = {'file_id': file_id,
                        'file_name': file_name}
            fileid2file_df = fileid2file_df.append(new_data, ignore_index=True)
"""

def get_file_mapper(ks2_input_file):
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


def get_filter_mapper(ks2_input_file):
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
    with open(filenames[2], 'r') as f:
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


if __name__ == "__main__":
    ks2_filemapper = get_file_mapper(filenames[2])
    ks2_filtermapper = get_filter_mapper(filenames[2])
