"""
This file contains utilities for handling the KS2 point source catalog
"""

import numpy as np
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

    ks2_input_file = Path(ks2_input_file).resolve().as_posix()
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
    ks2_input_file = Path(ks2_input_file).resolve().as_posix()
    with open(ks2_input_file, 'r') as f:
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
These helper functions keep you from having to duplicate code every time you want to pull a filter or file name from the above tables
TODO: I think it's possible to build a template function for this so I don't have to make a new one for every table and field
"""
def get_filter_name_from_ks2id(filter_id, filter_mapper=get_filter_mapper()):
    """
    Given a KS2 filter identifier, get the name of the filter

    Parameters
    ----------
    filter_id : str
      The identifier assigned to each filter by KS2. Typically F followed by a number, like F1 or F2.
    filter_mapper : pd.DataFrame (default: result of ks2_utils.get_filter_mapper())
      The dataframe that tracks the filter IDs and the corresponding filter names

    Returns
    -------
    filter_name : the HST name of the filter
    """
    filter_id = filter_id.upper()
    filter_name = filter_mapper.query(f"filter_id == '{filter_id}'")['filter_name'].values[0]
    return filter_name

def get_file_name_from_ks2id(file_id, file_mapper=get_file_mapper()):
    """
    Given a KS2 file identifier, get the name of the file

    Parameters
    ----------
    file_id : str
      The identifier assigned to each file by KS2. Typically F followed by a number, like F1 or F2.
    file_mapper : pd.DataFrame (default: result of ks2_utils.get_file_mapper())
      The dataframe that tracks the file IDs and the corresponding file names

    Returns
    -------
    file_name : the HST identifier for the file
    """
    file_id = file_id.upper()
    file_name = file_mapper.query(f"file_id == '{file_id}'")['file_name'].values[0]
    return file_name


"""
The following code block parses the master info for each *star* - that is, each astrophysical object.
Reminder: LOGR.XYVIQ1 gives the average position for each source on the master frame (cols 1 and 2), the average flux (cols 5 and 11), the flux sigma (cols 6 and 12), and fit quality (cols 7 and 13) in each filter)
It also renames the columns of interest with more intuitive names.
"""
# this dict is used to change the column names into something more readable
# the zmast, szmast, and q parameters are followed by an integer indicating the filter
# for now, we are just going to use the same column names as in the KS2 output.
# Their definitions can be found in docs/database/ks2_output_definitions.org
column_name_mapper = { # no longer used
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
        astro_id : ID for the astrophysical object
        umast0, vmast0 :  average x and y position [pix]
        zmast1, szmast1 : filter 1 counts and count sigma,
        q1, o1 : filter 1 fit quality and crowding
        f1, g1 : num exposures and good measurements where star was found in filter 1
        zmast, szmast, q, o, f, and g columns with higher indices indicate different filters
    """
    # get the columns
    ks2_master_file = Path(ks2_master_file).resolve().as_posix()
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
    """
    # remove this block for now
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
    """
    return master_catalog_df


"""
Finally, this code block parses LOGR.FIND_NIMFO. Notes from ES: 
LOGR.FIND_NIMFO gives you the coordinates and fluxes of each star in each exposure. Cols 14 and 15 contain the x and y coordinates in the flt images (i.e. *before* geometric distortion correction). col 36 is the ID number for each star (starts with R). col 39 is the ID for the image (starts with G). col 40 (starts with F) is the ID for the filter.
n.b. the column numbers start from 1
So, the column names in the file don't match up with the actual number of columns. I'm just going to have to trust Elena on this one and go by her column numbers.
Update: After a response from Jay Anderson, the developer of KS2, the full set of column names and descriptions are available in ../docs/database/ks2_output_definitions.org
"""
# column names and their descriptions
nimfo_cols = {
    0: 'umast',           # center along x-axis in pixels in the master frame
    1: 'vmast',           # center along y-axis in pixels in the master frame
    2: 'magu',            # probably -2.5*log(z1)
    3: 'utile',           # x-location within the tile where the star was found
    4: 'vtile',           # y-location within the tile where the star was found
    5: 'z0',              # method-0 counts (quick and dirty measurement)
    6: 'sz0',             # z0 error
    7: 'f0',              # was the star found in this exposure?
    8: 'g0',              # was it measured to be consistent with others, or rejected?
    9: 'u1',              # local tile measurement for this star in this exposure
    10: 'v1',             # "
    11: 'x1',             # local detector measurement for this star in this exposure (method 1)
    12: 'y1',             # "
    13: 'xraw1',          # x-position in pixels in the exposure identified in Col 38
    14: 'yraw1',          # y-position in pixels in the exposure identified in Col 38
    15: 'z1',             # method-1 counts for this exposure
    16: 'sz1',            # count stderr in "
    17: 'q1',             # PSF fit quality for above
    18: 'o1',             # Crowding parameter for above
    19: 'f1',             # see f0, but for method-1
    20: 'g1',             # see g0, but for method-1
    21: 'x0',             # expected position of the star in this image, from method-0
    22: 'y0',             # "
    23: 'z2',             # same as above, but for method-2
    24: 'sz2',            # "
    25: 'q2',             # "
    26: 'o2',             # "
    27: 'f2',             # "
    28: 'g2',             # "
    29: 'z3',             # method-3 info
    30: 'sz3',            # "
    31: 'q3',             # "
    32: 'o3',             # "
    33: 'f3' ,            # "
    34: 'g3',             # "
    35: 'NMAST',          # Astrophysical object ID number, name comes from master table
    36: 'ps_tile_id',     # point source number within the tile
    37: 'tile_id',        # local tile exposure number for this measurement
    38: 'master_exp_id',  # Master exposure number / file and header ID number
    39: 'filt_id',        # filter number
    40: 'unk',            # chip number (within the master exposure)
}
# this stores the data types for the FIND_NIMFO columns
nimfo_dtypes = {
    "umast": np.float64,
    "vmast": np.float64,
    "magu": np.float64,
    "utile": np.float64,
    "vtile": np.float64,
    "z0": np.float64,
    "sz0": np.float64,
    "f0": np.int64,
    "g0": np.int64,
    "u1": np.float64,
    "v1": np.float64,
    "x1": np.float64,
    "y1": np.float64,
    "xraw1": np.float64,
    "yraw1": np.float64,
    "z1": np.float64,
    "sz1": np.float64,
    "q1": np.float64,
    "o1": np.float64,
    "f1": np.int64,
    "g1": str,
    "x0": np.float64,
    "y0": np.float64,
    "z2": np.float64,
    "sz2": np.float64,
    "q2": np.float64,
    "o2": np.float64,
    "f2": np.int64,
    "g2": str,
    "z3": np.float64,
    "sz3": np.float64,
    "q3": np.float64,
    "o3": np.float64,
    "f3": np.int64,
    "g3": str,
    "NMAST": str,
    "ps_tile_id": str,
    "tile_id": str,
    "master_exp_id": str,
    "filt_id": str,
    "unk": str,
    "chip_id": np.int64,
}

# this array is useful for looping over photometry methods
# you can concatenate it with z, sz, q, o, f, and g
phot_method_ids = ['1', '2', '3']

def get_point_source_catalog(ps_file=ks2_files[1], clean=False):
    """
    This function reads the KS2 FIND_NIMFO file that stores *every* point source

    Parameters
    ----------
    ps_file : pathlib.Path or string
      full path to the LOGR.FIND_NIMFO
    clean : bool [False]
      if True, apply some preliminary cleaning to the catalog,
      like converting special characters to bool or NaN

    Returns
    -------
    point_sources_df : pd.DataFrame
      catalog of all point sources detected and associated information.
      For documentation on the column names, see docs/database/ks2_output_definitions.org
    """
    ps_file = Path(ps_file).resolve().as_posix()

    point_sources_df = pd.read_csv(ps_file,
                                   sep=' ',
                                   skipinitialspace=True, 
                                   index_col=False, 
                                   skiprows=5, 
                                   header=None,
                                   usecols=nimfo_cols.keys(),
                                   dtype=nimfo_dtypes,
    )
    point_sources_df.rename(columns=nimfo_cols, inplace=True)
    # split the file identifier into the file number and extension number
    point_sources_df['chip_id'] = point_sources_df['master_exp_id'].apply(lambda x: int(x.split('.')[1]))
    point_sources_df['master_exp_id'] = point_sources_df['master_exp_id'].apply(lambda x: x.split('.')[0])

    # should we clean it up?
    if clean == True:
        point_sources_df = clean_point_source_catalog(point_sources_df)
    return point_sources_df


def clean_point_source_catalog(cat):
    """
    This function, when called, cleans the point source catalog by applying the listed series of cuts:
    1) q > 0
    2) z > 0
    """

    # don't keep any point sources that have any q = 0
    q_gt_0 = ' and '.join([f"q{method_id} > 0" for method_id in phot_method_ids])
    z_gt_0 = ' and '.join([f"z{method_id} > 0" for method_id in phot_method_ids])
    #q_gt_95 = " and ".join([f"q{method_id} >= 0.95" for method_id in phot_method_ids])
    # convert the g* columns to int
    # g is 1 if the stamp is consistent with other stamps for this star, else 0
    # leave as int for easy summation
    #for col in cat.columns:
    #    if re.search('^g[0-9]+', col) is not None:
    #        g2bool = lambda x: True if (str(x) == '1') else False
    #        cat[col] = cat[col].apply(g2bool)
    #print(cut_q, cut_z)
    full_query = q_gt_0 + ' and ' + z_gt_0 #+ ' and ' + q_gt_95
    cut_df = cat.query(full_query)
    return cut_df


def generate_exposure_distance_matrix(df, same_nan=True):
    """
    Given a KS2 dataframe, compute a symmetric matrix of pixelwise distances.
    The dataframe should only contain point sources from the same exposure.

    Parameters
    ----------
    df : pd.DataFrame
      dataframe that at least contains the columns [NMAST, xraw1, and yraw1]
    same_nan : bool (False)
      if True, set the diagonal to nan instead of 0

    Returns
    -------
    dist_mat : pd.DataFrame
      N sources x N sources dataframe, where the index and columns are
      the NMAST identifiers, containing the pixelwise distance between sources

    """
    dist_mat = pd.DataFrame(index=df['NMAST'], columns=df['NMAST'], dtype=np.float)
    for i, row in df.iterrows():
        star = row['NMAST']
        diff = df[['xraw1','yraw1']] - row[['xraw1','yraw1']].values
        dist_mat[star] = np.linalg.norm(diff, axis=1)

    # if desired, set the diagonal to nan instead of 0
    if same_nan == True:
        for col in dist_mat.columns:
            dist_mat.loc[col, col] = np.nan

    return dist_mat


if __name__ == "__main__":
    # run it in script mode to get all the dataframes
    ks2_filemapper = get_file_mapper(ks2_files[2])
    ks2_filtermapper = get_filter_mapper(ks2_files[2])
    ks2_mastercat = get_master_catalog(ks2files[0])
    ks2_allsources = get_point_source_catalog(ks2files[1])
