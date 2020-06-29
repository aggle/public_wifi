"""
This file contains utilities for handling the KS2 point source catalog
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from configparser import ConfigParser

from astropy.io import fits

from . import shared_utils, image_utils, table_utils

"""
You only care about a few of the files:
INPUT.KS2 contains the instructions used to rnun the photometry, and it shows which flt files correspond to the file numbers in the LOGR files
LOGR.XYVIQ1 gives the average position for each source on the master frame (cols 1 and 2), the average flux (cols 5 and 11), the flux sigma (cols 6 and 12), and fit quality (cols 7 and 13) in each filter)
LOGR.FIND_NIMFO gives you the coordinates and fluxes of each star in each exposure. Cols 14 and 15 contain the x and y coordinates in the flt images (i.e. *before* geometric distortion correction). col 36 is the ID number for each star (starts with R). col 39 is the ID for the image (starts with G). col 40 (starts with F) is the ID for the filter.
"""
ks2_files = [shared_utils.ks2_path / i for i in ["LOGR.XYVIQ1",
                                                 "LOGR.FIND_NIMFO",
                                                 "INPUT.KS2"]]


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
                file_name = Path(re.search('PIX=".*"',\
                                           line).group().split('"')[1]).name
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

ks2_filemapper = get_file_mapper()
ks2_filtermapper = get_filter_mapper()

"""
These helper functions keep you from having to duplicate code every time you want to pull a filter or file name from the above tables
TODO: I think it's possible to build a template function for this so I don't have to make a new one for every table and field
"""
def get_filter_name_from_ks2id(filter_id, filter_mapper=get_filter_mapper()):
    """
    Given a KS2 filter identifier, get the name of the HST filter

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
    Given a KS2 file identifier, get the name of the FLT file

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

master_dtypes = {
    "umast0": np.float,
    "vmast0": np.float,
    "mmast1": np.float,
    "NMAST": object,
    "zmast1": np.float,
    "szmast1": np.float,
    "q1": np.float,
    "o1": np.float,
    "f1": np.int,
    "g1": np.int,
    "zmast2": np.float,
    "szmast2": np.float,
    "q2": np.float,
    "o2": np.float,
    "f2": np.int,
    "g2": np.int,
}
def get_master_catalog(ks2_master_file=ks2_files[0], clean=True):
    """
    From LOGR.XYVIQ1, pull out the master catalog of information about astrophysical objects

    Parameters
    ----------
    ks2_master_file : str or pathlib.Path
     full path to the file
    clean : bool (True)
      if True, returns the cleaned version of the catalog (i.e. with cuts)
    Returns
    -------
    master_catalog_df : pd.DataFrame
      pandas dataframe with the following columns:
        NMAST : ID for the astrophysical object
        umast0, vmast0 :  average x and y position [pix]
        zmast1, szmast1 : filter 1 counts and count sigma,
        q1, o1 : filter 1 fit quality and crowding
        f1, g1 : num exposures and good measurements where star was found in filter 1
        zmast, szmast, q, o, f, and g columns with higher indices indicate different filters
    """
    # get the columns
    ks2_master_file = Path(ks2_master_file).resolve().as_posix()

    # otherwise, parse the original KS2 output file
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
    # there may be some typing issues. make sure the floats are all float or nan
    float_cols = ['umast0', 'vmast0',
                  'mmast1', 'zmast1', 'szmast1', 'q1',
                  'mmast2', 'zmast2', 'szmast2', 'q2']
    def col2float(x):
        try:
            return np.float(x)
        except ValueError:
            return np.nan
    for col in float_cols:
        mast_cat[col] = mast_cat[col].apply(col2float)
    """

    # if desired, return the cleaned catalog. else, return raw
    if clean == True:
        master_catalog_df = clean_master_catalog(master_catalog_df)

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
    38: 'exp_id',  # Master exposure number / file and header ID number
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
    "g1": object,
    "x0": np.float64,
    "y0": np.float64,
    "z2": np.float64,
    "sz2": np.float64,
    "q2": np.float64,
    "o2": np.float64,
    "f2": np.int64,
    "g2": object,
    "z3": np.float64,
    "sz3": np.float64,
    "q3": np.float64,
    "o3": np.float64,
    "f3": np.int64,
    "g3": object,
    "NMAST": object,
    "ps_tile_id": object,
    "tile_id": object,
    "exp_id": object,
    "filt_id": object,
    "unk": object,
    "chip_id": np.int64,
}

# this array is useful for looping over photometry methods
# you can concatenate it with z, sz, q, o, f, and g
phot_method_ids = ['1', '2', '3']
# define these here so you don't have to define it every time
z_cols = ['z' + i for i in phot_method_ids]
sz_cols = ['sz' + i for i in phot_method_ids]
q_cols = ['q' + i for i in phot_method_ids]
o_cols = ['o' + i for i in phot_method_ids]
f_cols = ['f' + i for i in phot_method_ids]
g_cols = ['g' + i for i in phot_method_ids]


def get_point_source_catalog(ps_file=ks2_files[1], clean=True):
    """
    This function reads the KS2 FIND_NIMFO file that stores *every* point source

    Parameters
    ----------
    ps_file : pathlib.Path or string
      full path to the LOGR.FIND_NIMFO
    clean : bool [True]
      if False, return the cleaned point source catalog. If True, return the
      original KS2 catalog.

    Output
    ------
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
    )
    point_sources_df.rename(columns=nimfo_cols, inplace=True)
    # split the file identifier into the file number and extension number
    point_sources_df['chip_id'] = point_sources_df['exp_id'].apply(lambda x: int(x.split('.')[1]))
    point_sources_df['exp_id'] = point_sources_df['exp_id'].apply(lambda x: x.split('.')[0])
    point_sources_df = point_sources_df.astype(nimfo_dtypes, copy=True)

    # if desired, return the cleaned catalog. else, return raw
    if clean == True:
        point_sources_df = clean_point_source_catalog(point_sources_df)

    return point_sources_df

"""
It turns out there are a bunch of duplicated entries in both the FIND_NIMFO and master catalogs. Fortunately, they appear to be duplicated for every detection. We need to remove them.
"""

def remove_duplicates(ps_cat, master_cat, verbose=False):
    """
    Remove duplicate entries from the point source catalog and master catalogs.
    For all duplicate entries with different NMAST designations, a primary
    designation is chosen ("primary id"), and the alternative NMAST
    designations are referred to as "aliases".

    Parameters
    ----------
    ps_cat : pd.DataFrame
      pandas dataframe containing the point source catalog
    master_cat : pd.DataFrame
      pandas dataframe containing the master catalog
    verbose : bool [False]
      if True, print out the number of duplicates found

    Returns
    -------
    ps_cat_nodup : pd.DataFrame
      pandas dataframe of the point source catalog with duplicate entries removed
    master_cat_nodup : pd.DataFrame
      pandas dataframe of the master catalog that matches the remaining point sources
    """
    # drop the ps columns that will not be used to find duplicates
    ps_cols = list(ps_cat.columns)
    for i in ['NMAST', 'g0', 'g1', 'g2', 'g3', 'ps_tile_id',
              'tile_id', 'exp_id', 'filt_id', 'unk', 'chip_id']:
        ps_cols.pop(ps_cols.index(i))

    # drop the master columns that will not be used to find duplicates
    master_cols = list(master_cat.columns)
    for i in ['NMAST']:
        master_cols.pop(master_cols.index(i))

    # use groupby to find all the duplicates.
    # The reason to use groupby instead of drop_duplicates() or duplicated()
    # is so that you can track the names of the duplicates and create a lookup
    # table. N.B. no longer doing this
    # first, group by all the columns except the assigned labels
    ps_gb = ps_cat.groupby(ps_cols)
    # now, just keep the first index in all the groups
    keep_indices = [ind[0] for ind in ps_gb.groups.values()]
    # use set logic to find the indices to drop
    drop_indices = set.difference(set(ps_cat.index), set(keep_indices))
    if verbose == True:
        print(f"Number of duplicates found: {len(drop_indices)}")
    ps_cat_nodup = ps_cat.drop(index=drop_indices)
    # now reduced the master catalog to only the entries that have an NMAST in
    # the new reduced point source catalog
    keep_names = master_cat['NMAST'].isin(ps_cat_nodup['NMAST'].unique())
    master_cat_nodup = master_cat.loc[keep_names]

    return ps_cat_nodup, master_cat_nodup



"""
It is useful to quickly calculate the distances of objects from each other,
in a particular exposure. The following methods do that:
calc_exp_dist_from_obj efficiently gets the distances from a particular object, and
generate_exposure_distance_matrix uses this to get the distances of all objects
from each other
"""
def calc_exp_dist_from_obj(df, obj_id, set_nan=True):
    """
    Given a KS2 dataframe of a single exposure and a particular object,
    calculate the pixelwise distance of every other point source in that
    exposure from the specified object.

    Parameters
    ----------
    df : pd.DataFrame
      pandas dataframe for a single exposure with only one entry per astrophysical object
    obj_id : str
      identifier for the astrophysical object
    set_nan : bool [True]
      if True, set the distance of the object from itself to nan instead of 0
    """
    obj_row = df.query('NMAST == @obj_id').squeeze()
    obj_pos = obj_row[['xraw1', 'yraw1']]
    diff = df[['xraw1','yraw1']] - obj_row[['xraw1','yraw1']]
    # now set the index to be the obj_id
    # this is a little convoluted but it uses pandas' automatic
    # index matching to make sure the distances line up with the right
    # objects
    diff['NMAST'] = df['NMAST']
    diff.set_index('NMAST', inplace=True)
    dist = diff.apply(np.linalg.norm, axis=1)

    if set_nan == True:
        dist.loc[obj_id] = np.nan
    return dist

def generate_exposure_distance_matrix(df, same_nan=True):
    """
    Given a KS2 dataframe, compute a symmetric matrix of pixelwise distances.
    The dataframe should only contain point sources from the same exposure.
    this goes really slowly as the size of the dataframe increases; I'm not sure why

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
    obj_ids = df['NMAST'].values # all the object ids
    dist_mat = pd.DataFrame(data=np.nan,
                            index=obj_ids, columns=obj_ids,
                            dtype=np.float)
    for obj_id in dist_mat.columns:
        dist_mat[obj_id] = calc_exp_dist_from_obj(df, obj_id)

    # alternate algorithm:
    """
    dist_mat = df.apply(lambda x: calc_exp_dist_from_obj(df, x['NMAST']),
                        axis=1)
    dist_mat.set_index(dist_mat.columns)
    """

    # if desired, set the diagonal to nan instead of 0
    if same_nan == True:
        for col in dist_mat.columns:
            dist_mat.loc[col, col] = np.nan

    return dist_mat


"""
I have found it helpful to collect all the neighbors of an a star -- in a
particular exposure -- within some radius. This helps with plotting to see
if the nearby point sources are real or spurious.
"""
def get_exposure_neighbors(catalog, obj_id, exp_id, radius):
    """
    Get all the point sources within a certain radius of an object in a
    particular exposure. This tests the selected obj_id's xraw1 and yraw1
    against those of the rest of the objects in that exposure.
    Returns a dataframe of the neighbors.

    Parameters
    ----------
    catalog : pd.DataFrame
      a catalog of point sources
    obj_id : str
      the stellar object identifier. Only one should be present in any exposure
    exp_id : str
      identifier for the exposure
    radius : int
      distance in pixels from the object to keep

    Returns
    -------
    neighbor_df : pd.DataFrame
      catalog entries for the neighbors
    """
    # untested
    # first, make sure the catalog is a single exposure
    exp_df = catalog.query("exp_id == @exp_id")
    dist = calc_exp_dist_from_obj(exp_df, obj_id)
    neighbors = dist[dist <= radius].index
    neighbor_df = exp_df[exp_df['NMAST'].isin(neighbors)]
    return neighbor_df


"""
This utility does all the work to pull out a full exposure, given the KS2 file ID.
"""
def get_img_from_ks2_file_id(ks2_exp_id, hdr='SCI'):
    """
    Pull out an image from the fits file, given the KS2 exposure identifier.

    Parameters
    ----------
    ks2_exp_id : str
      the exposure identifier assigned by KS2
    hdr : str or int ['SCI']
      which header? allowed values: ['SCI','ERR','DQ','SAMP','TIME']

    Returns:
    img : numpy.array
      2-D image from the fits file
    """
    flt_name = get_file_name_from_ks2id(ks2_exp_id)
    flt_path = shared_utils.get_data_file(flt_name)
    img = fits.getdata(flt_path, hdr)
    return img


"""
This is a wrapper for get_stamp, which is a little clunky to use by itself.
This takes in a row of the FIND_NIMFO catalog and pulls out a stamp for that
point source, in that file, of the specified size
"""
def get_stamp_from_ks2(row, stamp_size=11, return_img_ind=False):
    """
    Given a row of the FIND_NIMFO dataframe, this gets a stamp of the specified
    size of the given point source
    TODO: accept multiple rows

    Parameters
    ----------
    row : pd.DataFrame row
      a row containing the position and file information for the source
    stamp_size : int or tuple [11]
      (row, col) size of the stamp [(int, int) if only int given]
    return_img_ind : bool (False)
      if True, return the row and col indices of the stamp in the image

    Returns
    -------
    stamp_size-sized stamp
    """
    # get the file name where the point source is located and pull the exposure
    flt_file = get_file_name_from_ks2id(row['exp_id'])
    img = fits.getdata(shared_utils.get_data_file(flt_file), 1)
    # location of the point source in the image
    xy = row[['xraw1','yraw1']].values
    # finally, get the stamp (and indices, if requested)
    return_vals = image_utils.get_stamp(img, xy, stamp_size, return_img_ind)
    return return_vals



"""
I need to have a single function that contains all the steps needed to turn the KS2 output into a catalog that's ready for me to use. The steps are:
1. Process INPUT.KS2 to get the file and filter name mappers
2. Process and clean the point source catalog
3. Process and clean the master catalog.
4. Subtract 1 from all the xraw1 and yraw1 values to convert from FORTRAN to PYTHON!
5. Make sure all the columns have the write dtype and have no flag values. All flag values must be handled appropriately, and changes matched between the master and point source catalogs.
This all can be found in the notebook: ???
"""

# we also need some helper functions

def clean_catalog_dtypes(cat, dtype_dict):
    """
    Make sure each entry in a catalog has the write data type.
    If it doesn't, assign None as the universal null value
    Parameters
    ----------
    cat : pd.Dataframe
      the catalog where each column has an appropriate data type
    dtype_dict : dict
      a dictionary whose keys are the column names and whose values are the dtypes

    Output
    ------
    fixed : pd.DataFrame
      dataframe identical to cat except the wrongly-typed values have been set to None
    """
    def col2dtype(x, dtype):
        # do this as a function so you can put it in pd.DataFrame.apply()
        try:
            return dtype(x)
        except ValueError:
            return None
    fixed = cat.copy()
    for col in cat.columns:
        dtype = dtype_dict[col]
        # pandas stores strings as objects
        if dtype == object:
            dtype = str
        fixed[col] = cat[col].apply(col2dtype, args=[dtype])
    del cat
    return fixed

"""
We also want to apply the following requirements to the catalog:
1. all q > 0
2. all z > 0
3. an object must be detected in at least 10 different exposures
"""
"""
We only want to keep stars that are detected in at least N (probably 10) exposures,
to remove spurious detections. This function cuts on that threshold
"""
def catalog_cut_ndet(cat, ndet_min=9, mast_cat=None):
    """
    This function only keeps stars that are detected more than ndet times in all exposures.

    Parameters
    ----------
    cat : pd.DataFrame
      a catalog of point source detections
    ndet_min : int [10]
      the lower threshold on the number of detections (>=)
    mast_cat : None or pd.DataFrame
      if a dataframe containing the master catalog is given, make sure the
      entries in the master catalog are consistent with the new cuts in the
      point source catalog
      NOTE: this functionality is deprecated and has been moved to
        clean_master_catalog(). Kept for backwards compatibility

    Returns
    -------
    cut_cat : pd.DataFrame
      a catalog containing only stars with >= ndet_min detections
    """
    # collect the number of detections per point source
    ndet = cat.groupby('NMAST').size()
    # collect the object ids for those that satisfy the threshold
    keep_objs = pd.Series(ndet[ndet >= ndet_min].index)
    # and finally use them to index the catalog
    cut_cat = cat[cat['NMAST'].isin(keep_objs)].copy()

    return cut_cat

def clean_point_source_catalog(cat, q_min=0, z_min=0,
                               cut_ndet_args={}):
    """
    This function, when called, cleans the point source catalog by applying the listed series of cuts:
    1) q > 0
    2) z > 0

    Parameters
    ----------
    cat : pd.DataFrame
      point source catalog
    q_min : float [0]
      lower bound on PSF fit parameter
    z_min : float
      lower bound on flux
    cut_ndet_args : dict [{}]
      arguments to pass to catalog_cut_ndet, e.g. {'ndet_min': 9}

    Output
    -------
    clean_cat : pd.DataFrame
      point source catalog with cuts applied
    """
    # first, make sure all the dtypes are correct:
    clean_cat = clean_catalog_dtypes(cat, nimfo_dtypes)

    # OK, now apply cuts

    # cut for q > 0 and z > 0 on *all* q and z
    q_gt_0 = " and ".join([f"{i} > 0" for i in q_cols])
    z_gt_0 = " and ".join([f"{i} > 0" for i in z_cols])
    qz_gt_0 = f"{q_gt_0} and {z_gt_0}"
    clean_cat = clean_cat.query(qz_gt_0).copy()

    # cut for q2 > 0.85
    q_min = 0.85
    q2_gt_85 = f"q2 >= {q_min}"
    clean_cat = clean_cat.query(q2_gt_85).copy()

    # finally, cut on the number of detections
    clean_cat = catalog_cut_ndet(clean_cat, **cut_ndet_args)

    return clean_cat


"""
Clean the master catalog:
1. Any value that cannot be converted to the official column dtype set to nan
2. Remove stars that are not in the point source catalog
3. 
"""
def recompute_master_catalog(mast_cat, ps_cat):
    """
    After making cuts on the point source catalog, recompute relevant
    values in the master catalog
    Parameters
    ----------
    mast_cat : pd.DataFrame
      the master catalog
    ps_cat : pd.DataFrame
      the point source catalog

    Output
    ------
    mast_cat_new : pd.DataFrame
      the master catalog with updated values
    """
    # compute the new PS values
    mean_cols = ['z2','sz2','q2']
    # group the point source catalog by star and filter to make computing faster
    ps_gb = ps_cat.groupby(["NMAST","filt_id"])
    ps_mean = ps_gb[mean_cols].apply(np.mean)
    ps_mean['f2'] = ps_gb.size()

    # separate the ps_mean dataframe by filter
    filt_mean_dfs = {}
    for filt in ps_cat['filt_id'].unique():
        # compute the new mean z2, sz2, and q2
        filt_mean_dfs[filt] = ps_mean.loc[(slice(None), filt), :].copy()
        # rename the columns to match the master catalog columns
        filt_num = filt[-1]
        filt_mean_dfs[filt].rename(columns = {'z2' : 'zmast'+filt_num,
                                              'sz2': 'szmast'+filt_num,
                                              'q2' : 'q'+filt_num,
                                              'f2' : 'f'+filt_num},
                                   inplace=True)

    # Now, assign the recomputed values to the master catalog. Use a
    # dataframe as a "linking table" between "NMAST" and the mast_cat_clean index
    mast_cat_clean = mast_cat.copy()
    index_mapper = pd.Series(index=mast_cat_clean['NMAST'],
                             data=mast_cat_clean.index)
    for filt_id, filt_df in filt_mean_dfs.items():
        # OK, the big challenge isto align the catalogs
        filt_nmast = filt_mean_dfs[filt_id].index.get_level_values('NMAST')
        nmast_2_mastind = index_mapper.loc[filt_nmast]
        mast_cat_clean.loc[nmast_2_mastind, filt_df.columns] = filt_df.values.copy()

    return mast_cat_clean


def clean_master_catalog(mast_cat, ps_cat=None, verbose=False):
    """
    Clean the master catalog. If no point source catalog is given, this just checks the types and assigns proper null values. If a point source catalog is provided, then it also makes sure that the list of objects present in each catalog are in agreement.
    If an object is out of range, it gets dropped from the catalog
    Parameters
    ----------
    mast_cat : pd.DataFrame
      the master catalog
    ps_cat : pd.DataFrame [None]
      (optional) the point source catalog
    verbose : bool [False]
      verbose output. If True, print the number of objects dropped at each cut

    Output
    ------
    mast_cat_clean : pd.DataFrame
      the master catalog with dtypes fixed and all objects in agreement with the point source catalog
    """
    # fix any typing issues
    mast_cat = clean_catalog_dtypes(mast_cat, master_dtypes)

    # if no point source catalog is given, you're done
    if ps_cat is None:
        return mast_cat

    # otherwise, make compatible with the point source catalog

    # first, drop all the stars that are not in the ps catalog
    nmast_list = ps_cat['NMAST'].unique()
    mast_cat = mast_cat[mast_cat['NMAST'].isin(nmast_list)].copy()
    if verbose == True:
        print(f"# stars cut from PS catalog: {len(mast_cat) - len(mast_cat_clean)}")
    
    # second, if a star has no detections in a particular filter, set
    # that filter's z, q, o, f, and g properties to nan instead of the
    # flag values KS2 uses
    ps_gb = ps_cat.groupby('NMAST')
    # this function yields True if missing, False if present
    ismissing = lambda x: ~ks2_filtermapper['filter_id'].isin(x)
    missing_filters = ps_gb['filt_id'].aggregate('unique').apply(ismissing)
    missing_filters.columns = ks2_filtermapper['filter_id']
    # loop through the filters, select the stars that have no observations
    # in that filter (either because they were cut or because they were
    # never observed), and set the corresponding columns to nan
    nan_cols = ['zmast','szmast','q','o','f','g']
    for filt in missing_filters.columns:
        filt_num = str(filt[-1])
        filt_nan_cols = [i+filt_num for i in nan_cols]
        missing_ind = missing_filters.query(f"{filt} == True").index
        mast_cat.loc[mast_cat['NMAST'].isin(missing_ind),
                     filt_nan_cols] = np.nan

    # recompute star properties (z, sz, q, and f) with remaining stars
    mast_cat = recompute_master_catalog(mast_cat, ps_cat)

    return mast_cat


def process_ks2_input_catalogs(mast_cat, ps_cat):
    """
    This function reads the KS2 output files, applies the cleaning and cutting procedures, and writes them to tables in the data/tables directory
    Clean each catalog independently, and then make them compatible
    """
    # apply cuts to the PS catalog
    ps_cat_clean = clean_point_source_catalog(ps_cat)
    # bring the master catalog in compliance
    mast_cat_clean = clean_master_catalog(mast_cat, ps_cat_clean)

    # all done!
    return mast_cat_clean, ps_cat_clean



#####################################
# CONVERTING TO TR14 CATALOG TABLES #
#####################################
"""
This dictionary stores the conversion between the Tr14 table fields and the KS2 fields
"""
field_conversion_dict = {
    "star_id"        : "NMAST",
    "u_mast"         : "umast0",
    "v_mast"         : "vmast0",
    "star_phot_f"    : "zmast",
    "star_phot_e_f"  : "szmast",
    "ps_exp_id"      : "exp_id",
    "ps_filter_id"   : "filt_id",
    "ps_phot"        : "z2",
    "ps_phot_e"      : "sz2",
    "ps_x_exp"       : "xraw1",
    "ps_y_exp"       : "yraw1",
}

def ks2_cat_to_tr14(cat):
    """
    Take the KS2 catalog and put it in the right format for the Tr14 database
    Parameters
    ----------
    cat : pd.DataFrame
      a KS2 catalog

    Output
    ------

    """
    pass



if __name__ == "__main__":
    # run it in script mode to get all the dataframes
    ks2_filemapper = get_file_mapper(ks2_files[2])
    ks2_filtermapper = get_filter_mapper(ks2_files[2])
    ks2_mastercat = get_master_catalog(ks2files[0])
    ks2_allsources = get_point_source_catalog(ks2files[1])
