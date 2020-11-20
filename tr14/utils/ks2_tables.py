"""
This module takes the KS2 tables and converts them to the final Tr14 format
The tables are defined in tr14/docs/database/list_of_tables.org
"""

import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from . import shared_utils
from . import header_utils
from . import ks2_utils
from . import table_utils
from . import image_utils

def frtn2py_ind(ind):
    """
    Given pixel indices (dataframe, array, whatever), subtract 1 to have them
    be indexed from 0 instead of 1
    Returns the same object with all the values subtracted by 1
    """
    """
    new_coords = ind - 1
    return new_coords
    """
    return ind

def make_stars_table(mast_cat):
    """
    Take the KS2 master catalog and put it in the right format for the Tr14 database

    Parameters
    ----------
    mast_cat : pd.DataFrame
      a KS2 master catalog

    Output
    ------
    stars_table : pd.DataFrame
      a master catalog according to the Tr14 specifications
    lookup_nmast : pd.DataFrame
      dataframe linking the new "star_id" field to the KS2 NMAST identifier
    """
    # translate between KS2 and star_table column names
    columns = dict(NMAST = "star_id",
                   umast0 = "u_mast",
                   vmast0 = "v_mast")
    for i in ks2_utils.ks2_filtermapper['filt_id']:
        columns[f"zmast{i[-1]}"] = f"star_phot_{i}"
        columns[f"szmast{i[-1]}"] = f"star_phot_e_{i}"

    # pull out the columns and convert the nanmes
    stars_table = mast_cat[columns.keys()].copy()
    stars_table = stars_table.rename(columns=columns)
    stars_table = stars_table.reset_index(drop=True)

    # make some changes
    # star IDs
    ks2_ids = stars_table['star_id'].copy()
    star_ids = stars_table.apply(lambda x: f"S{x.name:06d}", axis=1)
    lookup_nmast = pd.DataFrame(data=zip(star_ids, ks2_ids), columns=['star_id','NMAST'])
    stars_table['star_id'] = star_ids
    # transform coordinates from fortran to python
    stars_table[['u_mast','v_mast']] = frtn2py_ind(stars_table[['u_mast',
                                                                'v_mast']])
    # add the membership column
    stars_table['clust_memb'] = None
    # compute magnitudes
    for i in ks2_utils.ks2_filtermapper['filt_id']:
        # mag
        col_name = f"star_mag_{i}"
        mag = -2.5*np.log10(stars_table[f"star_phot_{i}"])
        stars_table[col_name] = mag
        # mag uncertainty
        col_name = f"star_mag_e_{i}"
        e_mag = stars_table[f"star_phot_e_{i}"]/stars_table[f"star_phot_{i}"]
        e_mag = e_mag / np.log(10)
        stars_table[col_name] = e_mag

    return stars_table, lookup_nmast


def make_point_source_table(ps_cat, lookup_nmast):
    """
    Take the KS2 point source catalog and put it in the right format for the
    Tr14 database

    Parameters
    ----------
    ps_cat : pd.DataFrame
      a KS2 point source catalog
    lookup_nmast : pd.DataFrame
      a dataframe that maps between the star_id and corresponding KS2 NMAST
      value; otherwise there is no map between the PS table and star table!

    Output
    ------
    ps_table : pd.DataFrame
      a point source catalog according to the Tr14 specifications

    """
    # translate between KS2 and star_table column names
    columns = dict(NMAST = "ps_star_id",
                   umast = "ps_u_mast",
                   vmast = "ps_v_mast",
                   exp_id = "ps_exp_id",
                   filt_id = "ps_filt_id",
                   epoch_id = "ps_epoch_id",
                   q2 = "ps_psf_fit",
                   z2 = "ps_phot",
                   sz2 = "ps_phot_e",
                   xraw1 = "ps_x_exp",
                   yraw1 = "ps_y_exp",
    )
    # drop stars that aren't in the master table because their ps_star_ids
    # won't make sense
    ps_cat = ps_cat.loc[ps_cat['NMAST'].isin(lookup_nmast['NMAST'])].copy()

    # pull out the columns and convert the names
    ps_table = ps_cat[columns.keys()].reset_index(drop=True)
    ps_table = ps_table.rename(columns=columns)

    # generate the unique key for ps_id!
    # first, reset the index to be continous, counting from 0
    ps_table = ps_table.reset_index(drop=True)
    # second, assign the integers to a column, so the ps_id is independent
    # give it the same format as stars, with P in front of the identifier
    ps_table['ps_id'] = ps_table.index.copy()
    ps_table['ps_id'] = ps_table['ps_id'].apply(lambda x: f"P{x:06d}")

    # change some leading letters
    # ps_star_id - use the lookup table
    new_ids = lookup_nmast.set_index("NMAST").loc[ps_table['ps_star_id']]
    ps_table['ps_star_id'] = new_ids.squeeze().reset_index(drop=True)
    # exposure ID
    new_ids = ps_table['ps_exp_id'].apply(lambda x: x.replace("G","E"))
    ps_table['ps_exp_id'] = new_ids
    # epoch (date) ID
    new_ids = ps_table['ps_epoch_id'].apply(lambda x: x.replace("E","D"))
    ps_table['ps_epoch_id'] = new_ids

    # transform coordinates from fortran to python
    ps_table[['ps_u_mast','ps_v_mast']] = frtn2py_ind(ps_table[['ps_u_mast',
                                                                'ps_v_mast']])
    ps_table[['ps_x_exp','ps_y_exp']] = frtn2py_ind(ps_table[['ps_x_exp',
                                                              'ps_y_exp']])


    # compute magnitudes
    # mag
    col_name = "ps_mag"
    mag = -2.5*np.log10(ps_table["ps_phot"])
    ps_table[col_name] = mag
    # mag uncertainty
    col_name = "ps_mag_e"
    e_mag = ps_table[f"ps_phot_e"]/ps_table[f"ps_phot"]
    e_mag = e_mag / np.log(10)
    ps_table[col_name] = e_mag

    return ps_table


def generate_stamp_table(ps_table):
    """
    Generate a stamp table to hold the stamp metadata and arrays

    Parameters
    ----------
    ps_table : pd.DataFrame
      point source table

    Output
    ------
    stamp_table : pd.DataFrame
      table of stamps and stamp metadata

    """
    print("Generating stamps... go get some coffee, this can take a while.")
    stamps = ps_table.set_index('ps_id').apply(image_utils.get_stamp_from_ps_row,
                                               return_img_ind=False,
                                               axis=1)
    stamps.index.name = 'ps_id'
    stamps = stamps.reset_index(name='stamp_array')
    stamps['stamp_id'] = stamps['ps_id'].apply(lambda x: x.replace("P","T"))

    print("Stamps finished!")
    # stamp info table
    columns = {'stamp_id': str,
               'stamp_ps_id': str,
               'stamp_star_id': str,
               'stamp_x_cent': int,
               'stamp_y_cent': int,
               'stamp_path': str,
               'stamp_ref_flag': bool,
               'stamp_array': object}
    stamp_table = pd.DataFrame(data=None, columns=columns.keys(), index=stamps.index)
    stamp_table['stamp_ps_id'] = stamps['ps_id'].copy()
    stamp_table['stamp_id'] = stamps['stamp_id'].copy()
    # use the ps_id to index the ps table
    ps_table = ps_table.set_index('ps_id')
    # get the star_ids
    stamp_table['stamp_star_id'] = ps_table.loc[stamp_table['stamp_ps_id'], 'ps_star_id'].values[:]
    # get the centers of the stamps
    stamp_table['stamp_x_cent'] = ps_table.loc[stamp_table['stamp_ps_id'], 'ps_x_exp'].apply(lambda x: np.int(np.floor(x))).values[:]
    stamp_table['stamp_y_cent'] = ps_table.loc[stamp_table['stamp_ps_id'], 'ps_y_exp'].apply(lambda x: np.int(np.floor(x))).values[:]
    # paths to the stamp files
    # NOTE: DEPRECATED, FOR NOW STORING ARRAY DIRECTLY IN THE DATAFRAME
    stamp_table['stamp_path'] = stamp_table['stamp_id'].apply(lambda x: f"/stamp_arrays/{x}")
    # store the arrays
    stamp_table['stamp_array'] = stamps['stamp_array'].copy()
    # assert data types and return
    for k, v in columns.items():
        stamp_table[k] = stamp_table[k].astype(v)
    return stamp_table


def write_fundamental_db(db_file=shared_utils.db_raw_file, stamps=False):
    """
    Write the basic tables to the database file:
    - stars
    - point_sources
    - lookup_files
    - lookup_filters
    - lookup_nmast
    - fits headers
    - stamps
    - stamp_arrays
    If this file exists, it gets clobbered.

    Parameters
    ----------
    db_file : str or Path [shared_utils.db_file]
    stamps : bool [False]
      if True, generate the stamps too (takes a long time)

    Output
    ------
    generates {0}

    """.format(shared_utils.db_file)

    # use this dict to collect all the tables for writing
    master_tables_dict = {}

    # get cleaned master and point source catalogs
    mast_cat, ps_cat = ks2_utils.get_ks2_catalogs(mast_file=ks2_utils.ks2_files[0],
                                                  ps_file=ks2_utils.ks2_files[1],
                                                  raw=False)
    # convert mast_cat to the proper format, and get the stars-KS2 lookup table
    stars_table, lookup_ks2_nmast = make_stars_table(mast_cat)
    master_tables_dict['stars'] = stars_table
    master_tables_dict['lookup_ks2_nmast'] = lookup_ks2_nmast

    # convert the point source catalog to the proper format
    ps_table = make_point_source_table(ps_cat, lookup_ks2_nmast)
    master_tables_dict['point_sources'] = ps_table

    # mapper between file names and file ids
    lookup_files = ks2_utils.ks2_filemapper.copy()
    lookup_files['file_id'] = lookup_files['file_id'].apply(lambda x: x.replace("G","E"))
    master_tables_dict['lookup_files'] = lookup_files

    # mapper between the filter names and filter ids
    lookup_filters = ks2_utils.ks2_filtermapper.copy()
    master_tables_dict['lookup_filters'] = lookup_filters

    # FITS header tables
    headers = header_utils.load_headers('all')
    for k, v in headers.items():
        master_tables_dict["hdr_"+k] = v

    # stamp table
    if stamps == True:
        stamp_table = generate_stamp_table(ps_table)
        master_tables_dict['stamps'] = stamp_table

    # write all the tables to the database file
    with pd.HDFStore(db_file, mode='w') as store:
        for k, v in sorted(master_tables_dict.items()):
            print(f"Writing table `{k}`...")
            # suppress PerformanceWarnings when saving non c-mapping objects
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.put(k, v, format='fixed', data_columns=True)

            print("\tDone.")
        store.close()
    print("Finished.")



def make_cand_master_table():
    """
    A master catalog for the candidates (i.e. each entry is for a unique candidate)

    Parameters
    ----------

    Output
    ------

    """
    pass


def make_cand_ps_table(args):
    """
    A point source catalog for the candidates (i.e. each entry is for each
    detection of a candidate)

    Parameters
    ----------

    Output
    ------

    """
    pass

