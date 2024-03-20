"""
This tool writes all the tables you need once you give it the catalog of point sources
"""
from pathlib import Path
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy import nddata

from . import table_utils
from . import table_io


# generic tool to generate unique IDs for some column in the point source table
def generate_mapping_table(
        values : list,
        prefix : str,
        column_names : list[str],
) -> pd.DataFrame :
    """
    Generate a mapping table from some set of values. Assigns each the label
    `prefix`+integer, where the integer is zero-padded and unique.

    Parameters
    ----------
    values : list-like of values
      if not unique, will be made unique
    prefix : str
      string prefix to the identifying numbers. Converted to UPPER
    column_names : list[str]
      a name for each column in order [id column, name column]

    Output
    ------
    2-column dataframe mapping each identifier to the value, with the column
    names
    """
    prefix = prefix.upper()
    # remove duplicates
    values = list(set(values))
    # find out how many digits you need for the unique IDs
    nints = len(str(len(values)))
    # start from 1, not 0
    mapping = [(f"{prefix}{i+1:0{nints}d}", v) for i, v in enumerate(values)]
    df = pd.DataFrame(mapping,
                      columns=column_names)
    return df


def generate_public_wifi_star_catalog(
    ps_table : pd.DataFrame    
) -> pd.DataFrame:
    """
    Generate the unique star dataframe from the point source catalog

    Parameters
    ----------
    ps_table : pd.DataFrame
      path to the location of the HST 17167 point source catalog

    Output
    ------
    star_catalog : pd.DataFrame
      catalog of astrophysical objects in the catalog
    """
    def group_star_info(group):
        # return the row info for a group
        row = pd.Series({
            'star_id': group.name,
            'u_mast': group['ps_u_mast'].mean(),
            'v_mast': group['ps_v_mast'].mean(),
            'star_phot_F1': group.query("ps_filt_id == 'F1'")['ps_phot'].mean(),
            'star_phot_e_F1': group.query("ps_filt_id == 'F1'")['ps_phot_e'].mean(),
            'star_phot_F2': group.query("ps_filt_id == 'F2'")['ps_phot'].mean(),
            'star_phot_e_F2': group.query("ps_filt_id == 'F2'")['ps_phot_e'].mean(),
            'clust_memb': True, # not used for this program
            'star_mag_F1': group.query("ps_filt_id == 'F1'")['ps_mag'].mean(),
            'star_mag_e_F1': group.query("ps_filt_id == 'F1'")['ps_mag_e'].mean(),
            'star_mag_F2': group.query("ps_filt_id == 'F2'")['ps_mag'].mean(),
            'star_mag_e_F2': group.query("ps_filt_id == 'F2'")['ps_mag_e'].mean(),
        })
        return row

    star_cat = ps_table.groupby("ps_star_id").apply(group_star_info, include_groups=False)
    return star_cat.reset_index(drop=True)


def generate_stamp_table(
        ps_table : pd.DataFrame,
        fits_folder : str | Path,
        file_mapper : pd.DataFrame,
        stamp_size : int = 15,
        verbose : bool = False
) -> pd.DataFrame :
    """
    Generate a stamp table to hold the stamp metadata and data arrays

    Parameters
    ----------
    ps_table : pd.DataFrame
      point source table
    fits_folder : str or Path
      path to the parent folder where the fits images are keps
    file_mapper : pd.DataFrame
      dataframe that maps the exposure names to the source fits file names
    stamp_size : int [15]
      dimension of the square stamp to cut out
    verbose : bool [False]
      print extra output

    Output
    ------
    stamp_table : pd.DataFrame
      table of stamps and stamp metadata

    """
    print("Generating stamps...")
    # the biggest bottleneck is file IO, so groupby by exposure so you only have
    # to read each file once
    # then reassemble the groups into a single Series
    # ps_id must be unique (but I don't actually guarantee it anywhere)
    gb_exp = ps_table.set_index('ps_id').groupby('ps_exp_id')
    def group_get_stamps(exp_group, verbose=verbose):
        """
        Operate on the group dataframe
        use nddata.Cutout2D to pull out the stamps from the image
        """
        exp_id = exp_group.name
        # make sure that the fits file mapper has the right naming convention
        if verbose == True:
            print(f"{exp_id} start")
        file_root = file_mapper.set_index("exp_id").loc[exp_id].squeeze()
        fits_file = Path(fits_folder) / file_root
        try:
            assert fits_file.exists()
        except AssertionError:
            print(f"{fits_file.as_posix()} not found")
            return None
        img = fits.getdata(fits_file, 'sci')
        # operate on dataframe rows
        row_func = lambda row: nddata.Cutout2D(img,
                                               tuple(row[['ps_x_exp','ps_y_exp']].values),
                                               size=stamp_size,
                                               mode='partial',
                                               fill_value=np.nan).data
        stamps = exp_group.apply(row_func, axis=1)
        if verbose == True:
            print(f"\t{exp_id} end")
        return stamps
    stamps = gb_exp.apply(group_get_stamps, verbose=verbose, include_groups=False)
    stamps = stamps.reset_index(name='stamp_array')
    # one stamp for each point source
    stamps['stamp_id'] = stamps['ps_id'].apply(lambda x: x.replace("P","T"))

    print("Stamps finished!")
    # stamp info table
    columns = {'stamp_id': str,
               'stamp_ps_id': str,
               'stamp_exp_id': str,
               'stamp_star_id': str,
               'stamp_x_cent': int,
               'stamp_y_cent': int,
               'stamp_ref_flag': bool,
               'stamp_array': object}
    stamp_table = pd.DataFrame(data=None,
                               columns=list(columns.keys()),
                               index=stamps.index)
    stamp_table['stamp_ps_id'] = stamps['ps_id'].copy()
    stamp_table['stamp_id'] = stamps['stamp_id'].copy()
    stamp_table['stamp_exp_id'] = stamps['ps_exp_id'].copy()

    # use the ps_id to index the ps table
    ps_table = ps_table.set_index('ps_id')
    # get the star_ids
    star_ids = ps_table.loc[stamp_table['stamp_ps_id'], 'ps_star_id'].values[:]
    stamp_table['stamp_star_id'] = star_ids
    # get the centers of the stamps
    stamp_x_cent = ps_table.loc[stamp_table['stamp_ps_id'], 'ps_x_exp']
    stamp_table['stamp_x_cent'] = stamp_x_cent.apply(lambda x: int(np.floor(x))).values[:]
    stamp_y_cent = ps_table.loc[stamp_table['stamp_ps_id'], 'ps_y_exp']
    stamp_table['stamp_y_cent'] = stamp_y_cent.apply(lambda x: int(np.floor(x))).values[:]
    # store the arrays
    stamp_table['stamp_array'] = stamps['stamp_array'].copy()
    # assert data types and return
    stamp_table = stamp_table.astype(columns)
    return stamp_table

# all together now
def convert_point_source_catalog_to_public_wifi(
        input_catalog : pd.DataFrame,
        column_mapper : dict,
        data_folder : str | Path,
        stamp_size : int = 15,
        db_file : str | Path = '',
        extra_columns : dict[str, type] = {},
) -> dict[str, pd.DataFrame]:
    """
    Once you have the point source catalog defined, this function converts it
    into PUBLIC-WIFI format. You just have to give it the right column names.
    You can get the right column names with the function,
    `table_utils.load_table_definition("POINT_SOURCES)`. Provide a dictionary
    with the required names as keys and the correspinding names from your
    catalog as values.
    It gives you a dictionary of all the tables you need to start out:
    - POINT_SOURCES
    - STARS
    - STAMPS
    - LOOKUP_STAR_PS_ID
    - LOOKUP_STAR_STAMP_ID
    - LOOKUP_PS_STAMP_ID
    - LOOKUP_FILES
    - LOOKUP_FILTERS

    Parameters
    ----------
    input_catalog : pd.DataFrame
      the input catalog of point sources and other parameters
    column_mapper : dict
      a dictionary whose keys are the columns of the PUBLIC WIFI catalog, and
      whose values are the columns of the input catalog that they correspond
      to, which can be found with the function
      table_utils.load_table_definition("POINT_SOURCES)
      If a corresponding column is not present, leave it out. It will be set to
      None.
      For `ps_id`, if no value is provided then the index will be used.
    data_folder : str | Path
      path to the parent folder where the fits files are stored (assumed flat
      dir structure)
    stamp_size : int = 15
      box size of the stamps to cut out
    db_file : str | Path = ''
      if this is a valid path, write the database object to this path. Else, do nothing.
    extra_columns : dict[str, type] = {}
      Extra columns from the input catalog that you want to include, along with datatype

    Output
    ------
    tables : dict
      the various tables that were generated. The key for the point source
      catalog is "point_sources"

    """
    # this dict will store the tables returned by this function
    tables = {}
    # make sure ps_id is set
    # if no column is passed, use the index
    if column_mapper.get('ps_id', '') == '':
        column_mapper['ps_id'] = 'index'
        input_catalog = input_catalog.reset_index(names='index')

    # Manage the list of columns you need to set
    # the canonical definition of the table
    ps_cols_def = table_utils.load_table_definition("POINT_SOURCES")
    # ps_cols will hold the passed columns that are not used as unique
    # identifiers for something in the dataset
    ps_cols = column_mapper.copy()
    # separate out the ID columns
    id_cols = {c: ps_cols.pop(c) for c in column_mapper.keys() if c[-3:] == '_id'}

    # Initialize the public wifi catalog that you will eventually return. For
    # the first step, just convert the column names for the columns for which
    # you can do that.
    cols = list(ps_cols.values())
    sloc = {v: k for k, v in ps_cols.items()} # 'cols', backwards!
    ps_table = input_catalog[cols].rename(columns=sloc)
    for col, dtype in extra_columns.items():
        ps_table['ps_'+col] = input_catalog[col].astype(dtype, copy=True)

    # make the id cols that create unique identifiers for
    # things like stars, point sources, files, filters
    # id_col is the PUBLIC-WIFI column name
    # cat_col is the input catalog column name
    for id_col , cat_col in id_cols.items():
        map_col = id_col if id_col == 'ps_id' else id_col.replace("ps_","")
        mapper = generate_mapping_table(input_catalog[cat_col],
                                        prefix=map_col[0],
                                        column_names = [map_col, cat_col])
        replace_dict = mapper.set_index(cat_col).squeeze()
        ps_table[id_col] = input_catalog[cat_col].replace(replace_dict)
        tables[map_col] = mapper

    # Whatever columns were not provided in the mapping dictionary, initialize
    # them to None
    for col in ps_cols_def.keys():
        if col not in ps_table.columns:
            ps_table[col] = None

    # # reorder columns for prettiness
    for col, dtype in ps_cols_def.items():
        ps_table[col] = ps_table[col].astype(dtype)
    tables['point_sources'] = ps_table

    # make the Star <-> Point Source lookup table
    star_ps_table = ps_table[['ps_star_id', 'ps_id']].copy()
    star_ps_table = star_ps_table.rename(columns={'ps_star_id': 'star_id'}).sort_values(by='star_id')
    tables['lookup_star_ps_id'] = star_ps_table

    # make the star table
    tables['stars'] = generate_public_wifi_star_catalog(ps_table)

    # make the stamps table
    tables['stamps'] = generate_stamp_table(
        ps_table,
        fits_folder = data_folder,
        file_mapper = tables['exp_id'],
        stamp_size = stamp_size,
        verbose = False
    )

    # point source <-> stamp lookup table
    ps_stamp_table = tables['stamps'][['stamp_ps_id', 'stamp_id']].copy()
    ps_stamp_table.rename(columns={'stamp_ps_id': 'ps_id'}, inplace=True)
    tables['lookup_ps_stamp_id'] = ps_stamp_table

    # star <-> stamp lookup table
    star_stamp_table = pd.merge(ps_stamp_table, tables['lookup_star_ps_id'], on='ps_id')
    star_stamp_table = star_stamp_table[['star_id', 'stamp_id']].sort_values(by='star_id')
    tables['lookup_star_stamp_id'] = star_stamp_table

    # rename the exp_id and filt_id tables
    tables['lookup_files'] = tables.pop('exp_id')
    tables['lookup_filters'] = tables.pop('filt_id')

    if db_file != '' and Path(db_file).parent.exists() == True:
        print(f"Writing database to {str(db_file)}.")
        for k, tab in tables.items():
            table_io.write_table(k, tab, pk=k, db_file=Path(db_file), clobber=True)

    return tables

