"""
Functions for gathering and using data quality information
"""

import numpy as np
import pandas as pd

from astropy.io import fits

from . import table_utils, image_utils, header_utils


"""
First, let's set up a wrapper to pull all the data quality stamps 
"""

######################
# Data Quality Table #
######################

def get_dq_stamps(point_sources_table, get_stamp_args={}):
    """
    Given a point sources table, create a table for the data quality flags.
    This can take a while to run.


    Parameters
    ----------

    Output
    ------

    """
    pass


def get_dq_table(point_sources_table, get_stamp_args={}):
    """
    Given a point sources table, create a table for the data quality flags.
    This can take a while to run.

    Parameters
    ----------
    point_sources_table : pd.DataFrame
      a point sources table - either full, or a subset
    get_stamp_args : {}
      keyword arguments to pass to image_utils.get_stamps_from_ps_table

    Output
    ------
    dq_table : pd.Series
      table whose index is the stamp ID and whose value is a dict of pixel
      coordinates for each flag

    """
    # make sure you're getting the data quality header
    get_stamp_args['hdr'] = get_stamp_args.get('hdr', 'DQ')
    dq_stamps = image_utils.get_stamps_from_ps_table(point_sources_table,
                                                     kwargs=get_stamp_args)
    dq_stamps = dq_stamps.reset_index(name='dq_array')
    dq_stamps['dq_array'] = dq_stamps['dq_array'].apply(lambda x: x.astype(np.int16))
    dq_stamps['dq_star_id'] = pd.merge(dq_stamps,
                                       point_sources_table,
                                       on='ps_id')['ps_star_id']
    dq_stamp_flags = dq_stamps.set_index('ps_id')['dq_array'].apply(header_utils.parse_dq_array)
    # now set the *stamp* id as the index
    lookup_ps_stamp_id = table_utils.load_table('lookup_ps_stamp_id')
    dq_stamp_flags = dq_stamp_flags.reset_index().merge(lookup_ps_stamp_id,
                                                        on='ps_id')
    dq_stamp_flags = dq_stamp_flags.set_index("stamp_id")['dq_array']
    return dq_stamp_flags
