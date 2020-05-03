"""
Routines for finding sources in the Tr14 data
"""

import numpy
import pandas as pd

"""
Architecture:
Find something
"""

def sources2df(table, file_id):
    """
    When you put the point sources in a dataframe, you need to add/modify some columns:
    id -> image_ps_id (point source ID for *this* image)
    file_id : the file identifier that the source comes from

    Parameters
    ----------
    table : pd.DataFrame
      The source dataframe, output of daofind
    file_id : str
      The file identifier that the source comes from ('ROOTNAME' keyword in PRI header)

    Returns
    -------
    source_df : pd.DataFrame
      Dataframe of point sources, with updated columns and other values

    """
    source_df = table.copy()
    source_df['file_id'] = file_id
    source_df.rename(columns={'id':'ps_img_id'}, inplace=True)
    return source_df

