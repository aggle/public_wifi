"""
Helper functions for catalog matching
"""

import pandas as pd
from pathlib import Path


cat_fit_columns = [
    'x_ref', #     Column 1: X (reference)
    'y_ref', #     Column 2: Y (reference)
    'x_input', #     Column 3: X (input)
    'y_input', #     Column 4: Y (input)
    'x_fit', #     Column 5: X (fit)
    'y_fit', #     Column 6: Y (fit)
    'x_resid', #     Column 7: X (residual)
    'y_resid', #     Column 8: Y (residual)
    'x_orig_ref', #     Column 9: Original X (reference)
    'y_orig_ref', #     Column 10: Original Y (reference)
    'x_orig_input', #     Column 11: Original X (input)
    'y_orig_input', #     Column 12: Original Y (input)
    'ref_id', #     Column 13: Ref ID
    'input_id', #     Column 14: Input ID
    'input_extver_id', #     Column 15: Input EXTVER ID 
    'ra', #     Column 16: RA (fit)
    'dec', #     Column 17: Dec (fit)
    'ref_source' #     Column 18: Ref source provenience
]

cat_xy_columns = [
    "Ref_X",
    "Ref_Y",
    "Input_X",
    "Input_Y",
    "Ref_X0",
    "Ref_Y0",
    "Input_X0",
    "Input_Y0",
    "Ref_ID",
    "Input_ID",
    "Ref_Source"
]

def catalog2pandas(filename):
    """
    Takes a *catalog_fit.match file and returns it as a pandas dataframe
    Parameters
    ----------
    filename : None
      full path to the *catalog_fit.match file

    Returns
    -------
    df : dataframe with the catalog information
    """
    nheader = 28 # number of lines to skip
    df = pd.read_csv(filename, sep=' *',
                     names=cat_fit_columns,
                     skiprows=nheader,
                     engine='python'
    )
    df['file_id'] = Path(filename).name.split('_')[0]
    return df


if __name__ == "__main__":
    pass
