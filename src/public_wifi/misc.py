"""
Useful tools
"""
import numpy as np
import pandas as pd

def get_stamp_center(stamp : np.ndarray | pd.Series) -> np.ndarray:
    """
    Get the central pixel of a stamp or cube
    """
    if isinstance(stamp, pd.Series):
        stamp = np.stack(stamp.values)
    shape = np.array(stamp.shape[-2:])
    center = np.floor((shape/2)).astype(int)
    return center
