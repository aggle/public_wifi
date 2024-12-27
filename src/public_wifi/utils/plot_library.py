"""
Holder for plots that you make repeatedly
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

from public_wifi import misc


def _contrast_list2map(
        contrast_df : pd.DataFrame,
        stamp_shape : int,
        thresh_col : str = '5'
):
    """
    Put the contrast dataframe in a 2-D image
    
    Parameters
    ----------
    contrast_df : pd.DataFrame
      a dataframe with a column for the pixel coordinates and the contrast at a
      particular threshold
    stamp_shape : int
      length of the side of the stamp
    thresh : int = '5'
      which threshold column to display
    """
    center = misc.get_stamp_center(stamp_shape)
    contrast_map = np.zeros((stamp_shape, stamp_shape), dtype=float) * np.nan
    for i, row in contrast_df.iterrows():
        x, y = int(row['x']+center[0]), int(row['y']+center[1])
        contrast_map[y, x] = row[thresh_col]
    return contrast_map

def plot_contrast(star):
    """
    Plot the contrast curves for a star.
    Reads the star.contrast_maps attribute
    """
    contrast_maps = star.contrast_maps
    n_maps = len(contrast_maps)
    ncols = n_maps
    nrows = 2

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize = (4*ncols, 4*nrows),
        layout='constrained',
        sharey='row', sharex='row',
        squeeze=False,
    )
    fig.suptitle(f"{star.star_id}\nContrast plots")

    for ax_col, contrast_df in zip(axes.T, contrast_maps.items()):
        map_id, contrast_df = contrast_df
        map_name = ' '.join(star.cat.loc[map_id, star.match_by].values)

        ax_col[0].set_title(map_name)
        # make the map
        ax = ax_col[0]
        contrast_map = _contrast_list2map(contrast_df, star.stamp_size, '5')
        # compute the radial contrast
        imax = ax.imshow(contrast_map, origin='lower', vmax=1)
        fig.colorbar(imax, ax=ax)

        ax = ax_col[1]
        contrast_df['rad'] = contrast_df.apply(
            lambda row: np.linalg.norm(row[['x','y']]),
            axis=1
        )
        contrast_rad = contrast_df.drop(columns=['x','y']).groupby("rad").mean()
        for sigma_thresh in contrast_rad.columns:
            ax.plot(
                contrast_rad.index,
                contrast_rad[sigma_thresh],
                label=sigma_thresh
            )
        ax.legend(title='Sigma threshold')
        # print(map_name, contrast_df)
        ax.set_yscale("log")
    return fig
