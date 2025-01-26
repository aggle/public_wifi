"""
Holder for plots that you make repeatedly
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

from public_wifi import misc
from public_wifi import starclass as sc


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
        norm = mpl.colors.LogNorm(vmax=1)
        imax = ax.imshow(contrast_map, origin='lower', norm=norm)
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

def mf_diagnostic_plots(
    star : sc.Star,
	cat_row : pd.Series,
    pca_df :  pd.DataFrame,
	mode_num : int,
    inj_flux : float | None = None,
	contrast : float | None = None
):
    imgs = {'stamp': star.cat.loc[cat_row.name, 'stamp']}
    # pca_df = pca_results.loc[cat_row.name]
    pca_row = pca_df.loc[mode_num]
    img_cols = [
        'klip_model', 'klip_sub', 'klip_basis', 'mf',
        'pca_bias', 'mf_map', 'detmap', 'fluxmap'
    ]
    imgs.update({col: pca_row[col] for col in img_cols})
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(4*4, 2*4))
    fig.suptitle(f"{star.star_id}\nKklip = {mode_num}")
    for ax, (col, img) in zip(axes.flat, imgs.items()):
        ax.set_title(col)
        imax = ax.imshow(img)
        fig.colorbar(imax, ax=ax, orientation='horizontal')

    pca_index = pca_df.index
    ax = axes[-1, 1]
    ax.set_title("Recovered flux")
    ax.set_xlabel("Kklip")
    if inj_flux is not None:
        ax.axhline(inj_flux, ls='--', c='k', label='Injected flux')
    ax.plot(pca_index, pca_df['detmap_posflux'], label='No KL corr')
    ax.plot(pca_index, pca_df['fluxmap_posflux'], label='With KL corr')
    ax.legend()

    ax = axes[-1, 2]
    ax.set_title("Contrast normalized to MF width")
    ax.set_xlabel("Kklip")
    if contrast is not None:
        ax.axhline(contrast, ls='--', c='k', label='Injected contrast')
    ax.plot(pca_index, pca_df['detmap_posflux']/pca_df['mf_prim_flux'], label='No KL corr')
    ax.plot(pca_index, pca_df['fluxmap_posflux']/pca_df['mf_prim_flux'], label='With KL corr')
    ax.legend()
    return fig
