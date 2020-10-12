import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from . import table_utils

filt1_label = table_utils.get_filter_name_from_filter_id('F1')
filt2_label = table_utils.get_filter_name_from_filter_id('F2')
filt_label = {'F1': filt1_label,
              'F2': filt2_label}


def plot_cmd(df, ax=None):
    """
    Given a stars table, plot the CMD

    Parameters
    ----------
    df : pd.DataFrame
      stars catalog
    ax : matplotlib axis object [None]
      axis object to draw on
    Output
    ------
    fig : matplotlib Figure object
    """
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel(f"{filt_label['F1']}-{filt_label['F2']}")
    ax.set_ylabel(f"{filt_label['F1']}");

    
    ax.scatter(df['star_mag_F1']-df['star_mag_F2'],
               df['star_mag_F1'],
               marker='.', s=50, lw=0, ec='none', alpha=1)
    # axis inversion
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim), min(ylim))

    fig = ax.get_figure()
    return fig
