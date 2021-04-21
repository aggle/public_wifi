import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from . import ks2_utils

filt1_label = ks2_utils.get_filter_name_from_ks2id('F1')
filt2_label = ks2_utils.get_filter_name_from_ks2id('F2')
filt_label = {'F1': filt1_label,
              'F2': filt2_label}


def plot_cmd_ks2(df, ax=None):
    """
    Given a KS2-format master catalog dataframe, plot the CMD

    Parameters
    ----------
    df : pd.DataFrame
      KS2 master catalog dataframe
    ax : matplotlib axis object [None]
      axis object to draw on
    Output
    ------
    fig : matplotlib Figure object
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(f"{filt_label['F1']}-{filt_label['F2']}")
    ax.set_ylabel(f"{filt_label['F1']}");


    ax.scatter(df['mmast1-mmast2'], df['mmast1'],
               marker='.', s=50, lw=0, ec='none', alpha=1)
    # axis inversion
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim), min(ylim))

    fig = ax.get_figure()
    return fig
