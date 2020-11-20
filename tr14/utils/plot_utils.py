import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from . import table_utils

filt1_label = table_utils.get_filter_name_from_filter_id('F1')
filt2_label = table_utils.get_filter_name_from_filter_id('F2')
filt_label = {'F1': filt1_label,
              'F2': filt2_label}


def plot_cmd(df, ax=None, ax_args={}):
    """
    Given a stars table, plot the CMD

    Parameters
    ----------
    df : pd.DataFrame
      stars catalog
    ax : matplotlib axis object [None]
      axis object to draw on
    ax_args : dict [{}]
      arguments to pass to the axis
    Output
    ------
    fig : matplotlib Figure object
    """
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlabel(f"{filt_label['F1']}-{filt_label['F2']}")
    ax.set_ylabel(f"{filt_label['F1']}");

    # set default values
    ax_args.setdefault('marker', '.')
    ax_args.setdefault('s', 50)
    ax_args.setdefault('lw', 0)
    ax_args.setdefault('ec', 'none')
    ax_args.setdefault('alpha', 1)
    ax.scatter(df['star_mag_F1']-df['star_mag_F2'],
               df['star_mag_F1'],
               **ax_args,
               )
    # axis inversion
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim), min(ylim))

    ax.legend()

    fig = ax.get_figure()
    return fig
