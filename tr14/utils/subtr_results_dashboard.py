"""
Result visualization dashboard
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from . import table_utils
from . import shared_utils
from . import plot_utils
from . import db_manager
from . import subtr_utils


def cube_scroller(stamps, titles=None, indices=None, ax=None):
    """
    Make a cube scroller that you can add to a figure
    """
    def update_image(img_ind):
        fig = ax.get_figure()
        img = stamps[img_ind]
        if indices == None:
            x = np.arange(0, img.shape[1]+1)
            y = np.arange(0, img.shape[0]+1)
        imax = ax.pcolor(x, y, img)#, **imshow_args, norm=norm_func(*norm_args))
        ax.set_aspect("equal")
        fig.colorbar(imax, ax=ax, shrink=0.75)
        ax.set_title(title)
    slider = widgets.IntSlider(min=0, max=len(stamps)-1, step=1, value=0,
                               description='stamp index')

    interactive_plot = interactive(update_image, img_ind=slider)#, fig=fixed(fig), ax=fixed(ax))
    output = interactive_plot.children[-1]
    width = '350px'
    height = '350px'
    output.layout.width = width
    output.layout.height = height
    return interactive_plot

def dashboard(stamps_dict):
    """
    Show the dashboard

    Parameters
    ----------
    tbd

    Output
    ------
    fig : plt.Figure instance
      if ax is given, returns a reference to the parent figure
    """
    nrows = 1
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))

    cube_scroller(stamps_dict['r'].values, titles=stamps_dict['r'].index, ax=axes[0])
    cube_scroller(stamps_dict['m'].values, titles=stamps_dict['m'].index, ax=axes[1])

    return fig

