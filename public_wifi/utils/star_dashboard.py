"""
Result visualization dashboard. Uses Bokeh to draw interactive plots.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import functools
 
import yaml
import bokeh
import bokeh.transform as bktrans
import bokeh.layouts as bklyts
import bokeh.plotting as bkplt
import bokeh.io as bkio
import bokeh.models as bkmdls

#from bokeh.models import ColumnDataSource, Slider, ColorBar, LogColorMapper
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.themes import built_in_themes
from bokeh.io import show, output_notebook, output_file

from astropy.io import fits
from public_wifi.utils.detection_utils import load_snr_maps

from public_wifi import starclass as sc

def standalone(func):
    """
    Take a method that returns a bokeh layout and return it as an app that can be displayed with bkplt.show()
    """
    @functools.wraps(func)
    def appwrap(*args, **kwargs):
        # wrap the layout-producing func in a bokeh app
        def app(doc):
            layout = func(*args, **kwargs)

            doc.add_root(layout)

            doc.theme = Theme(json=yaml.load("""
                attrs:
                    figure:
                        background_fill_color: white
                        outline_line_color: white
                        toolbar_location: above
                    Grid:
                        grid_line_dash: [6, 4]
                        grid_line_color: white
            """, Loader=yaml.FullLoader))
        return app
    return appwrap


def generate_cube_scroller_widgets(
        cube : pd.Series,
        cds : bkmdls.ColumnDataSource = None,
        slider_index : list | np.ndarray = None,
        plot_kwargs={},
        slider_kwargs={},
        cmap_class : bokeh.core.has_props.MetaHasProps = bkmdls.LinearColorMapper,
        add_log_scale : bool = False,
        diverging_cmap = False,
):
    """
    Generate a plot and a scroller to scroll through the cube stored in a provided ColumnDataSource
    
    Parameters
    ----------
    cube : pd.Series
      data cube to plot
    slider_index : list-like
      text to display when you move the slider. must be same length as cube
      defaults to cube.index
    plot_kwargs : dict
      kwargs to pass to bkplt.figure
    slider_kwargs : dict
      kwargs to pass to bkmdls.Slider
    add_log_scale : bool = False
      Add a switch to change the color mapping between linear and log scaling
    diverging_cmap : bool = False
      If True, use a diverging Blue-Red colormap that is white in the middle
    """
    # initialize the CDS. The plot looks inside here for what to plot
    # so when you change the slider, update these values
    if slider_index is None:
        slider_index = cube.index.copy()
    cds = bkmdls.ColumnDataSource(
        data={
            'img': [cube.iloc[0]],
            'index': [slider_index[0]],
            'x': [-0.5],
            'y': [-0.5],
            'dw': [cube.iloc[0].shape[-1]],
            'dh': [cube.iloc[0].shape[-2]],
            'len': [len(cube)],
        },
        tags = [cube.name, cube.index.name]
    )
    # use this to store the plot elements

    TOOLS='save'

    plot_kwargs['match_aspect'] = plot_kwargs.get('match_aspect', True)
    plot_kwargs['title'] = plot_kwargs.get('title', cds.tags[0])

    plot = bkplt.figure(
        tools=TOOLS,
        **plot_kwargs
    )
    plot.x_range.range_padding = plot.y_range.range_padding = 0
    if diverging_cmap == False:
        palette = 'Magma256'
    else:
        # For PSF subtraction results, you want a diverging colormap centered
        # on 0. Blue is negative, red is positive
        palette = bokeh.palettes.diverging_palette(
            bokeh.palettes.brewer['Blues'][256],
            bokeh.palettes.brewer['Reds'][256],
            256
        )

    # function to conditionally compute the colormap range; used in the
    # callback function
    def compute_cmap_lims(img, diverging):
        if diverging == False:
            low=np.nanmin(img)
            high=np.nanmax(img)
        else:
            # make the limits symmetric
            high = np.max(np.absolute([np.nanmin(img), np.nanmax(img)]))
            low = -1*high
        return low, high
    low, high = compute_cmap_lims(cds.data['img'], diverging_cmap)
    color_mapper = cmap_class(
        palette=palette,
        low=low, high=high,
    )

    img_plot = plot.image(
        image='img',
        source=cds,
        x='x', y='y',
        dw='dw', dh='dh',
        color_mapper=color_mapper,
    )
    # add a color bar to the plot
    color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
    plot.add_layout(color_bar, 'right')

    # add hover tool
    hover_tool = bkmdls.HoverTool()
    hover_tool.tooltips=[("value", "@curimg"),
                         ("(x,y)", "($x{0}, $y{0})")]
    plot.add_tools(hover_tool)
    plot.toolbar.active_inspect = None
    
    # add a slider to update the image and color bar when it moves
    slider_title_prefix = slider_kwargs.get("title", cds.tags[1])
    slider_title = lambda val: f"{slider_title_prefix}: {val}"
    slider_kwargs['title'] = slider_title(slider_index[0])
    def slider_change(attr, old, new):
        cds.data.update({
                    'img': [cube.iloc[new]],
                    'index': [slider_index[new]],
                })
        low, high = compute_cmap_lims(cds.data['img'][0], diverging_cmap)
        color_mapper.update(low=low, high=high)
        slider.update(title = slider_title(cds.data['index'][0]))

    slider_kwargs['show_value'] = slider_kwargs.get('show_value', False)
    slider = bkmdls.Slider(
        start=0, end=cube.index.size-1, value=0, step=1,
        **slider_kwargs
    )
    slider.on_change('value', slider_change)
    
    # if requested, add a switch to change the color scale to log
    if add_log_scale == True:
        menu = {"Linear": bkmdls.LinearColorMapper, "Log": bkmdls.LogColorMapper}
        cmap_switcher = bkmdls.Select(title='Switch color map',
                                      value=sorted(menu.keys())[0],
                                      options=sorted(menu.keys()))
        def cmap_switcher_callback(attr, old, new):
            cmap_class = menu[new]
            color_mapper = cmap_class(palette=palette,
                                      low=np.nanmin(cds.data['curimg']),
                                      high=np.nanmax(cds.data['curimg']))
            # update the color mapper on image
            img_plot.glyph.update(color_mapper=color_mapper)
        cmap_switcher.on_change('value', cmap_switcher_callback)
        # define the layout
        layout = bklyts.column(plot, bklyts.row(slider, cmap_switcher))
    else:
        layout = bklyts.column(plot, slider)
    return layout

# standalone cube scroller
make_cube_scroller = standalone(generate_cube_scroller_widgets)


def make_static_img_plot(
        img : np.ndarray | bkmdls.ColumnDataSource,
        width = 400,
        height = 400,
        # dw = 10,
        # dh = 10,
        title='',
        tools='',
        cmap_class : bokeh.core.has_props.MetaHasProps = bkmdls.LinearColorMapper,
        **kwargs,
):
    if not isinstance(img, bkmdls.ColumnDataSource):
        img = img_to_CDS(img)
    # update the kwargs with defaults
    # kwargs['palette'] = kwargs.get('palette', 'Magma256')
    kwargs['level']=kwargs.get("level", "image")
    p = bkplt.figure(
        width=width, height=height,
        title=title, tools=tools,
        match_aspect=True,
    )
    p.x_range.range_padding = p.y_range.range_padding = 0
    color_mapper = cmap_class(
        palette='Magma256',
        # low=np.nanmin(img.data['img'][0]), high=np.nanmax(img.data['img'][0]),
    )
    p.image(
        source=img,
        image='img',
        x='x', y='y',
        dw='dw', dh='dw',
        color_mapper=color_mapper,
        # palette = 'Magma256',
        # low='low', high='high',
        **kwargs
    )
    p.grid.grid_line_width = 0
    # add a color bar to the plot
    color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
    p.add_layout(color_bar, 'right')
    return p


def img_to_CDS(
        img : np.ndarray,
        cds : bkmdls.ColumnDataSource = None,
        properties : dict = {},
) -> None:
    # store the images in the CDS
    if cds is None:
        # if none provided, make one.
        cds = bkmdls.ColumnDataSource()
    # series metadata

    cds.data.update(img=[img])
    # add the new cube information to the properties update dictionary
    properties.update({
        'x': [-0.5], 'y': [-0.5],
        'dw': [img.shape[-1]], 'dh': [img.shape[-2]],
        'low': [np.nanmin(img)], 'high': [np.nanmax(img)],
    })
    cds.data.update(**properties)
    return cds

def series_to_CDS(
        cube : pd.Series,
        cds : bkmdls.ColumnDataSource = None,
        properties : dict = {},
) -> None:
    """
    When you get a new cube, you should update its corresponding CDS.
    This is a generic helper function to do that.
    You can also use this to provide an existing CDS with a new cube

    cube : pandas Series of the data to plot. Data are images.
    cds : bkmdls.ColumnDataSource | None
      a ColumnDataSource in which to store the data. Pass one in if you need it
      to be persistent, or leave as None to generate a new one.
    properties : dict = {}
      this argument is used to pass any other entries to the data field that you may want to add
    """
    # store the images in the CDS
    if cds == None:
        # if none provided, make one.
        cds = bkmdls.ColumnDataSource()
    # series metadata
    cds.tags = [cube.name, cube.index.name]

    cds.data.update(cube=[np.stack(cube.values)])
    cds.data.update(index=[list(cube.index.values)])
    cds.data.update(img=[cube.iloc[0]])
    cds.data.update(i=[cube.index[0]])
    cube_shape = np.stack(cube.values).shape
    # add the new cube information to the properties update dictionary
    properties.update({
        'x': [-0.5], 'y': [-0.5],
        'dw': [cube_shape[-1]], 'dh': [cube_shape[-2]],
    })
    cds.data.update(**properties)
    return cds
