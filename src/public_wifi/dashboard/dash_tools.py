"""
Generic dashboard-related utilities that are not specific to the dat\a
"""
import functools
import numpy as np
import pandas as pd

# bokeh-related imports
import yaml
import bokeh
import bokeh.layouts as bklyts
import bokeh.plotting as bkplt
import bokeh.models as bkmdls
#from bokeh.models import ColumnDataSource, Slider, ColorBar, LogColorMapper
from bokeh.plotting import figure
from bokeh.themes import Theme

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



### Helper functions for standardizing data input ###
def img_to_CDS(
        img : np.ndarray,
        cds : bkmdls.ColumnDataSource = None,
        properties : dict = {},
        cmap_range = (0.01, 0.99),
) -> None:
    """Make a CDS that can be used to display an image"""
    # store the images in the CDS
    if cds is None:
        # if none provided, make one.
        cds = bkmdls.ColumnDataSource()

    shape = img.shape
    center = np.floor(np.array(shape)/2).astype(int)[::-1]
    # low, high = np.nanquantile(img, cmap_range)
    data = dict(
        # these are the fields that the plots will inspect
        img=[img],
        # these fields help format the plots
        # lower left corner coordinates
        # x = [-0.5], y = [-0.5],
        x = [-center[0] - 0.5], y=[-center[1]-0.5],
        # image height and width in pixels
        dw = [img.shape[-1]], dh = [img.shape[-2]],
        # low = [np.nanmin(img)], high = [np.nanmax(img)],
        # low=[low], high=[high],
    )
    data.update(**properties)
    cds.data.update(**data)
    return cds


def series_to_CDS(
        cube : pd.Series,
        cds : bkmdls.ColumnDataSource = None,
        index : None | list | np.ndarray | pd.Series = None,
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
    index : [ None ] | list | np.ndarray | pd.Series
    properties : dict = {}
      this argument is used to pass any other entries to the data field that you may want to add
    """
    # store the images in the CDS
    if cds == None:
        # if none provided, make one.
        cds = bkmdls.ColumnDataSource()
    if index is None:
        index = list(cube.index.values.astype(str))
    # series metadata
    cds.tags = [cube.name, cube.index.name]

    cube_shape = np.stack(cube.values).shape
    center = np.floor(np.array(cube_shape[-2:])/2).astype(int)[::-1]
    data = dict(
        # these are the fields that the plots will inspect
        img=[cube.iloc[0]],
        i=[index[0]],
        # these fields store the available values
        cube=[np.stack(cube.values)],
        index=[index],
        # these fields help format the plots
        nimgs = [cube.shape[0]],
        # lower left corner coordinates
        x = [-center[0]-0.5], y = [-center[1]-0.5],
        # image height and width in pixels
        dw = [cube_shape[-1]], dh = [cube_shape[-2]],
    )
    data.update(**properties)
    cds.data.update(**data)
    return cds

series_cds_template = bkmdls.ColumnDataSource(
    data=dict(
        img = [], # the current image displayed
        i = [], # the index value of the current image
        cube = [], # all the images of the cube
        index = [], # the full index
        nimgs = [], # the length of the cube
        x = [], y = [], # lower left corner coordinate
        dw = [], dh = [], # image width and height in pixels
    )
)

def make_static_img_plot(
        img : np.ndarray | bkmdls.ColumnDataSource,
        size = 400,
        # dw = 10,
        # dh = 10,
        title='',
        tools='',
        cmap_class : bokeh.core.has_props.MetaHasProps = bkmdls.LinearColorMapper,
        **kwargs,
):
    """Plot an image that doesn't need to be scrolled through"""
    if not isinstance(img, bkmdls.ColumnDataSource):
        img = dt.img_to_CDS(img)
    # update the kwargs with defaults
    # kwargs['palette'] = kwargs.get('palette', 'Magma256')
    kwargs['level']=kwargs.get("level", "image")
    plot = bkplt.figure(
        width=size, height=size,
        title=title, tools=tools,
        match_aspect=True,
    )
    plot.x_range.range_padding = plot.y_range.range_padding = 0
    color_mapper = cmap_class(
        palette='Magma256',
        # low=np.nanmin(img.data['img'][0]), high=np.nanmax(img.data['img'][0]),
    )
    plot.image(
        source=img,
        image='img',
        x='x', y='y',
        dw='dw', dh='dw',
        color_mapper=color_mapper,
        # palette = 'Magma256',
        # low='low', high='high',
        **kwargs
    )
    plot.grid.grid_line_width = 0
    # add a color bar to the plot
    color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
    plot.add_layout(color_bar, 'right')

    # add hover tool
    hover_tool = bkmdls.HoverTool()
    hover_tool.tooltips=[("value", "@img"),
                         ("(x,y)", "($x{0}, $y{0})")]
    plot.add_tools(hover_tool)
    plot.toolbar.active_inspect = hover_tool

    return plot


def generate_cube_scroller_widget(
        source,
        plot_kwargs={},
        use_diverging_cmap : bool = False,
        cmap_range : tuple[float, float] = (0.01, 0.99),
):
    """
    Generate a contained layout object to scroll through a cube. Can be added to other documents.
    
    Parameters
    ----------
    source : bkmdls.ColumnDataSource
      a CDS containing the data to plot. Generate with series_to_CDS(pd.Series)
    plot_kwargs : dict
      kwargs to pass to bkplt.figure
    add_log_scale : bool = False
      Add a switch to change the color mapping between linear and log scaling
    diverging_cmap : bool = False
      If True, use a diverging Blue-Red colormap that is white in the middle
      This is useful for residual plots that are mean 0
    cmap_range : tuple[float] = (0.1, 0.9)
      color range
      
    """
    TOOLS = "box_select,pan,reset"

    palette = 'Magma256'
    if use_diverging_cmap:
        palette = bokeh.palettes.diverging_palette(
            bokeh.palettes.brewer['Greys'][256],
            bokeh.palettes.brewer['Reds'][256],
            256
        )

    # plot configuration and defaults
    # plot_kwargs['match_aspect'] = plot_kwargs.get('match_aspect', True)
    for k, v in dict(height=400, width=400, match_aspect=True).items():
        plot_kwargs[k] = plot_kwargs.get(k, v)
    plot = figure(
        tools=TOOLS,
        **plot_kwargs,
        name='plot',
    )

    # set the color bar
    # low, high = np.nanquantile(source.data['img'][0], cmap_range)
    color_mapper = bkmdls.LinearColorMapper(
        palette=palette,
        # low=low, high=high,
    )

    # Stamp image
    img_plot = plot.image(
        image='img',
        source=source,
        # x=-0.5, y=-0.5,
        # dw=source.data['img'][0].shape[-1], dh=source.data['img'][0].shape[-2],
        x='x', y='y',
        dw='dw', dh='dh',
        # palette=palette,
        color_mapper=color_mapper,
    )
    # plot crosshairs across the origin
    line_style = dict(line_width=0.5, line_color="white", line_dash='dashed')
    plot.hspan(y=[0], **line_style)
    plot.vspan(x=[0], **line_style)
    # Add a color bar to the image
    color_bar = bkmdls.ColorBar(
        color_mapper=color_mapper,
        height=int(0.9*plot_kwargs['height']),
        label_standoff=12
    )
    plot.add_layout(color_bar, 'right')

    # add hover tool
    hover_tool = bkmdls.HoverTool()
    hover_tool.tooltips=[("value", "@img"),
                         ("(x,y)", "($x{0}, $y{0})")]
    plot.add_tools(hover_tool)
    plot.toolbar.active_inspect = hover_tool

    # Slider
    slider = bkmdls.Slider(
        start=1, end=source.data['nimgs'][0],
        value=1, step=1,
        title = f"{len(source.data['index'][0])} / {source.data['i'][0]}",
        name='slider',
    )
    def slider_change(attr, old, new):
        # update the current index, used for the slider title
        source.data['i'] = [source.data['index'][0][new-1]]
        # update which slice of the data
        source.data['img'] = [source.data['cube'][0][new-1]]
        # update the slider title
        slider.update(
            title=f"{len(source.data['index'][0])} / {source.data['i'][0]}"
         )
        # low, high = np.nanquantile(source.data['img'][0], cmap_range)
        color_mapper.update(
            # low=np.nanmin(source.data['img'][0]),
            # high=np.nanmax(source.data['img'][0]),
            # low=low, high=high
        )
    slider.on_change('value', slider_change)

    

    layout = bklyts.column(plot, slider)
    return layout

# standalone cube scroller
make_cube_scroller = standalone(generate_cube_scroller_widget)
