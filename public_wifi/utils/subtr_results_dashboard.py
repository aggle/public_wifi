"""
Result visualization dashboard. Uses Bokeh to draw interactive plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd

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

from . import shared_utils
from . import table_utils
from .. import db_manager
from .. import subtr_manager



def load_composite_img_from_config(
        config_file : str ,
) -> np.array :
    """
    Load the composite image for display
    """
    img_path = shared_utils.load_config_path("tables",
                                             "composite_img_file",
                                             config_file)
    mast_img = fits.getdata(img_path)
    mast_img[mast_img<=0] = np.nan
    return mast_img


def show_sky_scene(config_file, star_id, dbm, zoom=61, alt_dbm={}, stamp_size=11, plot_size=300,
                   show_sky=False):
    """
    Plot the scene on sky

    Parameters
    ----------
    config_file : config file that has the path to the mast image
    star_id : the target star
    dbm : database manager for the active selection of stars, that includes the target star
    zoom : [61] size of the initial region to zoom in on
    alt_dbm : [{}] alternate star table(s), will plot stars not in main database
    stamp_size : [11] size of a box to draw around the star that indicates the stamp size
    plot_size : [300] size of the plot (x and y)

    Output
    ------
    Bokeh Figure object containing the plot

    """
    TOOLS = "pan,wheel_zoom,box_zoom,reset"

    # Sky scene
    # set initial range to zoom in around the target star
    zoom = np.floor(zoom/2).astype(int)
    star_pos = dbm.stars_tab.set_index('star_id').loc[star_id][['u_mast', 'v_mast']].astype(float)
    x_range = star_pos['u_mast'] + np.array([-zoom, zoom])
    y_range = star_pos['v_mast'] + np.array([-zoom, zoom])

    # initialize the sky scene figure
    p_sky = bkplt.figure(tools=TOOLS,
                         min_height=plot_size, min_width=plot_size,
                         sizing_mode='scale_both',
                         x_range=tuple(x_range), y_range=tuple(y_range),
                         title=f"{star_id} context on sky")

    # draw the sky scene
    if show_sky == True:
        mast_img = load_composite_img_from_config(config_file)
        mapper = bkmdls.LogColorMapper(palette='Magma256', low=np.nanmin(mast_img), high=np.nanmax(mast_img))
        p_sky.image(image=[mast_img], 
                    x=-0.5, y=-0.5, dw=mast_img.shape[1], dh=mast_img.shape[0], 
                    color_mapper=mapper)

    # draw the stamp limits
    stamp_width = np.floor(stamp_size/2).astype(int)
    box = star_pos.apply(lambda x: np.floor(x) + np.array([-stamp_width, stamp_width+1]))
    #stamp_width*np.array([-1, 1]) + np.array([0, 1]))
    p_sky.quad(left=box['u_mast'][0], right=box['u_mast'][1],
               bottom=box['v_mast'][0], top=box['v_mast'][1],
               alpha=0.5, line_color='white', fill_alpha=0, line_width=2)

    # scatter plot for the star and its the neighbors
    star_plot = p_sky.star(x='u_mast', 
                           y='v_mast',
                           source=dbm.stars_tab.query('star_id == @star_id'),
                           size=20, fill_alpha=0.2)


    # first, show the neighbors *in* the catalog
    nbr_plot = p_sky.circle(x='u_mast',
                 y='v_mast',
                 size=5, fill_alpha=0.2,
                 source=dbm.stars_tab.query("star_id != @star_id"),
                 legend_label='Selected stars')

    #if alt_dbm is not None:
    noncat_plots = []
    if isinstance(alt_dbm, dict):
        for alt_k, alt_d in alt_dbm.items():
            selected_stars = alt_d.stars_tab['star_id']
            noncat_plots.append(
                p_sky.x(x='u_mast',
                        y='v_mast',
                        size=5, fill_alpha=0.2,
                        color='gray',
                        source=alt_d.stars_tab.query("star_id not in @selected_stars"),
                        legend_label=alt_k)
            )
    else:
        print("alt_dbm argument changed: now supply a dictionary of kw-db pairs")

    # hover tool
    hover_tool = bkmdls.HoverTool(renderers=[star_plot, nbr_plot] + noncat_plots)
    hover_tool.tooltips=[("star_id", "@star_id"),
                         ("mag_F1", "@star_mag_F1{0.2f}"),
                         ("mag_F2", "@star_mag_F2{0.2f}"),
                         ]
    p_sky.add_tools(hover_tool)
    p_sky.toolbar.active_inspect = None

    return p_sky


def show_detector_scene(star_id, dbm, alt_dbm={}, plot_size=300):
    """
    Plot all the point sources on the detector sector

    Parameters
    ----------
    star_id : the target star
    dbm : db_manager object for the sector
    plot_size : [300] size of the plot (x and y)
    alt_dbm: [{}] alternate db_manager, will plot all point sources not in dbm

    Output
    ------
    Bokeh Figure object containing the plot

    """
    TOOLS = "pan,wheel_zoom,box_zoom,reset"
    p = figure(tools=TOOLS,
               min_height=plot_size, min_width=plot_size,
               title=f"{star_id} context on detector")

    # plot all the catalog detections in the sector
    exp_ids = list(dbm.ps_tab.groupby('ps_exp_id').groups.keys())
    catalog_plot = p.circle(x='ps_x_exp',
                            y='ps_y_exp',
                            source=dbm.ps_tab,
                            color=bktrans.factor_cmap('ps_exp_id',
                                                      # f"Category20_{len(exp_ids)}",
                                                      "Turbo256",
                                                      exp_ids),
                            size=10,
                            legend_label='Catalog detections')

    # Plot the location of the target star
    target_ps_ids = dbm.find_matching_id(star_id, 'P')
    target_locs = dbm.ps_tab.set_index('ps_id').loc[target_ps_ids, ['ps_x_exp', 'ps_y_exp']]
    target_plot = p.star(target_locs['ps_x_exp'].mean(),
                         target_locs['ps_y_exp'].mean(),
                         line_color='black', fill_alpha=0, size=20, line_width=2,
                         legend_label='Target star')

    # detections not in the catalog
    #if alt_dbm is not None:
    noncat_plots = []
    if isinstance(alt_dbm, dict):
        for alt_k, alt_d in alt_dbm.items():
            sector = dbm.find_sector(dbm.find_matching_id(star_id, 'p'))['sector_id'].unique()
            # first, find the point sources not in the main db
            cut_sec_ids = set(alt_d.find_sector(sector_id=sector)['ps_id'])
            cut_sec_ids = cut_sec_ids.difference(set(dbm.ps_tab['ps_id']))
            noncat_plots.append(
                p.x(x='ps_x_exp', y='ps_y_exp',
                    source=alt_d.ps_tab.query("ps_id in @cut_sec_ids"),
                    size=5,
                    color='gray',
                    legend_label="Cut detections")
            )
    else:
        print("alt_dbm argument changed: now supply a dictionary of kw-db pairs")

    # hover tool
    hover_tool = bkmdls.HoverTool(renderers=[catalog_plot] + noncat_plots)
    hover_tool.tooltips=[("star_id", "@ps_star_id"),
                         ("ps_id", "@ps_id")]
    p.add_tools(hover_tool)
    p.toolbar.active_inspect = None


    return p


def show_target_stamps(star_id, dbm, plot_size=300):
    """
    Docstring goes here

    Parameters
    ----------
    star_id : str
      star identifier
    dbm : db_manager instance
      database manager containing the star
    plot_size : int [300]
      size of the plot

    Output
    ------
    Bokeh figure that can be displayed with show()

    """
    #  target stamps
    target_stamp_ids = dbm.find_matching_id(star_id, 'T')
    target_stamp_ids = dbm.stamps_tab.query('stamp_id in @target_stamp_ids')['stamp_id']
    ncols = min(len(target_stamp_ids), 3)
    nrows = np.ceil(len(target_stamp_ids)/ncols).astype(int)
    stamps = dbm.stamps_tab.set_index("stamp_id").loc[target_stamp_ids, 'stamp_array']
    p_stamps = stamps.reset_index()['stamp_id'].apply(lambda x: figure(tools='', title=f'{x}'))
    mapper = bkmdls.LinearColorMapper(palette='Magma256',
                                   low=np.nanmin(np.stack(stamps)),
                                   high=np.nanmax(np.stack(stamps)))
    for p, target_stamp, stamp_id in zip(p_stamps, stamps, target_stamp_ids): 
        p.image(image=[target_stamp], 
                x=-0.5, y=-0.5, dw=target_stamp.shape[1], dh=target_stamp.shape[0], 
                color_mapper=mapper)
    #     color_bar = ColorBar(color_mapper=mapper, label_standoff=12)
    #     p.add_layout(color_bar, 'right')

    plot_size = min([int(plot_size/nrows),int(plot_size/ncols)])
    grid = bklyts.gridplot(list(p_stamps), ncols=ncols,
                           sizing_mode='scale_both',
                           height=plot_size,#int(plot_size/nrows),
                           width=plot_size,#int(plot_size/ncols),
                           merge_tools=True, toolbar_location=None)
    return grid


def cube_scroller_plot_slider(cube, title, scroll_title='',
                              cmap_class=bkmdls.LinearColorMapper,
                              info_col : pd.Series | None = None,
                              plot_size=400):
    """
    This is called by cube_scroller_app()
    Generate the plot and scroller objects for a cube_scroller application.
    Separating it out this way makes it easier to add multiple cube scrollers
    to the same figure.

    Parameters
    ----------
    cube : pd.Series
      pandas.Series object whose entries are arrays
    title : str
      title to put on the plot
    scroll_title : str ['']
      title for the scroll bar
    cmap_class : bokeh.models.ColorMapper class [bkmdls.LinearColorMapper]
      a color mapper for scaling the images
    info_col : pd.Series | None
      extra info to add to plot. Must have same index as `cube`
    Output
    ------
    bokeh figure, slider widget, and the that stores the data
    """
    TOOLS='save'

    if not isinstance(cube, pd.Series):
        cube = pd.Series({i: j for i, j in enumerate(cube)})

    data = cube[cube.map(lambda el: ~np.isnan(el).all())].copy()
    # initialize image
    img = data[data.index[0]]
    cds = bkmdls.ColumnDataSource(data={'image':[img],
                                        'x': [-0.5], 'y': [-0.5],
                                        'dw': [img.shape[0]], 'dh': [img.shape[1]]
                                        }
                                  )
    low, high = np.nanmin(img), np.nanmax(img)
    color_mapper = cmap_class(palette='Magma256',
                              low=low,
                              high=high)

    plot = figure(title=f"{title}",
                  min_height=plot_size, min_width=plot_size,
                  # aspect_ratio=1,
                  tools=TOOLS)
    g = plot.image(image='image',
                   x='x', y='y',
                   dw='dw', dh='dh',
                   color_mapper=color_mapper,
                   source=cds)
    # Hover tool
    hover_tool = bkmdls.HoverTool()
    hover_tool.tooltips=[("value", "@image"),
                         ("(x,y)", "($x{0}, $y{0})")]
    plot.add_tools(hover_tool)
    plot.toolbar.active_inspect = None

    # color bar
    color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
    plot.add_layout(color_bar, 'right')

    # slider
    # lambda function to generate the extra information string
    info_str = lambda index: '' if info_col is None else f"{info_col.name} = {info_col.iloc[index]}"
    slider_title = lambda title, index, info='': f"{title} :: {index} / {info}"
    slider = bkmdls.Slider(start=0, end=data.index.size-1, value=0, step=1,
                           title=slider_title(scroll_title, data.index[0], info_str(0)),
                           show_value = False,
                           # default_size=plot_size,
                           orientation='horizontal')
    def callback(attr, old, new):
        # update the image
        img = data[data.index[new]]
        cds.data['image'] = [img]
        color_mapper.update(low=np.nanmin(img), high=np.nanmax(img))
        slider.title = slider_title(scroll_title, data.index[new], info_str(new))
    slider.on_change('value', callback)

    # Switch color map
    menu = {"Linear": bkmdls.LinearColorMapper, "Log": bkmdls.LogColorMapper}
    cmap_switcher = bkmdls.Select(title='Switch color map',
                                  value=sorted(menu.keys())[0],
                                  width=plot_size,
                                  options=sorted(menu.keys()))
    def cmap_switcher_callback(attr, old, new):
        cmap_class = menu[new]
        color_mapper = cmap_class(palette='Magma256',
                                  low=np.nanmin(img),
                                  high=np.nanmax(img))
        # update the color mapper on image
        g.glyph.color_mapper=color_mapper
    cmap_switcher.on_change('value', cmap_switcher_callback)

    widgets = bklyts.column(slider, cmap_switcher, sizing_mode='scale_width')
    return plot, widgets, cds

def cube_scroller_app(
        cube : pd.Series,
        title : str = '',
        scroll_title : str = '',
        cmap_class : bkmdls.mappers.ColorMapper = bkmdls.LinearColorMapper, 
        info_col : pd.Series | None = None,
):
    """
    Make an app to scroll through a datacube. Returns the app; display using bokeh.io.show

    Parameters
    ----------
    cube : pd.Series
      pandas.Series object whose entries are stamp arrays
      The scroll bar will show the index as the label
    title : str
      title to put on the plot
    scroll_title : str ['']
      title for the scroll bar
    cmap_class : bokeh.models.ColorMapper class [bkmdls.LinearColorMapper]
      a color mapper for scaling the images
    info_col : pd.Series | None:
      a named series of extra information to display. Must have same index as cube

    Output
    ------
    bokeh application

    """
    def app(doc):
        plot, widgets, cds = cube_scroller_plot_slider(
            cube,
            title, scroll_title,
            cmap_class,
            info_col
        )

        doc.add_root(bklyts.column(plot, widgets))

        doc.theme = Theme(json=yaml.load("""
            attrs:
                figure:
                    background_fill_color: white
                    outline_line_color: white
                    toolbar_location: above
                    height: 500
                    width: 800
                Grid:
                    grid_line_dash: [6, 4]
                    grid_line_color: white
        """, Loader=yaml.FullLoader))
    return app


def df_scroller_app(df, title, scroll_title='', cmap_class=bkmdls.LinearColorMapper, plot_size=400):
    """
    Make an app to scroll through a dataframe. Assume the cubes are stored column-wise. 
    Select the cube (column) to scroll though using a select widget.
    Returns the app; display using bokeh.io.show

    Parameters
    ----------
    df : pd.DataFrame
      pandas.DataFrame object whose entries are arrays
    title : str
      title to put on the plot
    scroll_title : str ['']
      title for the scroll bar
    cmap_class : bokeh.models.ColorMapper class [bkmdls.LinearColorMapper]
      a color mapper for scaling the images

    Output
    ------
    bokeh application

    """
    def app(doc):
        TOOLS='' # no tools, for now

        data = df.copy()

        # initialize image
        column = data.columns[0]
        img = data.loc[data.index[0], column]
        cds = bkmdls.ColumnDataSource(data={'image':[img],
                                                'x': [-0.5], 'y': [-0.5],
                                                'dw': [img.shape[0]], 'dh': [img.shape[1]]})
        color_mapper = cmap_class(palette='Magma256',
                                  low=np.nanmin(img),
                                  high=np.nanmax(img))

        title_string = lambda title, column: f"{title} :: {column}"

        plot = figure(title=title_string(title, column),
                      min_height=plot_size, min_width=plot_size,
                      tools=TOOLS)
        g = plot.image(image='image',
                       x='x', y='y', dw='dw', dh='dh',
                       color_mapper=color_mapper,
                       source=cds)
        # Hover tool
        hover_tool = bkmdls.HoverTool()
        hover_tool.tooltips=[("value", "@image"),
                             ("(x,y)", "($x{0}, $y{0})")]
        plot.add_tools(hover_tool)
        plot.toolbar.active_inspect = None

        # color bar
        color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
        plot.add_layout(color_bar, 'right')

        # Interactions
        # PSF subtraction component slider
        slider_title = lambda title, index: f"{title} :{index}"
        slider = bkmdls.Slider(start=0, end=data.index.size-1, value=0, step=1,
                               title=slider_title(scroll_title, data.index[0]),
                               show_value=False,
                               orientation='horizontal')
        def slider_callback(attr, old, new):
            img = data[column][data.index[new]]
            cds.data = {'image':[img],
                        'x': [-0.5], 'y': [-0.5],
                        'dw': [img.shape[0]], 'dh': [img.shape[1]]}
            color_mapper.update(low=np.nanmin(img), high=np.nanmax(img))
            slider.title = slider_title(scroll_title, data.index[new])
        slider.on_change('value', slider_callback)

        # Target stamp selector
        selector = bkmdls.Select(title='Target stamp', 
                                     value=data.columns[0],
                                     options=[str(i) for i in data.columns])
        def select_callback(attr, old, new):
            column = new
            print(old, new, column)
            img = data.loc[data.index[slider.value], column]
            cds.data = {'image':[img],
                        'x': [-0.5], 'y': [-0.5],
                        'dw': [img.shape[0]], 'dh': [img.shape[1]]}
            color_mapper.update(low=np.nanmin(img), high=np.nanmax(img))
            plot.title.text = title_string(title, column)
        selector.on_change('value', select_callback)

        doc.add_root(bklyts.row(selector, bklyts.column(plot, slider)))

        doc.theme = Theme(json=yaml.load("""
            attrs:
                figure:
                    background_fill_color: white
                    outline_line_color: white
                    toolbar_location: above
                    height: 500
                    width: 800
                Grid:
                    grid_line_dash: [6, 4]
                    grid_line_color: white
        """, Loader=yaml.FullLoader))
    return app


def generate_inspector(
        config_file : str,
        ana_mgr,
        star_id : str, 
        alt_dbs : dict = {},
        plot_size_unit : int = 100
):
    """
    Create a Bokeh server app using the provided information.

    Parameters
    ----------
    config_file : path to the config file used in this analysis
    ana_mgr : an ana_utils.AnaManager instance
      contains the database, subtraction manager, and other analysis products
    star_id : star to show
    alt_dbs : [{}] dictionary of databases
      other databases to show on plots. Keys will be used for legend labels
    plot_size : [100] size of the plot increments
    Output
    ------
    bokeh app for use with show(app)

    """
    # do some cleanup for better filtering
    for k in ['snr', 'residuals', 'models']:
        # if an array is all nan, just set the entry to a single NaN value
        # that way you can drop it when assigning it to the cube scroller
        bool_df = ana_mgr.results_stamps[k].map(lambda x: np.isnan(x).all())
        ana_mgr.results_stamps[k].values[bool_df] = np.nan

    def app(doc):
        # sky scene, detector scene, and the target star's stamps
        plot_size = 5*plot_size_unit
        # plot to show the mosaic on-sky of all the sources
        p_sky = show_sky_scene(config_file, star_id, dbm=ana_mgr.db, alt_dbm=alt_dbs,
                               plot_size=plot_size)
        # plot to show where each star lands on the detector
        p_det = show_detector_scene(star_id, dbm=ana_mgr.db, alt_dbm=alt_dbs,
                                    plot_size=plot_size)
        # plot to show the different target stamps
        p_trg = show_target_stamps(star_id, dbm=ana_mgr.db,
                                   plot_size=plot_size)

        # reference stamps used to assemble the model PSF
        reference_ids = ana_mgr.results_stamps['references'].loc[star_id].dropna(how='all', axis=1)
        reference_ids = reference_ids.drop_duplicates().values.ravel()
        reference_stamps = ana_mgr.db.stamps_tab.set_index('stamp_id').loc[reference_ids, 'stamp_array']
        refs_plot, refs_slider, refs_cds = cube_scroller_plot_slider(
            reference_stamps.dropna().sort_index(),
            'Reference stamps',
            cmap_class=bkmdls.LinearColorMapper,
            plot_size=5*plot_size_unit
        )
        refs_col = bklyts.column(refs_plot, refs_slider)

        # load the widgets that depend on the stamp selection
        target_stamp_ids = list(ana_mgr.db.find_matching_id(star_id, 'T'))
        target_stamp_id = column = target_stamp_ids[0]
        
        # initialize stamp dataframes, putting the stamp IDs in the columns
        scroller_keys =  ['snr', 'residuals', 'models']
        stamp_dict = {k: ana_mgr.results_stamps[k].loc[star_id].dropna(how='all', axis=1).T
                      for k in scroller_keys}

        # shift the model PSFs so the min value is 0, so you can represent them in log scale
        stamp_dict['models'] = stamp_dict['models'] - stamp_dict['models'].apply(lambda col: col.apply(np.nanmin)).min()

        # initialize the images
        img_dict = {k: df.loc[df.index[0], column]
                    for k, df in stamp_dict.items()}

        # put them into ColumnDataSources
        cds_dict = {k: bkmdls.ColumnDataSource(data={'image':[img],
                                                     'x': [-0.5], 'y': [-0.5],
                                                     'dw': [img.shape[0]], 'dh': [img.shape[1]]})
                    for k, img in img_dict.items()}
        # initialize the color mappers
        color_mapper_dict = {
            'snr':  bkmdls.LinearColorMapper(palette='Magma256',
                                             low=np.nanmin(img_dict['snr']),
                                             high=np.nanmax(img_dict['snr'])),
            'residuals':  bkmdls.LinearColorMapper(palette='Magma256',
                                                   low=np.nanmin(img_dict['residuals']),
                                                   high=np.nanmax(img_dict['residuals'])),
            'models':  bkmdls.LogColorMapper(palette='Magma256',
                                             low=np.nanmin(img_dict['models']),
                                             high=np.nanmax(img_dict['models'])),
        }

        title_string = lambda title, column: f"{title} :: {column}"
        # initialize the actual plots
        TOOLS=''
        plot_dict = {k: figure(title=title_string(k.upper(), column),
                               min_height=plot_size, min_width=plot_size,
                               tools=TOOLS)
                     for k in stamp_dict.keys()}
        for k, plot in plot_dict.items():
            plot.image(image='image',
                       x='x', y='y', dw='dw', dh='dh',
                       color_mapper=color_mapper_dict[k],
                       source=cds_dict[k])
            # Hover tool
            hover_tool = bkmdls.HoverTool()
            hover_tool.tooltips=[("value", "@image"),
                                 ("(x,y)", "($x{0}, $y{0})")]
            plot.add_tools(hover_tool)
            plot.toolbar.active_inspect = None
            # color bar
            color_bar = bkmdls.ColorBar(color_mapper=color_mapper_dict[k],
                                        label_standoff=12)
            plot.add_layout(color_bar, 'right')

        # make the sliders
        slider_title = lambda title, index: f"{title} :{index}"
        slider_dict = {k: bkmdls.Slider(start=0, end=stamps.index.size-1, value=0, step=1,
                                        title=slider_title('N_components', stamps.index[0]),
                                        show_value=False,
                                        # default_size=plot_size,
                                        orientation='horizontal')
                       for k, stamps in stamp_dict.items()}
        def make_slider_callback(key): # generator for slider callback functions
            def slider_callback(attr, old, new):
                stamps = stamp_dict[key]
                img = stamps.loc[stamps.index[new], column]
                cds_dict[key].data = {'image':[img],
                                      'x': [-0.5], 'y': [-0.5],
                                      'dw': [img.shape[0]], 'dh': [img.shape[1]]}
                color_mapper_dict[key].update(low=np.nanmin(img), high=np.nanmax(img))
                slider_dict[key].title = slider_title("N_components", stamps.index[new])
            return slider_callback
        slider_callback_dict = {k: make_slider_callback(k) for k in stamp_dict.keys()}
        for k, slider in slider_dict.items():
            slider.on_change('value', slider_callback_dict[k])

        # combine them for the layout
        scroller_columns = {k: bklyts.column(plot_dict[k], slider_dict[k])
                            for k in scroller_keys}


        # make the target stamp selector
        def select_callback(attr, old, new):
            column = new
            for k in scroller_keys:
                stamp_col = stamp_dict[k]
                index = stamp_dict[k].index[slider_dict[k].value]
                img_dict[k] = stamp_dict[k].loc[index, column]
                img = img_dict[k]
                cds_dict[k].data = {'image':[img],
                                    'x': [-0.5], 'y': [-0.5],
                                    'dw': [img.shape[0]], 'dh': [img.shape[1]]}
                color_mapper_dict[k].update(low=np.nanmin(img), high=np.nanmax(img))
                plot_dict[k].title.text = title_string(k.upper(), column)
        stamp_selector = bkmdls.Select(title='Target stamp',
                                       value=column,
                                       width=plot_size,
                                       options=[str(i) for i in target_stamp_ids])
        stamp_selector.on_change('value', select_callback)

        # layout
        lyt = bklyts.column(
            bklyts.row(p_sky, p_det, p_trg, refs_col),
            bklyts.row(stamp_selector,
                       scroller_columns['snr'],
                       scroller_columns['residuals'],
                       scroller_columns['models'])
        )

        doc.add_root(lyt)

        doc.theme = Theme(json=yaml.load("""
            attrs:
                figure:
                    background_fill_color: white
                    outline_line_color: white
                    toolbar_location: above
                    height: 500
                    width: 800
                Grid:
                    grid_line_dash: [6, 4]
                    grid_line_color: white
        """, Loader=yaml.FullLoader))
        #doc.theme = "night_sky"


    return app


# for backwards compatibility
generate_inspector_ana = generate_inspector
