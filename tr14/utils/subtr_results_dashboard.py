"""
Result visualization dashboard. Uses Bokeh to draw interactive plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import yaml
import bokeh
import bokeh.layouts as bklyts
import bokeh.plotting as bkplt
import bokeh.io as bkio
import bokeh.models as bkmdls

from bokeh.models import ColumnDataSource, Slider, ColorBar, LogColorMapper
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.io import show, output_notebook, output_file

from astropy.io import fits

from . import shared_utils
from . import table_utils
from . import db_manager
from . import subtr_utils

mast_img = fits.getdata(shared_utils.load_config_path("composite_img_file"))
mast_img[mast_img<=0] = np.nan

def show_sky_scene(star_id, dbm, zoom=61, alt_dbm=None, stamp_size=11, plot_size=300):
    """
    Plot the scene on sky

    Parameters
    ----------
    star_id : the target star
    dbm : database manager for the active selection of stars, that includes the target star
    zoom : [61] size of the initial region to zoom in on
    alt_dbm : [None] alternate star table, will plot all stars
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
                         plot_height=plot_size, plot_width=plot_size,
                         x_range=x_range, y_range=y_range, title=f"{star_id} context on sky")

    # draw the sky scene
    mapper = LogColorMapper(palette='Magma256', low=np.nanmin(mast_img), high=np.nanmax(mast_img))
    p_sky.image(image=[mast_img], 
                x=0.5, y=0.5, dw=mast_img.shape[1], dh=mast_img.shape[0], 
                color_mapper=mapper)
    p_sky.star(x=star_pos['u_mast'], 
               y=star_pos['v_mast'],
               size=20, fill_alpha=0.2)

    # draw the stamp limits
    stamp_width = np.floor(stamp_size/2).astype(int)
    box = star_pos.apply(lambda x: np.floor(x) + np.array([-stamp_width, stamp_width+1]))
    #stamp_width*np.array([-1, 1]) + np.array([0, 1]))
    p_sky.quad(left=box['u_mast'][0], right=box['u_mast'][1],
               bottom=box['v_mast'][0], top=box['v_mast'][1],
               alpha=0.5, line_color='white', fill_alpha=0, line_width=2)

    # show the neighbors
    # first, show the neighbors *in* the catalog
    neighbors = dbm.stars_tab.query("star_id != @star_id")
    p_sky.circle(x=neighbors['u_mast'],
                 y=neighbors['v_mast'],
                 size=5, fill_alpha=0.2,
                 legend_label='Selected stars')

    if alt_dbm is not None:
        selected_stars = dbm.stars_tab['star_id']
        neighbors = alt_dbm.stars_tab.query("star_id not in @selected_stars")
        p_sky.x(x=neighbors['u_mast'],
                y=neighbors['v_mast'],
                size=5, fill_alpha=0.2,
                color='gray',
                legend_label='Cut stars')

    return p_sky


def show_detector_scene(star_id, dbm, alt_dbm=None, plot_size=300):
    """
    Plot all the point sources on the detector sector

    Parameters
    ----------
    star_id : the target star
    dbm : db_manager object for the sector
    plot_size : [300] size of the plot (x and y)
    alt_dbm: [None] alternate db_manager, will plot all point sources not in dbm

    Output
    ------
    Bokeh Figure object containing the plot

    """
    TOOLS = "pan,wheel_zoom,box_zoom,reset"
    p = figure(tools=TOOLS,
               plot_height=plot_size, plot_width=plot_size,
               title=f"{star_id} context on detector")

    # plot all the catalog detections in the sector
    p.scatter(dbm.ps_tab['ps_x_exp'],
              dbm.ps_tab['ps_y_exp'],
              size=np.sqrt(dbm.ps_tab['ps_phot']),
              legend_label='Catalog detections')

    # Plot the location of the target star
    target_ps_ids = dbm.find_matching_id(star_id, 'P')
    target_locs = dbm.ps_tab.set_index('ps_id').loc[target_ps_ids, ['ps_x_exp', 'ps_y_exp']]
    p.star(target_locs['ps_x_exp'].mean(),
           target_locs['ps_y_exp'].mean(),
           line_color='black', fill_alpha=0, size=20, line_width=2,
           legend_label='Target star')

    # detections not in the catalog
    cut_sec_ids = set(alt_dbm.find_sector(sector_id=15)['ps_id']).difference(set(dbm.ps_tab['ps_id']))
    cut_star_pos = alt_dbm.join_all_tables().query("ps_id in @cut_sec_ids")
    cut_star_pos = cut_star_pos.groupby('star_id')[['ps_x_exp', 'ps_y_exp']].mean()
    p.scatter(cut_star_pos['ps_x_exp'], cut_star_pos['ps_y_exp'],
              marker='x', size=5, color='gray',
              legend_label="Cut detections")
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
    nrows = np.ceil(len(target_stamp_ids)/ncols).astype(np.int)
    stamps = dbm.stamps_tab.set_index("stamp_id").loc[target_stamp_ids, 'stamp_array']
    #stamps = [table_utils.get_stamp_from_id(stamp_id) for stamp_id in target_stamp_ids]
    p_stamps = stamps.reset_index()['stamp_id'].apply(lambda x: figure(tools='', title=f'{x}'))
    mapper = bkmdls.LogColorMapper(palette='Magma256',
                                   low=np.nanmin(np.stack(stamps)),
                                   high=np.nanmax(np.stack(stamps)))
    for p, target_stamp, stamp_id in zip(p_stamps, stamps, target_stamp_ids): 
        p.image(image=[target_stamp], 
                x=-0.5, y=-0.5, dw=target_stamp.shape[1], dh=target_stamp.shape[0], 
                color_mapper=mapper)
    #     color_bar = ColorBar(color_mapper=mapper, label_standoff=12)
    #     p.add_layout(color_bar, 'right')

    grid = bklyts.gridplot(list(p_stamps), ncols=ncols,
                           #sizing_mode='scale_both',
                           plot_height=np.int(plot_size/nrows),
                           plot_width=np.int(plot_size/ncols),
                           merge_tools=True, toolbar_location=None)
    return grid


def cube_scroller_plot_slider(cube, title, scroll_title='',
                              cmap_class=bkmdls.LinearColorMapper,
                              plot_size=400):
    """
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

    Output
    ------
    bokeh application
    """
    TOOLS='' # no tools, for now

    data = cube.copy()
    #color_mapper = cmap_class(palette='Magma256',
    #                          low=np.nanmin(np.stack(data)),
    #                          high=np.nanmax(np.stack(data)))
    # initialize image
    img = data[data.index[0]]
    cds = bkmdls.ColumnDataSource(data={'image':[img],
                                 'x': [-0.5], 'y': [-0.5],
                                 'dw': [img.shape[0]], 'dh': [img.shape[1]]})
    color_mapper = cmap_class(palette='Magma256',
                              low=np.nanmin(img),
                              high=np.nanmax(img))

    plot = figure(title=f"{title}",
                  plot_height=plot_size, plot_width=plot_size,
                  tools=TOOLS)
    g = plot.image(image='image',
                   x='x', y='y', dw='dw', dh='dh',
                   color_mapper=color_mapper,
                   source=cds)
    # color bar
    color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
    plot.add_layout(color_bar, 'right')

    def callback(attr, old, new):
        img = data[data.index[new]]
        cds.data = {'image':[img],
                    'x': [-0.5], 'y': [-0.5],
                    'dw': [img.shape[0]], 'dh': [img.shape[1]]}
        color_mapper.update(low=np.nanmin(img), high=np.nanmax(img))
    slider = bkmdls.Slider(start=0, end=data.index.size-1, value=0, step=1,
                           title=scroll_title,
                           orientation='horizontal')
    slider.on_change('value', callback)
    return plot, slider

def cube_scroller_app(cube, title, scroll_title='', cmap_class=bkmdls.LinearColorMapper):
    """
    Make an app to scroll through a datacube. Returns the app; display using bokeh.io.show

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

    Output
    ------
    bokeh application

    """
    def app(doc):
        plot, slider = cube_scroller_plot_slider(cube, title, scroll_title, cmap_class)

        doc.add_root(bklyts.column(slider, plot))

        doc.theme = Theme(json=yaml.load("""
            attrs:
                Figure:
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

def generate_inspector(star_id,
                       dbm, alt_dbm, # for the sky and detector scene plots
                       #snr_stamps, residual_stamps, model_stamps, # for the cube scrollers
                       #reference_stamps=None, # the reference images
                       subtr_results_dict=None, # subtraction results object
                       snr_maps=None, # SNR maps
                       plot_size_unit=100):
    """
    Create a Bokeh server app using the provided information.

    Parameters
    ----------
    star_id : the target star
    dbm : db_manager object for the sector
    alt_dbm: [None] alternate db_manager, for plotting sources not selected
    snr_stamps : pd.Series of SNR stamps
    residual_stamps : pd.Series of PSF subtraction residuals
    model_stamps : pd.Series of PSF models
    reference_stamps : pd.Series of the reference stamps used to construct the PSF
    subtr_results_dict : [None] dict with keys 'references', 'residuals', 'models', and 'snr'
    plot_size : [100] size of the plot increments
    Output
    ------
    bokeh app for use with show(app)

    """
    # do some cleanup for better filtering
    for k in ['snr', 'residuals', 'models']:
        # if an array is all nan, just set the entry to a single NaN value
        bool_df = subtr_results_dict[k].applymap(lambda x: np.isnan(x).all())
        subtr_results_dict[k].values[bool_df] = np.nan

    def app(doc):

        # sky scene, detector scene, and the target star's stamps
        plot_size = 5*plot_size_unit
        p_sky = show_sky_scene(star_id, dbm=dbm, alt_dbm=alt_dbm,
                               plot_size=plot_size)
        p_det = show_detector_scene(star_id, dbm=dbm, alt_dbm=alt_dbm,
                                    plot_size=plot_size)
        p_trg = show_target_stamps(star_id, dbm=dbm,
                                   plot_size=plot_size)

        # reference stamps used to assemble the model PSF
        reference_ids = subtr_results_dict['references'].loc[star_id].dropna(how='all', axis=1).drop_duplicates()
        reference_ids = reference_ids.values.ravel()
        reference_stamps = dbm.stamps_tab.set_index('stamp_id').loc[reference_ids, 'stamp_array']
        refs_plot, refs_slider = cube_scroller_plot_slider(reference_stamps,
                                                           'Reference stamps',
                                                           cmap_class=bkmdls.LogColorMapper,
                                                           plot_size=5*plot_size_unit)
        refs_col = bklyts.column(refs_plot, refs_slider)

        # load the widgets that depend on the stamp selection
        # make a selector widget to choose the stamp to inspect
        target_stamp_ids = list(dbm.find_matching_id(star_id, 'T'))
        target_stamp_id = target_stamp_ids[0]
        select_target_stamp_id = bkmdls.Select(title='Target stamp',
                                               value=target_stamp_id,
                                               options=target_stamp_ids)

        # initialize stamps
        snr_stamps = subtr_results_dict['snr'].loc[star_id, target_stamp_id].dropna()
        residual_stamps = subtr_results_dict['residuals'].loc[star_id, target_stamp_id].dropna()
        model_stamps = subtr_results_dict['models'].loc[star_id, target_stamp_id].dropna()

        # create the update rule
        def update_target_stamp(attrname, old, new):
            target_stamp_id = select_target_stamp_id.value
            # pull the SNR, residual, and model stamps
            snr_stamps = subtr_results_dict['snr'].loc[star_id, target_stamp_id].dropna()
            residual_stamps = subtr_results_dict['residuals'].loc[star_id, target_stamp_id].dropna()
            model_stamps = subtr_results_dict['models'].loc[star_id, target_stamp_id].dropna()

        select_target_stamp_id.on_change('value', update_target_stamp)

        # make the SNR, residual, and PSF model sliders
        plot_size = 4*plot_size_unit
        snr_plot, snr_slider = cube_scroller_plot_slider(snr_stamps,
                                                         'SNR',
                                                         'Ncomponent index',
                                                         bkmdls.LinearColorMapper,
                                                         plot_size=plot_size)
        resid_plot, resid_slider = cube_scroller_plot_slider(residual_stamps,
                                                             'Residuals',
                                                             'Ncomponent index',
                                                             bkmdls.LinearColorMapper,
                                                             plot_size=plot_size)
        model_plot, model_slider = cube_scroller_plot_slider(model_stamps,
                                                             'PSF Model',
                                                             'Ncomponent index',
                                                             bkmdls.LogColorMapper,
                                                             plot_size=plot_size)

        snr_col = bklyts.column(snr_plot, snr_slider)
        resid_col = bklyts.column(resid_plot, resid_slider)
        model_col = bklyts.column(model_plot, model_slider)
        cube_row = bklyts.row(snr_col, resid_col, model_col) # all together now


        # generate the layout
        #lyt = bklyts.layout([
        #    [[p_sky, p_trg],
        #    [p_det, refs_col]],
        #    [[select_target_stamp_id,  output_target_stamp_id]],[cube_col],
        #])
        lyt = bklyts.column(
            bklyts.row(p_sky, p_det, p_trg, refs_col),
            bklyts.row(select_target_stamp_id, cube_row)
        )
        doc.add_root(lyt)

        doc.theme = Theme(json=yaml.load("""
            attrs:
                Figure:
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



