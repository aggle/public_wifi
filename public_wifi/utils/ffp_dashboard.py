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

from . import shared_utils
from . import table_utils
from .. import db_manager
from .. import subtr_manager

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
                        height: 500
                        width: 800
                    Grid:
                        grid_line_dash: [6, 4]
                        grid_line_color: white
            """, Loader=yaml.FullLoader))
        return app
    return appwrap


def generate_cube_scroller_widgets(
        cds : bkmdls.ColumnDataSource | pd.Series,
        cmap_class : bokeh.core.has_props.MetaHasProps = bkmdls.LinearColorMapper,
        slider_title_prefix : str = '',
        plot_kwargs={},
        slider_kwargs={},
        add_log_scale : bool = False,
        diverging_cmap = False,
):
    """
    Generate a plot and a scroller to scroll through the cube stored in a provided ColumnDataSource
    
    Parameters
    ----------
    plot_kwargs : dict
      kwargs to pass to bkplt.figure
    slider_kwargs : dict
      kwargs to pass to bkmdls.Slider
    add_log_scale : bool = False
      Add a switch to change the color mapping between linear and log scaling
    diverging_cmap : bool = False
      If True, use a diverging Blue-Red colormap that is white in the middle
    """
    if isinstance(cds, pd.Series):
        cds = series_to_CDS(cds)
    # use this to store the plot elements

    TOOLS='save'

    plot_kwargs['match_aspect'] = plot_kwargs.get('match_aspect', True)
    plot = figure(
        title=cds.data.get('title', [''])[0],
        tools=TOOLS,
        **plot_kwargs
    )

    if diverging_cmap == False:
        palette = 'Magma256'
    else:
        # For PSF subtraction results, you want a diverging colormap centered
        # on 0. Blue is negative, red is positive
        palette = bokeh.palettes.diverging_palette(bokeh.palettes.brewer['Blues'][256],
                                                   bokeh.palettes.brewer['Reds'][256],
                                                   256)

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
    low, high = compute_cmap_lims(cds.data['curimg'], diverging_cmap)
    color_mapper = cmap_class(
        palette=palette,
        low=low, high=high,
    )

    img_plot = plot.image(
        image='curimg',
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
    slider_title = lambda val: f"{slider_title_prefix}: {val}"
    def slider_change(attr, old, new):
        # update the 'image' entry to the desired image
        try:
            new_index = cds.data['index'][0][new]
        except IndexError:
            # if the index is out of range, do nothing
            return
        img = cds.data['cube'][0][new_index]
        cds.data['curimg'] = [img]
        # set the color scaling for the new image
        low, high = compute_cmap_lims(img, diverging_cmap)
        color_mapper.update(low=low, high=high)
        slider.update(title = slider_title(new_index))

    slider = bkmdls.Slider(
        start=0, end=len(cds.data['index'][0])-1, value=0, step=1,
        title = slider_title(cds.data['index'][0][0]),
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

##### FFP Dashboard - no mosaic! ####

def series_to_CDS(
        cube : pd.Series,
        cds : bkmdls.ColumnDataSource = None,
        properties : dict = {},
) -> None:
    """
    When you get a new cube, you should update its corresponding CDS.
    This is a generic helper function to do that.

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
    # cds.data.update({str(i): [j] for i, j in enumerate(cube)})
    cds.data.update(cube=[cube])
    cds.data.update(index=[list(cube.index)])
    # cds.data.update({f"label_{i}" : [j] for i, j in enumerate(cube.index)})
    cube_shape = np.stack(cube.values).shape
    # add the new cube information to the properties update dictionary
    properties.update({
        'curimg': [cube.iloc[0]],
        'x': [-0.5], 'y': [-0.5],
        'dw': [cube_shape[-1]], 'dh': [cube_shape[-2]],
        'len': [len(cube)],
    })
    cds.data.update(**properties)
    return cds

def load_new_references(
    ana_mgr, 
    star_id : str, 
    cds : bkmdls.ColumnDataSource,
):
    reference_ids = ana_mgr.results_stamps['references'].loc[star_id].dropna(how='all', axis=1)
    reference_ids = reference_ids.drop_duplicates().values.ravel()
    reference_stamps = ana_mgr.db.stamps_tab.set_index('stamp_id').loc[reference_ids, 'stamp_array']
    series_to_CDS(
        reference_stamps.sort_index(), #np.stack(reference_stamps.values),
        cds,
        properties={'title': ["Reference stamps"]}
    )
    return


def load_target_stamps(
    ana_mgr,
    star_id, 
    cds
) -> None:
    """
    Load the target stamps for a star
    """
    target_stamp_ids = ana_mgr.db.find_matching_id(star_id, 'T')
    target_stamps = ana_mgr.db.stamps_tab.set_index('stamp_id').loc[target_stamp_ids, 'stamp_array']
    series_to_CDS(
        target_stamps.sort_index(),
        cds,
        properties={'title': ['Target star stamps']}
    )
    return

def load_result_stamps(
    ana_mgr,
    kind : str,
    star_id : str,
    stamp_id : str,
    cds,
    title : str = '',
) -> None:
    """Load the SNR results for a stamp"""
    results = ana_mgr.results_stamps[kind].loc[(star_id, stamp_id)]
    results = results.drop(results.index[results.apply(lambda arr: np.isnan(arr).all())])
    series_to_CDS(
        results,
        cds,
        properties={'title': [title]}
    )
    return


def get_star_name_from_id(star_id, ana_mgr):
    """
    Get the catalog name of a star from the star_id
    """
    ps_table = ana_mgr.db.ps_tab.set_index("ps_star_id")
    star_name = str(ps_table.loc[star_id, 'ps_target'].unique().squeeze())
    return star_name

def get_star_id_from_name(star_name, ana_mgr):
    """
    Get the star_id of a star from the catalog name
    """
    ps_table = ana_mgr.db.ps_tab.set_index('ps_target')
    star_id = str(ps_table.loc[star_name, "ps_star_id"].unique().squeeze())
    return star_id

def update_star_table(table, star_id, ana_mgr):
    """
    Return a table of the point source rows for a star
    star_id : star identifier
    ana_mgr : analysis manager
    table : bkmdls.DataTable object
    """
    columns = [
        'ps_target', 'ps_star_id', 'ps_id', 'ps_exp_id', 'ps_filt_id',
        'ps_mag', 'ps_mag_e', 'ps_snr', 'ps_u_mast', 'ps_v_mast',
        'ps_x_exp', 'ps_y_exp'
    ]
    rows = ana_mgr.db.ps_tab[columns].query(f"ps_star_id == '{star_id}'").copy()
    cds = bkmdls.ColumnDataSource(rows)
    table.update(
        source=cds,
        columns=[bkmdls.TableColumn(field=i, title=i) for i in rows.columns]
    )
    return table

def dashboard(
        ana_mgr,
        plot_size_unit : int = 50,
):

    def app(doc):

        ### INITIALIZE DATA STRUCTURES ###

        # start with the most likely detection
        candidate_df = ana_mgr.results_stamps['detections']

        candidate_star_order = candidate_df['num_modes'].groupby("star_id").mean().sort_values(ascending=False).index
        candidate_ids = [' / '.join(i) for i in candidate_df.loc[candidate_star_order].index]
        candidate_init = candidate_ids[0]
        star_init, target_stamp_init = candidate_init.split(" / ")

        # the list of star IDs
        star_ids = sorted(ana_mgr.db.stars_tab['star_id'])
        star_catalog_names = [
            get_star_name_from_id(star_id, ana_mgr) for star_id in star_ids
        ]
        star_name_init = get_star_name_from_id(star_init, ana_mgr)

        # load the catalog rows for this star
        star_info_table = bkmdls.DataTable(width=1500, height=100)
        update_star_table(star_info_table, star_init, ana_mgr)

        # load this star's target stamps
        target_stamps_cds = bkmdls.ColumnDataSource()
        load_target_stamps(ana_mgr, star_init, target_stamps_cds)
        # target_stamp_init = target_stamps_cds.data['index'][0][0]

        # load the references stamps
        reference_stamps_cds = bkmdls.ColumnDataSource()
        load_new_references(ana_mgr, star_init, reference_stamps_cds)

        # load the PSF subtraction products
        # SNR map
        snrmap_cds = bkmdls.ColumnDataSource()
        load_result_stamps(ana_mgr, kind='snr', 
                           star_id=star_init, stamp_id=target_stamp_init,
                           cds=snrmap_cds, title='SNR map')
        # Residuals
        residuals_cds = bkmdls.ColumnDataSource()
        load_result_stamps(ana_mgr, kind='residuals', 
                           star_id=star_init, stamp_id=target_stamp_init,
                           cds=residuals_cds, title='Residuals')        # PSF models
        psfmodel_cds = bkmdls.ColumnDataSource()
        load_result_stamps(ana_mgr, kind='models', 
                           star_id=star_init, stamp_id=target_stamp_init,
                           cds=psfmodel_cds, title='PSF Model')        

        # Color-magnitude diagram
        cmd_cds = bkmdls.ColumnDataSource(
            data={'x': ana_mgr.db.stars_tab['star_mag_F1'] - ana_mgr.db.stars_tab['star_mag_F2'],
                  'y': ana_mgr.db.stars_tab['star_mag_F1'],
                  'star_id': ana_mgr.db.stars_tab['star_id']}
        )
        cmd_plot = figure()
        def update_cmd_plot(star_id):
            # this like empties the plot of previous data points
            cmd_plot.renderers = []
            # plot the CMD
            cmd_plot.scatter(x='x', y='y', source=cmd_cds)
            cmd_plot.y_range.flipped = True
            # filter the CDS for the star you want to highlight
            idx = cmd_cds.data['star_id'].index[cmd_cds.data['star_id'] == star_id]
            view = bkmdls.CDSView(filter=bkmdls.IndexFilter(idx))
            cmd_plot.scatter(x='x', y='y', source=cmd_cds,
                             view = view,
                             size=20, color='gold', marker='star')
        update_cmd_plot(star_init)

        ### CALLBACK FUNCTIONS ###
        # When you change the star, also update the stamps choices, and run the
        # change function for the stamp selector
        def change_star(attrname, old_id, new_id):
            """
            Change star: when you change the star, you need to:
            1. Update the choice of target stamps
            2. Update the target stamp display
            2a. Load the subtraction products for the default target stamp.
            3. Update the reference stamps and reference stamp cube scroller
            """
            # load new table rows
            update_star_table(star_info_table, new_id, ana_mgr)
            # load a new set of stamps
            target_stamp_ids = sorted(ana_mgr.db.find_matching_id(new_id, 'T'))
            target_stamp_init = target_stamp_ids[0]
            # this should trigger the selector_stamp.on_change method
            selector_stamp.update(
                value=target_stamp_init,
                options=target_stamp_ids,
            )
            # load a new set of reference stamps
            load_new_references(ana_mgr, new_id, reference_stamps_cds)
            # get_new_reference_stamps(new_id, reference_stamp_cds, refstamp_scroller)
            # get the catalog name of this star
            star_name = get_star_name_from_id(new_id, ana_mgr)
            display_star_name.update(value=star_name)
            # update the CMD plot
            update_cmd_plot(new_id)


        # define what happens when you change the target stamp
        # All PSF subtraction products should update - references, residuals,
        # SNR maps, models
        def change_target_stamp(attrname, old_id, new_id):
            # first, update the target stamp plot
            update_target_stamp_plot(new_id, target_stamp_plot, ana_mgr.db)
            # update the snr, residual, and psf models
            load_result_stamps(ana_mgr, kind='snr', 
                               star_id=selector_star.value, stamp_id=selector_stamp.value,
                               cds=snrmap_cds, title='SNR map')
            load_result_stamps(ana_mgr, kind='residuals', 
                               star_id=selector_star.value, stamp_id=selector_stamp.value,
                               cds=residuals_cds, title='Residuals')
            load_result_stamps(ana_mgr, kind='models', 
                               star_id=selector_star.value, stamp_id=selector_stamp.value,
                               cds=psfmodel_cds, title='PSF Model')
            # print out the catalog name for this star


        # define what happens when you change the detection candidate
        def change_detection(attrname, old_id, new_id):
            """
            When you change the detection candidate, you are just setting the
            star and target stamps
            """
            new_star, new_stamp = new_id.split(" / ")
            selector_star.update(value=new_star)
            selector_stamp.update(value=new_stamp)

        def update_target_stamp_plot(
                stamp_id : str,
                figure : bokeh.plotting.figure,
                db,
        ) -> None:
            """Updates the target stamp on the target stamp figure object"""
            # update the title text
            # add the filter name to the plot title
            ps_id = db.stamps_tab.set_index("stamp_id").loc[f'{stamp_id}', 'stamp_ps_id']
            filt_id = db.ps_tab.set_index("ps_id").loc[ps_id, 'ps_filt_id']
            filt_name = db.lookup_dict['lookup_filters'].set_index('filt_id').loc[filt_id, 'filter']
            figure.title.update(text = f'{selector_star.value} / {selector_stamp.value} ({filt_name})')
            # get the new stamp
            target_stamp = db.stamps_tab.set_index("stamp_id").loc[f'{stamp_id}', 'stamp_array']
            mapper = bkmdls.LinearColorMapper(
                palette='Magma256',
                low=np.nanmin(target_stamp),
                high=np.nanmax(target_stamp)
            )
            figure.image(image=[target_stamp],
                         x=-0.5, y=-0.5,
                         dw=target_stamp.shape[1], dh=target_stamp.shape[0],
                         color_mapper=mapper)
            return None

        def update_scroller_plot_on_new_cube(
                cds : bkmdls.ColumnDataSource,
                scroller = bklyts.column,
        ):
            """Update a cube scroller when you update the data"""
            figure, slider = scroller.children
            slider.update(end=cds.data['len'][0]-1)


        ### GUI ELEMENTS ###
        # make the selector for the detection candidates
        selector_detections = bkmdls.Select(
            title = 'Detection Candidates',
            value = candidate_init,
            options = candidate_ids,
        )
        selector_detections.on_change("value", change_detection)

        # make the selector for the stars based on ID
        selector_star = bkmdls.Select(
            title = 'Star ID',
            value = star_init,
            options = star_ids,
        )
        selector_star.on_change("value", change_star)

        # make the selector for the stars based on the catalog name
        display_star_name = bkmdls.TextInput(
            title = 'Star Name',
            value = star_name_init
        )

        # make the selector for the star's stamps
        target_stamp_ids = sorted(ana_mgr.db.find_matching_id(star_init, 'T'))
        target_stamp_init = target_stamp_ids[0]
        selector_stamp = bkmdls.Select(
            title='Stamp ID',
            value = target_stamp_init,
            options = target_stamp_ids
        )
        selector_stamp.on_change("value", change_target_stamp)


        # make a plot for the corresponding stamp
        target_stamp_plot = figure(match_aspect=True)
        # initialize it
        update_target_stamp_plot(selector_stamp.value, target_stamp_plot, ana_mgr.db)

        # make the cube scroller for the references
        refstamp_scroller = generate_cube_scroller_widgets(reference_stamps_cds, add_log_scale=True)

        # PSF subtraction results - SNR map, residuals, PSF model
        snr_scroller = generate_cube_scroller_widgets(
            snrmap_cds,
            slider_title_prefix='Nmodes',
            slider_kwargs={'show_value': False},
            diverging_cmap=True,
        )
        resid_scroller = generate_cube_scroller_widgets(
            residuals_cds,
            slider_title_prefix='Nmodes',
            slider_kwargs={'show_value': False},
            diverging_cmap=True,
        )
        psfmodel_scroller = generate_cube_scroller_widgets(
            psfmodel_cds,
            slider_title_prefix='Nmodes',
            slider_kwargs={'show_value': False},
            add_log_scale=True,
        )


        # define the dashboard layout
        tab_overview = bkmdls.TabPanel(
            child = bklyts.row(
                bklyts.column(target_stamp_plot, refstamp_scroller),
                cmd_plot,
            ),
            title = 'System Overview'
        )
        tab_snr = bkmdls.TabPanel(
            child = bklyts.column(
                bklyts.row(snr_scroller, resid_scroller),
                psfmodel_scroller
            ),
            title = 'SNR'
        )
        tabs = bkmdls.Tabs(tabs=[
            tab_overview,
            tab_snr
        ])
        lyt = bklyts.layout([
            # the target selectors
            bklyts.row(
                selector_detections,
                selector_star,
                selector_stamp,
                display_star_name,
            ),
            # the information table
            bklyts.row(
                star_info_table,
            ),
            # the tabs with the plots
            bklyts.row(tabs),
        ])

        doc.add_root(lyt)

        # define the layout look
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

