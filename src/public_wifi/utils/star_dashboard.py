"""
Result visualization dashboard. Uses Bokeh to draw interactive plots.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import functools
from itertools import combinations

from public_wifi import catalog_processing as catproc

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
        title = str(source.data['i'][0]),
        name='slider',
    )
    def slider_change(attr, old, new):
        # update the current index, used for the slider title
        source.data['i'] = [source.data['index'][0][new-1]]
        # update which slice of the data
        source.data['img'] = [source.data['cube'][0][new-1]]
        # update the slider title
        slider.update(title=str(source.data['i'][0]))
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
    if not isinstance(img, bkmdls.ColumnDataSource):
        img = img_to_CDS(img)
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


# helper functions for the dashboard
def make_row_cds(row, star, cds_dict={}, jackknife_kklip : int = 1):
    """
    Make CDSs with the row data, updating the existing CDSs if they are provided

    Parameters
    ----------
    row : pd.Series
      A row with the analysis results
    star : starclass.Star
      A star object, for selecting references
    cds_dict : dict = {}
      A dictionary of pre-existing CDSs. If if the CDSs exist, they will be
      updated with the new information. If not, CDSs will be created.
    jackknife_kklip : int = 1
      select this jackknife mode

    """
    filt = row['filter']
    # filter stamp
    cds = cds_dict.get("stamp", None)
    cds_dict['stamp'] = img_to_CDS(row['stamp'], cds=cds)
    # cube of references
    refs = star.row_get_references(row).query("used == True").copy()
    refs = refs.sort_values(by='sim', ascending=False)
    refcube = refs['stamp']#.apply(getattr, args=['data'])
    refcube_index = refs.reset_index().apply(
        lambda row: f"{row['target']} / SIM {row['sim']:0.3f}",
        axis=1
    )
    cds = cds_dict.get("references", None)
    cds_dict['references'] = series_to_CDS(
        refcube,
        cds,
        index=list(refcube_index.values)
    )
    # PSF model
    cds = cds_dict.get("klip_model", None)
    cds_dict['klip_model'] = series_to_CDS(
        row['klip_model'],
        cds,
        index=list(row['klip_model'].index.astype(str).values)
    )
    # klip residuals
    cds = cds_dict.get("klip_residuals", None)
    cds_dict['klip_residuals'] = series_to_CDS(
        row['klip_sub'],
        cds,
        index=list(row['klip_sub'].index.astype(str).values)
    )

    # detection maps
    cds = cds_dict.get("klip_mf", None)
    cds_dict['klip_mf'] = series_to_CDS(
        row['detmap'],
        cds,
        index=list(row['detmap'].index.astype(str).values)
    )
    cds = cds_dict.get("snr_maps", None)
    cds_dict['snr_maps'] = series_to_CDS(
        row['snrmap'],
        cds,
        index=list(row['snrmap'].index.astype(str).values)
    )

    cds = cds_dict.get("flux_maps", None)
    cds_dict['flux_maps'] = series_to_CDS(
        row['fluxmap'],
        cds,
        index=list(row['fluxmap'].index.astype(str).values)
    )

    # jackknife test maps
    cds = cds_dict.get("klip_jackknife", None)
    kklip = min([jackknife_kklip, len(row['snrmap'])-1])
    jackknife_data = pd.DataFrame(row['klip_jackknife']).query(f"numbasis == {kklip}")['klip_jackknife']
    cds_dict['klip_jackknife'] = series_to_CDS(
        jackknife_data,
        cds,
        index = [f"{i[0]} / {i[1]}" for i in jackknife_data.index],
    )
    return cds_dict

def make_row_plots(row, row_cds, size=400):
    plots = {}

    filt = row['filter']
    plots['stamp'] = make_static_img_plot(row_cds['stamp'], title=filt, size=size)

    # References
    cds = row_cds["references"]
    plot_kwargs={
        "title": f"References",
        "width": size, "height": size
    }
    plots['references'] = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
    )

    # PSF model
    cds = row_cds["klip_model"]
    plot_kwargs={
        "title": f"KLIP model",
        "width": size, "height": size
    }

    plots['klip_model'] = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
    )

    # KLIP residuals
    cds = row_cds["klip_residuals"]
    plot_kwargs={
        "title": f"KLIP residuals",
        "width": size, "height": size
    }
    plots['klip_residuals'] = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
        use_diverging_cmap=False,
    )

    # Detection maps
    cds = row_cds["klip_mf"]
    plot_kwargs={
        "title": f"Matched filter - detection",
        "width": size, "height": size
    }
    plots['klip_mf'] = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
        use_diverging_cmap=False,
    )

    # Flux maps
    cds = row_cds["flux_maps"]
    plot_kwargs={
        "title": f"Matched filter - flux",
        "width": size, "height": size
    }
    plots['flux_maps'] = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
        use_diverging_cmap=False,
    )

    # SNR map
    cds = row_cds["snr_maps"]
    plot_kwargs={
        "title": f"SNR map",
        "width": size, "height": size
    }
    plots['snr_maps'] = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
        use_diverging_cmap=False,
    )
    # Jackknife
    cds = row_cds["klip_jackknife"]
    plot_kwargs={
        "title": f"Jackknife SNR maps",
        "width": size, "height": size
    }
    plots['klip_jackknife'] = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
        use_diverging_cmap=False,
    )

    return plots

def make_catalog_display(
        catalog_rows : pd.DataFrame,
        plot_size : int = 1000,

) -> tuple[bkmdls.ColumnDataSource, bkmdls.DataTable] :
    """
    Generete the column data source and display widget for a star's catalog rows

    Parameters
    ----------
    catalog_rows : pd.DataFrame
      The entries in a catalog that correspond to a particular star
    plot_size : int = 400
      the size parameter for the widget

    Output
    ------
    Define your output

    """
    catalog_cds = bkmdls.ColumnDataSource(catalog_rows)
    catalog_columns = []
    for k in catalog_cds.data.keys():
        formatter = bkmdls.StringFormatter()
        if isinstance(catalog_cds.data[k][0], float):
            formatter = bkmdls.NumberFormatter(format='0.00')
        catalog_columns.append(
            bkmdls.TableColumn(field=k, title=k, formatter=formatter)
        )
    catalog_table = bkmdls.DataTable(
        source=catalog_cds,
        columns=catalog_columns,
        # sizing_mode="stretch_width",
        width=plot_size, height=40*len(catalog_rows),
    )
    return catalog_cds, catalog_table

def make_table_cds(
        dataframe : pd.DataFrame,
        cds=None,
        table=None,
        plot_scale=60
):
    if cds is None:
        cds = bkmdls.ColumnDataSource()
    cds.data.update(dataframe)
    if table is None:
        # if no table provided, make a new one. Else, just return the existing table
       columns = [bkmdls.TableColumn(field=k, title=k) for k in cds.data.keys()]
       table = bkmdls.DataTable(
           source=cds,
           columns=columns,
           width = plot_scale * (dataframe.shape[1]+1),
           height = plot_scale * (dataframe.shape[0]+1),
           # sizing_mode='stretch_height',
       )
    return cds, table

def make_candidate_cds(
        result_rows,
        cds=None,
        table=None,
        plot_scale=60,
):
    """
    Make the CDS and display table for the candidates

    result_rows : dataframe
      rows to show. each must have the columns ['filter','snr_candidates']
    cds : ColumnDataSource
      existing CDS to update, or None to make a new one
    table : DataTable
      existing Table to update, or None to make a new one
    plot_scale : scale the plot size by this times the dataframe shape
    """
    candidates = result_rows.set_index("filter")['snr_candidates']
    cand_df = pd.concat(candidates.to_dict(), names=['filter', 'pix_id'])#.reset_index('filter')
    cand_df = cand_df.reset_index().drop(columns="pix_id").sort_values(by=['filter', 'cand_id'])
    # source = bkmdls.ColumnDataSource(cand_df)
    cds, table = make_table_cds(cand_df, cds=cds, table=table, plot_scale=plot_scale)
    cds.data.update(cand_df)
    return cds, table

def all_stars_dashboard(
    stars : pd.Series,
    plot_size = 400,
):
    # This returns a Bokeh application that takes a doc for displaying

    def app(doc):

        init_star = stars.index[0]

        catalog_cds, catalog_table = make_catalog_display(
            stars.loc[init_star].cat.drop(["cutout", "stamp" ], axis=1),
            70*len(stars.loc[init_star].cat.columns),
        )
        # candidates table
        candidate_cds, candidate_table = make_candidate_cds(
            stars.loc[init_star].results,
            plot_scale=60,
        )
        candidate_table.update(sizing_mode='stretch_width')

        # each row of the catalog corresponds to a particular set of plots
        cds_dicts = stars.loc[init_star].results.apply(
            lambda row: make_row_cds(row, star=stars.loc[init_star], cds_dict={}),
            axis=1
        )

        # pass these ColumnDataSources to the plots that will read from them
        plot_dicts = stars.loc[init_star].results.apply(
            lambda row: make_row_plots(row, cds_dicts[row.name], size=plot_size),
            axis=1
        )

        # Button to stop the server
        quit_button = bkmdls.Button(label="Stop server", button_type="warning")
        def stop_server():
            quit_button.update(button_type='danger')
            sys.exit()
        quit_button.on_click(stop_server)
        
        # star selector
        star_selector = bkmdls.Select(
            title="Star",
            value = init_star,
            options = list(stars.index),
        )
        def change_star(attrname, old_id, new_id):
            ## update the data structures and scroller limits
            update_reference_switch()
            update_candidates_switch()
            update_catalog_cds()
            update_candidate_cds()
            update_cds_dicts()
            update_cube_scrollers()
        star_selector.on_change("value", change_star)
        # Print a star name to terminal for copying
        print_button = bkmdls.Button(label="Print star name", button_type="default")
        def print_star_name():
            print(star_selector.value)
        print_button.on_click(print_star_name)

        def update_reference_switch():
            status  = stars.loc[star_selector.value].is_good_reference
            button_type = "success" if status else "danger"
            good_reference_switch.update(button_type=button_type)

        def update_candidates_switch():
            status  = stars.loc[star_selector.value].has_candidates
            button_type = "success" if status else "danger"
            has_candidates_switch.update(button_type=button_type)
            # also update the reference switch
            update_reference_switch()

        def update_candidate_cds():
            make_candidate_cds(
                stars.loc[star_selector.value].results,
                cds=candidate_cds,
                table=candidate_table,
                plot_scale=plot_size,
            )
        def update_catalog_cds():
           catalog_cds.data.update(
               stars.loc[star_selector.value].cat.drop(["cutout", "stamp" ], axis=1)
           )

        def update_cds_dicts():
            """Update the target filter stamp"""
            for i, row in stars.loc[star_selector.value].results.iterrows():
                # assign new data to the existing CDSs
                nrefs = stars.loc[star_selector.value].references.shape[0]//2 - 1
                cds_dicts[i] = make_row_cds(
                    row,
                    stars.loc[star_selector.value],
                    cds_dicts[i],
                    jackknife_mode_selector.value,
                )

        def update_cube_scrollers():
            scroller_keys = [
                'references', 'klip_model',
                'klip_residuals', 'klip_mf', 'snr_maps', 'flux_maps',
                'klip_jackknife',
            ]
            for i, row_plot in plot_dicts.items():
                for k in scroller_keys:
                    source = cds_dicts[i][k]
                    # update slider range and options
                    row_plot[k].children[1].update(
                        value = 1,
                        end = source.data['nimgs'][0],
                        title = source.data['i'][0],
                    )

        # link all Kklip scrollers together:
        link_scrollers = [
            'klip_residuals', 'klip_mf', 'snr_maps', 'flux_maps',
        ]
        for kp in plot_dicts.keys():
            for combo in combinations(link_scrollers, 2):
                plot_dicts[kp][combo[0]].children[1].js_link(
                    'value', plot_dicts[kp][combo[1]].children[1], 'value'
                )
                plot_dicts[kp][combo[1]].children[1].js_link(
                    'value', plot_dicts[kp][combo[0]].children[1], 'value'
                )

        # good reference flat
        good_reference_switch = bkmdls.Button(
            # label = str(stars[star_selector.value].is_good_reference)
            label = "'Good reference' state",
            button_type = "success" if stars[star_selector.value].is_good_reference else "danger",
            sizing_mode='stretch_width',
        )
        def change_reference_status():
            status = stars[star_selector.value].is_good_reference
            new_status = not status
            stars[star_selector.value].is_good_reference = new_status
            print(stars[star_selector.value].star_id, f" reference flag set to {new_status}")
            button_type = "success" if new_status else "danger"
            good_reference_switch.update(button_type=button_type)
        good_reference_switch.on_click(change_reference_status)

        has_candidates_switch = bkmdls.Button(
            # label = str(stars[star_selector.value].is_good_reference)
            label = "'Has candidates' state",
            button_type = "success" if stars[star_selector.value].has_candidates else "danger",
            sizing_mode='stretch_width',
        )
        def change_candidates_status():
            status = stars[star_selector.value].has_candidates
            new_status = not status
            stars[star_selector.value].has_candidates = new_status
            print(stars[star_selector.value].star_id, f" candidates flag set to {new_status}")
            button_type = "success" if new_status else "danger"
            has_candidates_switch.update(button_type=button_type)
        has_candidates_switch.on_click(change_candidates_status)

        # reprocessing tools
        ssim_spinner = bkmdls.Spinner(
            title='SSIM thresh', low=-1.0, high=1.0, step=0.05, value=0.5, width=80
        )
        min_nref_spinner = bkmdls.Spinner(
            title='Min. refs', low=2, high=len(stars), step=1, value=5, width=80
        )
        snr_thresh_spinner = bkmdls.Spinner(
            title='SNR thresh', low=0, high=len(stars), step=1, value=5, width=80
        )
        nmodes_thresh_spinner = bkmdls.Spinner(
            title='# modes thresh', low=1, high=len(stars), step=1, value=3, width=80
        )
        jackknife_mode_selector = bkmdls.Spinner(
            title="Jackknife Kklip",
            low=1, value=1, high=len(stars), step=1, width=80,
        )
        jackknife_mode_selector.on_change('value', lambda attr, old, new: update_cds_dicts)

        subtraction_button = bkmdls.Button(
            label='Run subtraction', button_type='primary',
            sizing_mode='stretch_width',
        )
        def rerun_subtraction_and_update():
            catproc.catalog_subtraction(
                stars, ssim_spinner.value, min_nref_spinner.value
            )
            catproc.catalog_detection(
                stars, snr_thresh_spinner.value, nmodes_thresh_spinner.value
            )
            # update stuff as if it were a new star
            change_star('value', None, star_selector.value)
            print("Finished re-running analysis and updating values.")
        subtraction_button.on_click(rerun_subtraction_and_update)

        detection_button = bkmdls.Button(
            label='Run detection only', button_type='primary',
            sizing_mode='stretch_width',
        )
        def rerun_detection_and_update():
            catproc.catalog_detection(
                stars, snr_thresh_spinner.value, nmodes_thresh_spinner.value
            )
            # update stuff as if it were a new star
            change_star('value', None, star_selector.value)
            print("Finished re-running analysis and updating values.")
        detection_button.on_click(rerun_detection_and_update)

        reanalysis_lyt = bklyts.row(
            bklyts.column(
                good_reference_switch,
                has_candidates_switch,
                subtraction_button,
                detection_button,
            ),
            bklyts.column(
                ssim_spinner, min_nref_spinner,
                snr_thresh_spinner, nmodes_thresh_spinner,
                jackknife_mode_selector,
            ),
        )

        tab1 = bkmdls.TabPanel(
            title='Overview',
            child= bklyts.layout([
                # the target selectors
                bklyts.row(
                    plot_dicts[0]['stamp'],
                    plot_dicts[0]['references'],
                    plot_dicts[0]['klip_model'],
                ),
                bklyts.row(
                    plot_dicts[1]['stamp'],
                    plot_dicts[1]['references'],
                    plot_dicts[1]['klip_model'],
                ),
            ]),
        )
        tab2 = bkmdls.TabPanel(
            title='KLIP results',
            child= bklyts.layout([
                # the target selectors
                bklyts.row(
                    plot_dicts[0]['klip_residuals'],
                    plot_dicts[0]['snr_maps'],
                    # plot_dicts[0]['klip_mf'],
                    plot_dicts[0]['flux_maps'],
                    plot_dicts[0]['klip_jackknife'],
                ),
                bklyts.row(
                    plot_dicts[1]['klip_residuals'],
                    plot_dicts[1]['snr_maps'],
                    # plot_dicts[1]['klip_mf'],
                    plot_dicts[1]['flux_maps'],
                    plot_dicts[1]['klip_jackknife'],
                ),
            ]),
        )
        # tab3 = bkmdls.TabPanel(
        #     title='NMF results (tbd)',
        #     child= bklyts.layout([
        #         bklyts.column()
        #     ])
        # )
        tabs = bkmdls.Tabs(tabs=[tab1, tab2])#, tab3])

        lyt = bklyts.layout(
            [
                bklyts.row(
                    bklyts.column(star_selector, print_button),
                    catalog_table
                ),
                bklyts.row(
                    bklyts.column(reanalysis_lyt, candidate_table, sizing_mode='stretch_height'),
                    tabs
                ),
                bklyts.row(quit_button),
            ]
        )


        doc.add_root(lyt)
    return app
