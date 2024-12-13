"""
Result visualization dashboard. Uses Bokeh to draw interactive plots.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import functools
 
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
) -> None:
    # store the images in the CDS
    if cds is None:
        # if none provided, make one.
        cds = bkmdls.ColumnDataSource()

    data = dict(
        # these are the fields that the plots will inspect
        img=[img],
        # these fields help format the plots
        # lower left corner coordinates
        x = [-0.5], y = [-0.5],
        # image height and width in pixels
        dw = [img.shape[-1]], dh = [img.shape[-2]],
        low = [np.nanmin(img)], high = [np.nanmax(img)],
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
        x = [-0.5], y = [-0.5],
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
):
    """
    Generate a contained layout object to scroll through a cube. Can be added to other documents.
    
    Parameters
    ----------
    source : bkmdls.ColumnDataSource
      a CDS containing the data to plot. Generate with series_to_CDS(pd.Series)
    plot_kwargs : dict
      kwargs to pass to bkplt.figure
    slider_kwargs : dict
      kwargs to pass to bkmdls.Slider
    add_log_scale : bool = False
      Add a switch to change the color mapping between linear and log scaling
    diverging_cmap : bool = False
      If True, use a diverging Blue-Red colormap that is white in the middle
      This is useful for residual plots that are mean 0
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
        **plot_kwargs
    )
    # Stamp image
    img_plot = plot.image(
        image='img',
        source=source,
        x=-0.5, y=-0.5,
        dw=source.data['img'][0].shape[-1], dh=source.data['img'][0].shape[-2],
        palette=palette,
    )
    # Add a color bar to the image
    color_mapper = bkmdls.LinearColorMapper(
        palette=palette,
        low=np.nanmin(source.data['img'][0]), high=np.nanmax(source.data['img'][0]),
    )
    color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
    plot.add_layout(color_bar, 'right')

    # Slider
    slider = bkmdls.Slider(
        start=0, end=source.data['nimgs'][0]-1,
        value=0, step=1,
        title = str(source.data['i'][0])
    )
    def slider_change(attr, old, new):
        # update the current index, used for the slider title
        source.data['i'] = [source.data['index'][0][new]]
        # update which slice of the data
        source.data['img'] = [source.data['cube'][0][new]]
        # update the slider title
        slider.update(title=str(source.data['i'][0]))
        color_mapper.update(
            low=np.nanmin(source.data['img'][0]),
            high=np.nanmax(source.data['img'][0]),
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
    p = bkplt.figure(
        width=size, height=size,
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


# helper functions for the dashboard
def make_row_cds(row, star, cds_dict={}):
    # make CDSs with the row data, updating the existing CDSs if they are provided
    filt = row['filter']
    # filter stamp
    cds = cds_dict.get("stamp", None)
    cds_dict['stamp'] = img_to_CDS(row['stamp'].data, cds=cds)
    # cube of references
    refs = star.row_get_references(row).query("used == True").copy()
    refs = refs.sort_values(by='sim', ascending=False)
    refcube = refs['stamp'].apply(getattr, args=['data'])
    refcube_index = refs.reset_index().apply(lambda row: f"{row['target']} / SIM {row['sim']:0.3f}", axis=1)
    # print(len(refs), len(refcube))
    cds = cds_dict.get("references", None)
    cds_dict['references'] = series_to_CDS(refcube, cds, index=list(refcube_index.values))
    # PSF model
    cds = cds_dict.get("psf_model", None)
    cds_dict['psf_model'] = series_to_CDS(row['psf_model'], cds, index=list(row['psf_model'].index.astype(str).values))
    # KLIP residuals
    cds = cds_dict.get("klip_residuals", None)
    cds_dict['klip_residuals'] = series_to_CDS(row['kl_sub'], cds, index=list(row['kl_sub'].index.astype(str).values))
    return cds_dict

def make_row_plots(row, row_cds, size=400):
    filt = row['filter']
    img_plot = make_static_img_plot(row_cds['stamp'], title=filt, size=size)
    # References
    cds = row_cds["references"]
    plot_kwargs={
        "title": f"References",
        "width": size, "height": size
    }
    ref_scroller = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
    )
    # PSF model
    cds = row_cds["psf_model"]
    plot_kwargs={
        "title": f"PSF model",
        "width": size, "height": size
    }
    psf_model_scroller = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
    )
    # KLIP residuals
    cds = row_cds["klip_residuals"]
    plot_kwargs={
        "title": f"KLIP residuals",
        "width": size, "height": size
    }
    klip_resid_scroller = generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
        use_diverging_cmap=True,
    )

    # Put them all in a dict
    plots = dict(
        stamp=img_plot,
        references=ref_scroller, 
        psf_model=psf_model_scroller, 
        klip_residuals=klip_resid_scroller
    )
    return plots

def all_stars_dashboard(
    stars : pd.Series,
    plot_size = 400,
):
    # This returns a Bokeh application that takes a doc for displaying
    
    def app(doc):
        
        init_star = stars.index[0]

        # generate the data structures and initialize the plots
        cds_dicts = stars.loc[init_star].results.apply(
            lambda row: make_row_cds(row, star=stars.loc[init_star], cds_dict={}),
            axis=1
        )
        plot_dicts = stars.loc[init_star].results.apply(
            lambda row: make_row_plots(row, cds_dicts[row.name], size=plot_size),
            axis=1
        )
        catalog_cds = bkmdls.ColumnDataSource(stars.loc[init_star].cat.drop("stamp", axis=1))
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
            width=3*plot_size, height=100,
        )

        # Button to stop the server
        quit_button = bkmdls.Button(label="Stop server", button_type="warning")
        quit_button.on_click(lambda: sys.exit())

        # star selector
        star_selector = bkmdls.Select(
            title="Star",
            value = init_star,
            options = list(stars.index),
        )
        def change_star(attrname, old_id, new_id):
            ## update the data structures and scroller limits
            update_catalog_cds()
            update_cds_dicts()
            update_scrollers()
        star_selector.on_change("value", change_star)


        def update_catalog_cds():
           catalog_cds.data.update(stars.loc[star_selector.value].cat.drop("stamp", axis=1))

        def update_cds_dicts():
            """Update the target filter stamp"""
            for i, row in stars.loc[star_selector.value].results.iterrows():
                # assign new data to the existing CDSs
                cds_dicts[i] = make_row_cds(row, stars.loc[star_selector.value], cds_dicts[i])

        def update_scrollers():
            # update scroller range and options
            for i, row_plot in plot_dicts.items():
                for k in ['references', 'psf_model', 'klip_residuals']:
                    source = cds_dicts[i][k]
                    row_plot[k].children[1].update(
                        value = 0,
                        end = source.data['nimgs'][0]-1,
                        title = source.data['i'][0],
                    )

        # lyt = bklyts.layout([
        #     # the target selectors
        #     bklyts.row(star_selector, catalog_table, quit_button),
        #     bklyts.row(
        #         plot_dicts[0]['stamp'],
        #         plot_dicts[0]['references'],
        #         plot_dicts[0]['psf_model'],
        #         plot_dicts[0]['klip_residuals'],
        #     ),
        #     bklyts.row(
        #         plot_dicts[1]['stamp'],
        #         plot_dicts[1]['references'],
        #         plot_dicts[1]['psf_model'],
        #         plot_dicts[1]['klip_residuals'],
        #     ),
        # ])
        tab1 = bkmdls.TabPanel(
            title='Overview',
            child= bklyts.layout([
                # the target selectors
                bklyts.row(
                    plot_dicts[0]['stamp'],
                    plot_dicts[0]['references'],
                    plot_dicts[0]['psf_model'],
                ),
                bklyts.row(
                    plot_dicts[1]['stamp'],
                    plot_dicts[1]['references'],
                    plot_dicts[1]['psf_model'],
                ),
            ]),
        )
        tab2 = bkmdls.TabPanel(
            title='Detection',
            child= bklyts.layout([
                # the target selectors
                bklyts.row(
                    plot_dicts[0]['klip_residuals'],
                ),
                bklyts.row(
                    plot_dicts[1]['klip_residuals'],
                ),
            ]),
        )
        tab = bkmdls.Tabs(tabs=[tab1, tab2])

        lyt = bklyts.layout(
            [
                bklyts.row(star_selector, catalog_table, quit_button),
                bklyts.row(tab),
            ]
        )
            
        
        doc.add_root(lyt)
    
    return app
