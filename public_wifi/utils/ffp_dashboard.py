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


def generate_cube_scroller_widgets(
    cds : bkmdls.ColumnDataSource,
    cmap_class : bokeh.core.has_props.MetaHasProps = bkmdls.LinearColorMapper,
    slider_title_prefix : str = '',
    plot_kwargs={},
    slider_kwargs={}
):
    """
    Generate a plot and a scroller to scroll through the cube stored in a provided ColumnDataSource
    
    Parameters
    ----------
    plot_kwargs : dict
      kwargs to pass to bkplt.figure
    slider_kwargs : dict
      kwargs to pass to bkmdls.Slider
    """
    TOOLS='save'

    plot = figure(title=cds.data.get('title', [''])[0], tools=TOOLS, **plot_kwargs)
    color_mapper = cmap_class(
        palette='Magma256',
        low=np.nanmin(cds.data['curimg']),
        high=np.nanmax(cds.data['curimg'])
    )

    plot.image(
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
        new_index = cds.data['index'][0][new]
        img = cds.data['cube'][0][new_index]
        cds.data['curimg'] = [img]
        color_mapper.update(low=np.nanmin(img), high=np.nanmax(img))
        slider.update(title = slider_title(cds.data['index'][0][new]))

    slider = bkmdls.Slider(
        start=0, end=len(cds.data['index'][0])-1, value=0, step=1,
        title = slider_title(cds.data['index'][0][0]),
        **slider_kwargs
    )
    slider.on_change('value', slider_change)
    
    return bklyts.column(plot, slider)

##### FFP Dashboard - no mosaic! ####

def series_to_CDS(
    cds : bkmdls.ColumnDataSource,
    cube : pd.Series,
    properties : dict = {},
) -> None:
    """
    When you get a new cube, you should update its corresponding CDS.
    This is a generic helper function to do that.

    properties : dict = {}
      this argument is used to pass any other entries to the data field that you may want to add
    """
    # store the images in the CDS
    # cds.data.update({str(i): [j] for i, j in enumerate(cube)})
    cds.data.update(cube=[cube])
    cds.data.update(index=[list(cube.index)])
    # cds.data.update({f"label_{i}" : [j] for i, j in enumerate(cube.index)})
    cube_shape = np.stack(cube.values).shape
    # add the new cube information to the properties update dictionary
    properties.update({
        'curimg': [cube.iloc[0]],
        'x': [-0.5], 'y': [-0.5],
        'dw': [cube_shape[-2]], 'dh': [cube_shape[-1]],
        'len': [len(cube)],
    })
    cds.data.update(**properties)
    return None

def load_new_references(
    ana_mgr, 
    star_id : str, 
    cds : bkmdls.ColumnDataSource,
):
    reference_ids = ana_mgr.results_stamps['references'].loc[star_id].dropna(how='all', axis=1)
    reference_ids = reference_ids.drop_duplicates().values.ravel()
    reference_stamps = ana_mgr.db.stamps_tab.set_index('stamp_id').loc[reference_ids, 'stamp_array']
    series_to_CDS(
        cds,
        reference_stamps.sort_index(), #np.stack(reference_stamps.values),
        properties={'title': ["Reference stamps"]}
    )


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
        cds,
        target_stamps.sort_index(),
        properties={'title': ['Target star stamps']}
    )

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
        cds,
        results,
        properties={'title': [title]}
    )

def dashboard(
        ana_mgr,
        plot_size_unit : int = 50,
):

    def app(doc):

        ### INITIALIZE DATA STRUCTURES ###
        star_ids = star_ids = sorted(ana_mgr.db.stars_tab['star_id'])
        star_init = star_ids[0]

        # load this star's target stamps
        target_stamps_cds = bkmdls.ColumnDataSource()
        load_target_stamps(ana_mgr, star_init, target_stamps_cds)
        target_stamp_init = target_stamps_cds.data['index'][0][0]

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
        # make the selector for the stars.
        
        selector_star = bkmdls.Select(
            title = 'Star ID',
            value = star_init,
            options = star_ids,
        )
        selector_star.on_change("value", change_star)

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
        target_stamp_plot = figure()
        # initialize it
        update_target_stamp_plot(selector_stamp.value, target_stamp_plot, ana_mgr.db)

        # make the cube scroller for the references
        refstamp_scroller = generate_cube_scroller_widgets(reference_stamps_cds)

        # # PSF subtraction results
        snr_scroller = generate_cube_scroller_widgets(snrmap_cds, slider_title_prefix='Nmodes', slider_kwargs={'show_value': False})
        resid_scroller = generate_cube_scroller_widgets(residuals_cds, slider_title_prefix='Nmodes', slider_kwargs={'show_value': False})
        psfmodel_scroller = generate_cube_scroller_widgets(psfmodel_cds, slider_title_prefix='Nmodes', slider_kwargs={'show_value': False})


        # define the dashboard layout
        lyt = bklyts.column(
            bklyts.row(
                selector_star,
                selector_stamp,
            ),
            bklyts.row(
                target_stamp_plot,
                refstamp_scroller,
            ),
            bklyts.row(
                snr_scroller, 
                resid_scroller, 
                psfmodel_scroller
            ),
        )

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

