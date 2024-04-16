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




##### FFP Dashboard - no mosaic! ####

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
    target_stamp_ids = sorted(dbm.find_matching_id(star_id, 'T'))
    target_stamp_ids = dbm.stamps_tab.query('stamp_id in @target_stamp_ids')['stamp_id']
    ncols = 1
    nrows = len(target_stamp_ids)
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

    # grid = bklyts.gridplot(
    #     list(p_stamps),
    #     ncols=len(p_stamps),
    #     sizing_mode='scale_both',
    #     height=plot_size * nrows,#int(plot_size/nrows),
    #     width=plot_size * ncols,#int(plot_size/ncols),
    #     merge_tools=True, toolbar_location=None
    # )
    grid = bklyts.column(*list(p_stamps), sizing_mode='scale_both')
    return grid

# let's just show a plot where we can select the stars
def dashboard(
        ana_mgr,
        plot_size_unit : int = 50,
):

    def app(doc):

        # this dictionary will hold all the data that needs to be displayed
        # Any method that accesses data should do it through this dictionary
        # Any updates to the data should change the entries of this dictionary
        data_dict = {}

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
            get_new_reference_stamps(new_id, reference_stamp_cds)

        # define what happens when you change the target stamp
        # All PSF subtraction products should update - references, residuals,
        # SNR maps, models
        def change_target_stamp(attrname, old_id, new_id):
            # first, update the target stamp plot
            update_target_stamp_plot(new_id, target_stamp_plot, ana_mgr.db)


        def update_target_stamp_plot(
                stamp_id : str,
                figure : bokeh.plotting.figure,
                db,
        ) -> None:
            """Updates the target stamp on the target stamp figure object"""
            # update the title text
            figure.title.update(text = f'{selector_star.value} / {selector_stamp.value}')
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
    


        # make the selector for the stars.
        star_ids = sorted(ana_mgr.db.stars_tab['star_id'])
        star_init = star_ids[0]
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


        # cube scroller tools
        def generate_cube_scroller_widgets(
                cds : bkmdls.ColumnDataSource,
                cmap_class : bokeh.core.has_props.MetaHasProps = bkmdls.LinearColorMapper,

        ):
            """Generate a plot and a scroller to scroll through the cube stored in a provided ColumnDataSource"""
            plot = figure(title=cds.data.get('title', [''])[0])
            color_mapper = cmap_class(
                palette='Magma256',
                low=np.nanmin(cds.data['image']),
                high=np.nanmax(cds.data['image'])
            )

            plot.image(
                image='image',
                source=cds,
                x='x', y='y',
                dw='dw', dh='dh',
                color_mapper=color_mapper,
            )
            # add a color bar to the plot
            color_bar = bkmdls.ColorBar(color_mapper=color_mapper, label_standoff=12)
            plot.add_layout(color_bar, 'right')

            # add a slider to update the image and color bar when it moves
            def slider_change(attr, old, new):
                # update the 'image' entry to the desired image
                img = cds.data[str(new)]
                print(str(new))
                cds.data['image'] = img
                print(img)
                color_mapper.update(low=np.nanmin(img), high=np.nanmax(img))
                slider.update(title = cds.data[f'label_{new}'][0])

            slider = bkmdls.Slider(
                start=0, end=cds.data['len'][0]-1, value=0, step=1,
                title=cds.data['label_0'][0],
            )
            slider.on_change('value', slider_change)
            
            return bklyts.column(plot, slider)

        def update_cube_data(
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
            # reset the data field
            cds.data = {}
            # store the images in the CDS
            cds.data.update({str(i): [j] for i, j in enumerate(cube)})
            cds.data.update({f"label_{i}" : [j] for i, j in enumerate(cube.index)})
            cube_shape = np.stack(cube.values).shape
            properties.update({
                'image': [cube.iloc[0]],
                'x': [-0.5], 'y': [-0.5],
                'dw': [cube_shape[-2]], 'dh': [cube_shape[-1]],
                'len': [len(cube)],
            })
            cds.data.update(properties)
            return None

        def get_new_reference_stamps(
                star_id : str, 
                cds : bkmdls.ColumnDataSource,
        ) -> None:
            # set the reference stamp CDS values
            # these are set on the ColumnDataSource in-place
            reference_ids = ana_mgr.results_stamps['references'].loc[star_id].dropna(how='all', axis=1)
            reference_ids = reference_ids.drop_duplicates().values.ravel()
            reference_stamps = ana_mgr.db.stamps_tab.set_index('stamp_id').loc[reference_ids, 'stamp_array']
            update_cube_data(
                cds,
                reference_stamps.sort_index(), #np.stack(reference_stamps.values),
                properties={'title': ["Reference stamps"]}
            )
            return None

        reference_stamp_cds = bkmdls.ColumnDataSource()
        get_new_reference_stamps(selector_star.value, reference_stamp_cds)
        # get the reference stamps and make a ColumnDataSource
        refstamp_scroller = generate_cube_scroller_widgets(reference_stamp_cds)




        # define the dashboard layout
        lyt = bklyts.column(
            bklyts.row(
                bklyts.column(
                    selector_star,
                    selector_stamp,
                ),
                bklyts.column(
                    target_stamp_plot,
                    refstamp_scroller,
                )
            )
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



