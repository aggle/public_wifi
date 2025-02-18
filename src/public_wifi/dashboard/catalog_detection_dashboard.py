"""
This has the dashboard for the catalog-wide detection interface
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import functools
from itertools import combinations

import yaml
import bokeh
import bokeh.layouts as bklyts
import bokeh.plotting as bkplt
import bokeh.models as bkmdls

#from bokeh.models import ColumnDataSource, Slider, ColorBar, LogColorMapper
from bokeh.plotting import figure
from bokeh.themes import Theme

from public_wifi.dashboard import dash_tools as dt
from public_wifi import catalog_processing as catproc
from public_wifi import catalog_detection as catdet
from public_wifi import analysis_manager



def make_row_cds(
        det_mgr : catdet.CatDet,
        index,
        cds_dict={}
) -> dict:
    """
    det_mgr : an instance of the detection manager class
    index : which index of the cat/results dataframe does this correspond to
    cds_dict = {} : a dictionary of the CDSs that is checked for updates
    """
    # first, check to see if a CDS with this name already exists
    # if so, we will update it. if not, we will make a new one
    cds = cds_dict.get('matched_filter', None)
    cds_dict['matched_filter'] = dt.series_to_CDS(
        det_mgr.detection_maps[index],
        cds=cds,
        index_val = None if cds is None else cds.data['i'][0]
    )

    cds = cds_dict.get("residual_snr", None)
    cds_dict['residual_snr'] = dt.series_to_CDS(
        det_mgr.residual_snr_maps[index],
        cds=cds,
        index_val = None if cds is None else cds.data['i'][0]
    )
    cds = cds_dict.get("contrast", None)
    cds_dict['contrast'] = dt.series_to_CDS(
        det_mgr.contrast_maps[index],
        cds=cds,
        index_val = None if cds is None else cds.data['i'][0]
    )
    return cds_dict

def make_row_plots(det_mgr, index, row_cds, size=400):
    plots = {}

    cds = row_cds["matched_filter"]
    plot_kwargs={
        "title": f"Matched filter [SNR]",
        "width": size, "height": size
    }
    plots['matched_filter'] = dt.generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
    )

    cds = row_cds["residual_snr"]
    plot_kwargs={
        "title": f"Residuals [SNR]",
        "width": size, "height": size
    }
    plots['residual_snr'] = dt.generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
    )

    cds = row_cds["contrast"]
    plot_kwargs={
        "title": f"Contrast",
        "width": size, "height": size
    }
    plots['contrast'] = dt.generate_cube_scroller_widget(
        cds, plot_kwargs=plot_kwargs,
    )
    return plots

def link_scrollers(
        scroller_keys : list[str],
        plot_dicts : dict,
):
    """For this one, link *all* the scrollers together"""
    all_plots = [i for k in plot_dicts for i in plot_dicts[k].values()]
    # for kp in plot_dicts.keys():
    for combo in combinations(all_plots, 2):
            combo[0].children[1].js_link(
                'value', combo[1].children[1], 'value'
            )
            combo[1].children[1].js_link(
                'value', combo[0].children[1], 'value'
            )

    return

def link_multiindex_scrollers(
        plot_dicts : dict,
) -> None:
    all_plots = list(plot_dicts.values())
    for combo in combinations(all_plots, 2):
        children = combo[0].children[1:]
        # iterate over the scrollers and link both ways
        for scroller0 in children:
            scroller1 = combo[1].select_one(dict(name=scroller0.name))
            scroller0.js_link(
                "value", scroller1, "value"
            )
            print(f"{scroller0.name} linked to {scroller1.name}" )
            scroller1.js_link(
                "value", scroller0, "value"
            )
            print(f"{scroller1.name} linked to {scroller0.name}" )
    return

def detection_layout(
    anamgr : analysis_manager.AnalysisManager,
    plot_size = 400,
):
    cds_dicts = {
        i: make_row_cds(anamgr.det, i, cds_dict={})
        for i in [0, 1]
    }
    plot_dicts = {
        i: make_row_plots(anamgr.det, i, cds_dicts[i], size=plot_size)
        for i in [0, 1]
    }

    def update_cds_dicts():
        for i in cds_dicts.keys():
            cds_dicts[i] = make_row_cds(
                anamgr.det, i, cds_dict=cds_dicts[i]
            )
        return
    def update_cube_scrollers():
        scroller_keys = ['matched_filter', 'residual_snr', 'contrast']
        for i, row_plot in plot_dicts.items():
            for k in scroller_keys:
                source = cds_dicts[i][k]
                curr_index = source.data['i'][0]
                scroller_value = min(
                    [source.data['nimgs'][0], source.data['index'][0].index(curr_index)+1]
                )
                source.data['img'] = [source.data['cube'][0][scroller_value-1]]
                # update the slider range and options
                row_plot[k].children[1].update(
                    # this is necessary to keep the scroller in sync with the plot
                    value = scroller_value,
                    end = source.data['nimgs'][0],
                title=f"{len(source.data['index'][0])} / {source.data['i'][0]}"
                )
        return
    def update_dashboard():
        # whenever you update the data in the dashboard, call this function.
        # all updaters go in here
        update_cds_dicts()
        update_cube_scrollers()
        return


    # link the scrollers together
    link_scrollers(['matched_filter', 'residual_snr', 'contrast'], plot_dicts)

    # reprocessing tools
    # when this is changed, you just need to update kklip on
    # the anamgr.det object, which will re-filter the results. then you can
    # just update the plots.
    kklip_spinner = bkmdls.Spinner(
        title='Kklip',
        low=1, high=len(anamgr.stars), step=1,
        value=anamgr.det.kklip,
        width=110,
    )
    def update_kklip(attr, old, new):
        anamgr.det.kklip = new
        update_dashboard()
        return
    kklip_spinner.on_change('value', update_kklip)

    # when this is changed, update the matched filter width and reprocess
    mf_width_spinner = bkmdls.Spinner(
        title='Matched filter width',
        low=3, high=99,#anamgr.det.stamp_size,
        step=2,
        value=anamgr.det.mf_width,
        width=110,
    )
    redetect_button = bkmdls.Button(
        label='Re-run detection', button_type='primary',
        sizing_mode='stretch_width',
    )
    def rerun_detection_and_update():
        anamgr._update_detection_parameters(
            kklip=kklip_spinner.value,
            mf_width=mf_width_spinner.value
        )
        update_dashboard()
        return
    redetect_button.on_click(rerun_detection_and_update)

    lyt = bklyts.layout(
        [
            bklyts.row(
                bklyts.column(
                    redetect_button,
                    kklip_spinner,
                    mf_width_spinner,
                ),
                bklyts.column(
                    bklyts.row(
                        plot_dicts[0]['matched_filter'],
                        plot_dicts[0]['residual_snr'],
                        plot_dicts[0]['contrast'],
                    ),
                    bklyts.row(
                        plot_dicts[1]['matched_filter'],
                        plot_dicts[1]['residual_snr'],
                        plot_dicts[1]['contrast'],
                    ),
                ),
            )
        ]
    )
    return lyt

def detection_dashboard(
    anamgr : analysis_manager.AnalysisManager,
    plot_size = 400,
):

    # This returns a Bokeh application that takes a doc for displaying
    def app(doc):

        lyt = detection_layout(anamgr, plot_size)

        doc.title = "PUBLIC-WIFI Dashboard"
        doc.add_root(lyt)
    return app

def linked_multiindex_display(
        scrollers : dict
):
    """
    Link scrollers with the same names in two multiindex scrollers
    """
    def app(doc):
        link_multiindex_scrollers(scrollers)
        lyt = bklyts.layout(
            [
                bklyts.row(
                    *[scrollers[k] for k in scrollers.keys()],
                )
            ]
        )
        doc.add_root(lyt)
    return app
