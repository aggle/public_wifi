import pytest

from bokeh.server.server import Server

import pandas as pd
from public_wifi.dashboard import star_dashboard as sd
from public_wifi.dashboard import dash_tools as dt

def show_widget(widget, port=5006):
    app = dt.standalone(widget)
    apps = {'/': app}
    server = Server(apps, port=port)
    print(f'\nOpening Bokeh application on http://localhost:{port}/\n')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
    return

def test_make_cds_table_candidates(star_with_candidates):
    """Display """
    star = star_with_candidates
    print(star.star_id)
    candidates = star.results.set_index("filter")['snr_candidates']
    df = pd.concat(candidates.to_dict(), names=['filter', 'pix_id'])#.reset_index('filter')
    df = df.reset_index().drop(columns="pix_id").sort_values(by=['filter', 'cand_id'])
    source = sd.bkmdls.ColumnDataSource(df)
    cds, table = sd.make_table_cds(df)
    print(df)
    print(table)
    # show_widget(table)
    return

def test_make_jackknife_cds(star_with_candidates):
    star = star_with_candidates
    jackknife = star.results.loc[1, 'klip_jackknife']
    cds = dt.series_to_CDS(
        jackknife,
        None,
        index = [f"{i[0]} / {i[1]}" for i in jackknife.index],
    )
    # for k in cds.data.keys():
    #     print(k)
    #     print(cds.data[k])
    print(cds.data['img'][0].shape)
    print(cds.data['cube'][0].shape)
    print(len(cds.data['index'][0]))
    return

def test_make_row_cds(nonrandom_processed_star):
    star = nonrandom_processed_star
    print(f"Star: {star.star_id}")

    return

