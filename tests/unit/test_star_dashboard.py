import pytest

from bokeh.server.server import Server

import pandas as pd
from public_wifi.utils import star_dashboard as sd

# def app(doc):
#     layout = func(*args, **kwargs)

#     doc.add_root(layout)

#     doc.theme = Theme(json=yaml.load("""
#         attrs:
#             figure:
#                 background_fill_color: white
#                 outline_line_color: white
#                 toolbar_location: above
#             Grid:
#                 grid_line_dash: [6, 4]
#                 grid_line_color: white
#     """, Loader=yaml.FullLoader))
#     return app

def show_widget(widget, port=5006):
    app = sd.standalone(widget)
    apps = {'/': app}
    server = Server(apps, port=port)
    print(f'\nOpening Bokeh application on http://localhost:{port}/\n')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

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

