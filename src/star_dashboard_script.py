# run with `bokeh serve test_star_dashboard.py --show` from a terminal
import pandas as pd

from public_wifi import starclass as sc
from public_wifi.dashboard import star_dashboard as sd
from public_wifi import catalog_processing as catproc

from bokeh.server.server import Server

catalog_file = sc.Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc.csv")
data_folder = sc.Path("~/Projects/Research/hst17167-ffp/data/HST/")

catalog = catproc.load_catalog(catalog_file, 100)

# reduce catalog for quicker testing
targets = list(catalog['target'].unique())
catalog = catalog.query(f"target in {targets}")




print("processing catalog")
# set the SIM threshold low to make sure you have enough references
stars = catproc.process_catalog(
    input_catalog = catalog,
    star_id_column = 'target',
    match_references_on = ['filter'],
    data_folder = data_folder,
    stamp_size = 11,
    bad_references = ['J042705.86+261520.3'],
    scale_stamps = False,
    center_stamps = False,
    min_nref = 5,
    sim_thresh = 0.5,
    snr_thresh = 5.,
    n_modes = 5,
)
print("displaying dashboard")
dash = sd.all_stars_dashboard(stars, plot_size=350)



if __name__ == "__main__":
    port = 5007
    apps = {'/': dash}
    server = Server(apps, port=port)
    print(f'\nOpening Bokeh application on http://localhost:{port}/\n')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

