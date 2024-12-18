# run with `bokeh serve test_star_dashboard.py --show` from a terminal
import pandas as pd
from public_wifi import starclass as sc
from public_wifi.utils import star_dashboard as sd
from public_wifi import catalog_processing as catproc
from bokeh.server.server import Server

catalog_file = sc.Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc.csv")
data_folder = sc.Path("/Users/jaguilar/Projects/Research/hst17167-ffp/data/HST/")

catalog = catproc.load_catalog(catalog_file, 100)

# reduce catalog for quicker testing
targets = list(catalog['target'].unique())
catalog = catalog.query(f"target in {targets}")




print("processing catalog")
# set the SIM threshold low to make sure you have enough references
stars = catproc.process_catalog(
    catalog,
    'target',
    ['filter'],
    data_folder,
    11,
    sim_thresh=-1.0,
    scale_stamps=True
)
print("displaying dashboard")
dash = sd.all_stars_dashboard(stars, plot_size=350)




if __name__ == "__main__":
    port = 5006
    apps = {'/': dash}
    server = Server(apps, port=port)
    print(f'\nOpening Bokeh application on http://localhost:{port}/\n')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

