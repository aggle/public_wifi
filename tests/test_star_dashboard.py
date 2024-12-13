# run with `bokeh serve test_star_dashboard.py --show` from a terminal
import pandas as pd
from public_wifi import starclass as sc
from public_wifi.utils import star_dashboard as sd
from bokeh.server.server import Server

catalog_file = sc.Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc.csv")
data_folder = sc.Path("/Users/jaguilar/Projects/Research/hst17167-ffp/data/HST/")

dtypes = {
    'target': str,
    'file': str,
    'filter': str,
    'ra': float,
    'dec': float,
    'x': float,
    'y': float,
    'mag_aper': float,
    'e_mag_aper': float,
    'dist': float,
    'snr': float,
}
catalog = pd.read_csv(str(catalog_file), dtype=dtypes)
catalog['x'] = catalog['x'] - 1
catalog['y'] = catalog['y'] - 1

# reduce catalog for quicker testing
catalog = catalog.query(f"target in {list(catalog['target'].unique()[:10])}")

print("processing catalog")
# set the SIM threshold low to make sure you have enough references
stars = sc.process_stars(catalog, 'target', ['filter'], data_folder, 11, sim_thresh=-1.0)
print("displaying dashboard")
dash = sd.all_stars_dashboard(stars, plot_size=350)

if __name__ == "__main__":
    port = 5006
    apps = {'/': dash}
    server = Server(apps, port=port)
    print(f'\nOpening Bokeh application on http://localhost:{port}/\n')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

