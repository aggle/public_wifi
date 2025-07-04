# run with `bokeh serve test_star_dashboard.py --show` from a terminal
import time
import pandas as pd

from public_wifi import starclass as sc
from public_wifi.dashboard import star_dashboard as sd
from public_wifi import catalog_processing as catproc
from public_wifi.analysis_manager import AnalysisManager

from bokeh.server.server import Server

# catalog_file = sc.Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc.csv")
catalog_file = sc.Path("~/Projects/Research/hst17167-ffp/catalogs/targets_drc-low_snr.csv")
data_folder = sc.Path("~/Projects/Research/hst17167-ffp/data/HST/")

catalog = catproc.load_catalog(catalog_file, 50)


# load file with list of bad references
# with open("/Users/jaguilar/Projects/Research/hst17167-ffp/ffp_tools/src/ffp_tools/bad_references.txt") as f:
#     bad_references = [i.strip() for i in f.readlines()]
bad_references = ['J042705.86+261520.3']


nobs = len(catalog)
nstars = len(catalog['target'].unique())
print(f"Processing catalog: {nobs} observations of {nstars} stars...")


def make_analysis_manager():
    anamgr = AnalysisManager(
        input_catalog = catalog,
        star_id_column = 'target',
        match_references_on = ['filter'],
        data_folder = data_folder,
        stamp_size = 21,
        bad_references = bad_references,
        scale_stamps = False,
        center_stamps = False,
        min_nref = 20,
        sim_thresh = 0.5,
        snr_thresh = 5.,
        n_modes = 5,
        cat_det_kklip=10,
        mf_width=15
    )
    return anamgr



if __name__ == "__main__":
    t0 = time.time()
    anamgr = make_analysis_manager()
    t1 = time.time()
    print(
        f"Elapsed time to instantiate AnalysisManager object: {t1-t0:0f} seconds."
    )
    dash = sd.all_stars_dashboard(anamgr, plot_size=350)
    print("\nDisplaying dashboard\n")
    port = 5007
    apps = {'/': dash}
    server = Server(apps, port=port)
    print(f'\nOpening Bokeh application on http://localhost:{port}/\n')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

