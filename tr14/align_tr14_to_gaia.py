"""
V. Bajaj notebook as a python script to be run from the command line.
Run as: python align_tr14_to_gaia.py
Source location: https://spacetelescope.github.io/notebooks/notebooks/DrizzlePac/align_to_catalogs/align_to_catalogs.html
"""

import astropy.units as u
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.mast import Observations
from astroquery.sdss import SDSS

#from ccdproc import ImageFileCollection
from IPython.display import Image

from drizzlepac import tweakreg
from drizzlepac import astrodrizzle


from utils import shared_utils
from utils import header_utils
from pathlib import Path

datapath = Path("../data/align_catalog/")

prihdrs = header_utils.load_headers('pri')
scihdrs = header_utils.load_headers('sci')

RA, Dec = prihdrs[['RA_TARG','DEC_TARG']].mean()
coord = SkyCoord(ra=RA, dec=Dec, unit=(u.deg, u.deg))
radius = Quantity(6., u.arcmin)




# if you want to run over all the images, then run this module as a script
if __name__ == "__main__":
    
    gaia_query = Gaia.query_object_async(coordinate=coord, radius=radius)
    query_path = datapath / 'gaia.cat'
    # check if catalog exists; if it does, ask to rerun
    run_query = "N"
    if query_path.exists() == False:
        run_query = "Y"
    elif query_path.exists() == True:
        # if it exists, ask about running
        run_query = input(f"{query_path} exists - rerun? Y/n: ").upper()
    if run_query == "Y":
        print("Querying Gaia...")
        reduced_query = gaia_query['ra', 'dec', 'phot_g_mean_mag']
        reduced_query.write(query_path.as_posix(), format='ascii.commented_header')
        print(f"Gaia query written to {query_path.as_posix()}.\n")
    else:
        print(f"Skipping Gaia query; reading catalog from existing {query_path}")

    subarray = input("Subarray, T or F? ")

    if subarray.upper() == 'T':
        subarray = True
    else:
        subarray = False

    list_of_images = [(Path("../data/my_data/") / i).as_posix() 
                      for i in prihdrs.query(f'SUBARRAY == {subarray}')['FILENAME']]
    print(f"Processing {len(list_of_images)} images...")

    refcat = query_path.as_posix()#'gaia.cat'
    cw = 3.5  # Set to two times the FWHM of the PSF.
    wcsname = 'Gaia'  # Specify the WCS name for this alignment

    tweakreg.TweakReg(list_of_images,#'*flc.fits',  # Pass input images
                      updatehdr=False,  # update header with new WCS solution
                      imagefindcfg={'threshold':500.,'conv_width':cw},  # Detection parameters, threshold varies for different data
                      refcat=refcat,  # Use user supplied catalog (Gaia)
                      interactive=False,
                      see2dplot=False,
                      shiftfile=True,  # Save out shift file (so we can look at shifts later)
                      outshifts='Gaia_shifts.txt',  # name of the shift file
                      wcsname=wcsname,  # Give our WCS a new name
                      reusename=True,
                      sigma=2.3,
                      ylimit=0.2,
                      fitgeometry='general')  # Use the 6 parameter fit
    #tweakreg.TweakReg(list_of_images,
    #                  refcat=refcat)

    print("\nTransformations calculated. Now run TweakReg again to apply them.\n")

    tweakreg.TweakReg(list_of_images,#'*flc.fits',  # Pass input images
                      updatehdr=True,  # update header with new WCS solution
                      imagefindcfg={'threshold':500.,'conv_width':cw},  # Detection parameters, threold varies for different data
                      refcat=refcat,  # Use user supplied catalog (Gaia)
                      interactive=False,
                      see2dplot=False,
                      shiftfile=True,  # Save out shift file (so we can look at shifts later)
                      outshifts='Gaia_shifts.txt',  # name of the shift file
                      wcsname=wcsname,  # Give our WCS a new name
                      reusename=True,
                      sigma=2.3,
                      ylimit=0.2,
                      fitgeometry='general')  # Use the 6 parameter fit

