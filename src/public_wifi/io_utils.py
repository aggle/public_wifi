"""
Useful tools
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits



# This dictionary contains descriptions of the residual columns
column_descriptions = {
	'klip_basis' : 'Ordered KLIP basis images',
	'klip_sub' : 'Image - KLIP PSF model',
	'klip_model' : 'PSF model constructed from successive KLIP basis vectors',
	'snrmap' : 'klip_sub divided by std(klip_sub)',
	'mf' : 'A matched filter used for detecting point sources (the PSF)',
	'mf_prim_flux' : 'The flux of the primary, as measured by a matched filter',
	'mf_norm' : 'The norm of the matched filter',
	'mf_map' : 'klip_sub convolved with the matched filter',
	'detmap' : 'mf_map, normalized by mf_norm',
	'detpos' : 'The index of the brightest pixel in detmap',
	'detmap_posflux' : 'The value of detmap at detpos',
	'pca_bias' : 'Bias introduced by the klip basis at a particular pixel',
	'fluxmap' : 'detmap, corrected for PCA bias',
	'contrastmap' : 'fluxmap divided by the flux of the primary',
	'fluxmap_posflux' : 'The flux at detpos',
	'contrastmap_posflux' : 'The contrast at detpos',
}

def write_star_results(
        results_df : pd.DataFrame,
        outfile : str | Path,
) -> None :
    """
    Write the results dataframe to a fits file

    Parameters
    ----------
    results_df : pd.DataFrame
      A dataframe indexed by catalog row and Kklip
    outfile : str | Path
      Full path to the output file

    Output
    ------
    None
    Writes file to disk at the specified path

    """
    pass

