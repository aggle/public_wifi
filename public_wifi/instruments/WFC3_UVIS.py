"""
Create an instrument object from a config file
Author: Jonathan Aguilar
Created: 2021-03-10
"""

import os
import yaml
import configparser
from astropy import units

from .Instrument import Instrument


class WFC3UVISClass(Instrument):
    """
    Abstract class for instruments with the required fields that need to be implemented.
    Currently used to store instrument properties that affect the analysis, like detector
    width in pixels, plate scale, and filter information.
    """

    def __init__(self):
        """
        Initialize an instrument from the config file
        """
        config_file = "wfc3_uvis.yaml"
        ## read in configuration file and set these static variables
        #package_directory = os.path.dirname(os.path.abspath(__file__))
        #config_file = os.path.join(package_directory, file_name)
        #with open(config_file) as f:
        #    yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
        #    # assign all the parts of the config file directly
        #    for k, v in yaml_dict.items():
        #        setattr(self, k, v)
        super().__init__(config_file)
        # modify any required variables
        self.pix_scale = units.Quantity(self.pix_scale, unit=self.pix_scale_unit)
        # add filters
        filt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../filters")
        self.load_filter(os.path.join(filt_path, 'F814W.yaml'))
        self.load_filter(os.path.join(filt_path, 'F850LP.yaml'))
