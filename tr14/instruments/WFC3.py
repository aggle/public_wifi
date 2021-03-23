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


class WFC3Class(Instrument):
    """
    Abstract class for instruments with the required fields that need to be implemented.
    Currently used to store instrument properties that affect the analysis, like detector
    width in pixels, plate scale, and filter information.
    """

    def __init__(self):
        """
        Initialize an instrument from the config file
        """
        file_name = "wfc3.yaml"
        ## read in configuration file and set these static variables
        package_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(package_directory, file_name)
        with open(config_file) as f:
            yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
            # assign all the parts of the config file directly
            for k, v in yaml_dict.items():
                setattr(self, k, v)

        # modify any required variables
        self.pix_scale = units.Quantity(self.pix_scale, unit=self.pix_scale_unit)
