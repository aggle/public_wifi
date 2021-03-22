"""
Create an instrument object from a config file
Author: Jonathan Aguilar
Created: 2021-03-10
"""

import os
import configparser

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
        ## read in configuration file and set these static variables
        package_directory = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(package_directory, "wfc3.cfg")
        self.config = configparser.ConfigParser()
        with open(config_file) as f:
            self.config.read_file(f)

        self.npix_x = self.config.getint("Properties", "npix_x")
        self.npix_y = self.config.getint("Properties", "npix_y")
        self.stamp_size = self.config.getint("Properties", "stamp_size")
