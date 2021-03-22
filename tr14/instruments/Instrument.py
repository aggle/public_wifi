"""
Create an instrument object from a config file.
Structure, and some code, shamelessly stolen from pyklip (pyklip.readthedocs.io)
Author: Jonathan Aguilar
Created: 2021-03-10
"""

import abc
import configparser


class Instrument(object):
    """
    Abstract class for instruments with the required fields that need to be implemented.
    Currently used to store instrument properties that affect the analysis, like detector
    width in pixels, plate scale, and filter information.
    """

    def __init__(self, config_file):
        """
        Initialize an instrument from the config file
        """
        self.confpar = configparser.ConfigParser()
        with open(config_file) as f:
            confpar.read_file(f)

        
