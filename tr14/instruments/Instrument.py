"""
Create an instrument object from a config file.
Structure, and some code, shamelessly stolen from pyklip (pyklip.readthedocs.io)
Author: Jonathan Aguilar
Created: 2021-03-10
"""

import os
import abc
import yaml
from collections import defaultdict

class Instrument(object):
    """
    Abstract class for instruments with the required fields that need to be implemented.
    Currently used to store instrument properties that affect the analysis, like detector
    width in pixels, plate scale, and filter information.
    """

    def __init__(self, config_file):
        """
        Initialize an instrument from the config file. By default, assigns
        all top-level entries from the config file as attributes.
        Overwrite in the class-specific implementation if you need something
        different.
        """
        ## read in configuration file and set these static variables
        package_directory = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(package_directory, config_file)
        with open(full_path) as f:
            yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
            # assign all the parts of the config file directly
            for k, v in yaml_dict.items():
                setattr(self, k, v)

        self.filters = {}

    def load_filter(self, filter_file):
        """Add a filter entry to the self.filters dict"""
        with open(filter_file, 'r') as f:
            filt_dict = yaml.load(f, Loader=yaml.SafeLoader)
        name = filt_dict.pop("name")
        self.filters[name] = filt_dict
