"""
Includes shared useful stuff, like path definitions
"""

from pathlib import Path

"""
This block defines some useful paths
"""
# HST data files for manipulation
data_path = Path(__file__).parent.absolute() / "../../data/my_data/"
# Database tables
table_path = Path(__file__).parent.absolute() / "../../data/tables/"
# Gaia catalog and source matches
align_path = Path(__file__).parent.absolute() / "../../data/align_catalog/"
# KS2 output files
ks2_path = Path(__file__).parent.absolute() / "../../data/ks2/"
