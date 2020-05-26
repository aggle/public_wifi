"""
Includes shared useful stuff, like path definitions
"""

from pathlib import Path

"""
This block defines some useful paths
"""
# HST data files for manipulation
data_path = (Path(__file__).parent.absolute() / "../../data/my_data/").resolve()
# Database tables
table_path = (Path(__file__).parent.absolute() / "../../data/tables/").resolve()
# Gaia catalog and source matches
align_path = (Path(__file__).parent.absolute() / "../../data/align_catalog/").resolve()
# KS2 output files
ks2_path = (Path(__file__).parent.absolute() / "../../data/ks2/").resolve()
