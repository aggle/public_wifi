"""
Includes shared useful stuff, like path definitions
"""

from pathlib import Path

# Define some useful paths here

# HST data files for manipulation
data_path = Path(__file__).parent / "../../data/my_data/"
# Database tables
table_path = Path(__file__).parent / "../../data/tables/"
# Gaia catalog and source matches
align_path = Path(__file__).parent / "../../data/align_catalog"
