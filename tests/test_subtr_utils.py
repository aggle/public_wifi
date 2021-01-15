"""
Tests for utils.subtr_utils
"""

import pytest
from pathlib import Path

import random
from tr14.utils import subtr_utils


# common resources
db_file = subtr_utils.shared_utils.db_clean_file
db_mast = subtr_utils.db_manager.DBManager(db_file)

