"""
Tests for utils.subtr_utils
"""

import pytest
from pathlib import Path

import random
from tr14.utils import subtr_utils


@pytest.mark.skip("Not used for anything right now")
def test_subtr_utils(subtr_mgr):
    """I don't know what I'm doing so figure this out here"""
    pass

class TestNMF():
    """Suite of tests for NMF subtraction"""
    #self.subtr_mgr = subtr_mgr

    def test_subtr_mgr_seq_nmf(self, subtr_mgr):
        """Test sequential NMF"""
        targ_id, targ_stamp = subtr_mgr.db.stamps_tab.iloc[100][['stamp_id', 'stamp_array']]
        ref_stamps = subtr_mgr.get_reference_stamps(targ_id, dmag_max=1)
        residuals, models = subtr_mgr.subtr_nmf(targ_stamp, ref_stamps)
        #assert (residuals is not None) and (models is not None)
        assert subtr_utils.np.stack(residuals).shape == subtr_utils.np.stack(models).shape
