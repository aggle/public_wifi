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
    
    #@pytest.mark.
    def test_nmf_default(self, subtr_mgr):
        """Test NMF works with default arguments"""
        targ_id, targ_stamp = subtr_mgr.db.stamps_tab.iloc[100][['stamp_id', 'stamp_array']]
        ref_stamps = subtr_mgr.get_reference_stamps(targ_id, dmag_max=1)
        residuals, models = subtr_mgr.subtr_nmf(targ_stamp, ref_stamps)
        assert subtr_utils.np.stack(residuals).shape == subtr_utils.np.stack(models).shape

    #@pytest.mark.skip("Not yet implemented")
    def test_nmf_sequential(self, subtr_mgr):
        """Test ordered NMF"""
        targ_id, targ_stamp = subtr_mgr.db.stamps_tab.iloc[100][['stamp_id', 'stamp_array']]
        ref_stamps = subtr_mgr.get_reference_stamps(targ_id, dmag_max=1)
        residuals, models = subtr_mgr.subtr_nmf(targ_stamp, ref_stamps,
                                                {'ordered': True, 'n_components': 5})
        assert subtr_utils.np.stack(residuals).shape == subtr_utils.np.stack(models).shape

    @pytest.mark.nmf_star
    def test_subtr_nmf_one_star(self, subtr_mgr):
        """Test that NMF for one star is working"""
        stars = subtr_mgr.db.stars_tab['star_id'].unique()
        star = random.choice(stars)
        subtr_utils.shared_utils.debug_print(True, star)

        n_components = 5
        kwargs = {'ordered': True, 'n_components': n_components}
        subtr_results = subtr_mgr.subtr_nmf_one_star(star, kwargs=kwargs)

        # the results should have attributes residuals, models, and references
        result_attrs = ['residuals', 'models', 'references']
        for attr in result_attrs:
            assert hasattr(subtr_results, attr) == True
        # the results should have dimension: n_stamps x n_components
        n_stamps = len(subtr_mgr.db.find_matching_id(star, 'stamp'))

        assert subtr_results.models.shape == (n_stamps, n_components)
        assert subtr_results.residuals.shape == (n_stamps, n_components)

    @pytest.mark.nmf_star
    def test_subtr_nmf_by_star(self, subtr_mgr):
        """Test that NMF for one star is working"""
        n_components = 3
        kwargs = {'ordered': True, 'n_components': n_components}
        subtr_results = subtr_mgr.subtr_nmf_by_star(nmf_args=kwargs)

        # the results should have attributes residuals, models, and references
        result_attrs = ['residuals', 'models', 'references']
        for attr in result_attrs:
            assert hasattr(subtr_results, attr) == True
        # the results should have dimension: n_stamps x n_components
        n_stamps = len(subtr_mgr.db.stamps_tab)

        assert subtr_results.models.shape == (n_stamps, n_components)
        assert subtr_results.residuals.shape == (n_stamps, n_components)
