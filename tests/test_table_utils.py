"""
Tests for table_utils
"""

import pytest
from pathlib import Path


import random
import tr14.utils.table_utils as table_utils


def test_index_into_table():
    """
    test table_utils.index_into_table()
    Given a list of stars,
    When you ask for rows containing those stars from a table,
    Then the return table should contain rows with all of the requested stars and no more
    """
    star_table = table_utils.load_table("stars")
    # repeat the same entry 3 times
    list_of_stars = (star_table.iloc[0]['star_id'],) * 3
    table = table_utils.index_into_table(star_table, 'star_id', list_of_stars)
    # test length
    assert(len(table) == len(list_of_stars))
    # test all elements
    assert(all(table['star_id'].isin(list_of_stars)))
    assert(all(table_utils.pd.Series(list_of_stars).isin(table['star_id'])))


def test_create_database_subset():
    """
    Given a list of stars
    When a database subset is created
    Then all the database tables should only contain data related to the subset
      of stars. 
    """
    master_stars = table_utils.load_table("stars")
    # choose some random stars
    nstars = 100
    star_subset = list(table_utils.np.random.choice(master_stars['star_id'],
                                                    size=nstars))
    # make the database subset
    table_subset = table_utils.create_database_subset(star_subset,
                                                      #['stars', 'point_sources', 'stamps']
    )
    # now, make sure you have all the stars and only the stars in the subset
    for k, table in table_subset.items():
        print(f"Testing {k}...")
        star_col = table_utils.find_star_id_col(table.columns)
        # check that each set is a subset of the other
        assert(set(table[star_col]).issubset(set(star_subset)))
        assert(set(star_subset).issubset(set(table[star_col])))


