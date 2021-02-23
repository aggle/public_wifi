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



def test_lookup_from_id():
    """
    Test that you get the right subset of a table
    """
    ps_table = table_utils.load_table('point_sources')


# # generate some fake data
# @pytest.mark.parametrize()
# def test_write_read_update_tables(tmpdir, table_name, table, pk_name):
#     """
#     Test that you can write, read, and update tables
#     """
#     tables = pd.Data
#     test_file = tmpdir.join("test_db.hdf5")

#     table_utils.write_table(table_name, table, test_file)
#     # check that the table exists
#     with h5py.File(test_file, mode='r') as f:
#         # make sure the group was written
#         assert table_name in f.keys()
#         # make sure the columns were written
#         assert all([i in f['/'+table_name].keys() for i in table.columns])
#         assert all([len(f['/'+table_name + '/' + i][...]) > 0 for i in table.columns])

#     # now update the table
#     # choose 2 columns at random, not the primary key
#     columns = table.columns.drop(pk_name)
#     cols = random.choices(columns, k=2)
#     dtypes = [type(table[col].values[0]) for col in cols]
#     new_vals = [dtype(np.random.rand()) for dtype in dtypes]
#     # choose 2 rows
#     rows = sorted(random.choices(table[pk_name].values, k=2))
#     for col, val in zip(cols, new_vals):
#         table_utils.update_table(table_name, pk_name, rows, col, val, table_file=test_file)
#     # now assert that the new table has the new values
#     new_table = table_utils.load_table(table_name, test_file)

#     assert all(new_table.set_index(pk_name).loc[rows, cols] == new_vals)
