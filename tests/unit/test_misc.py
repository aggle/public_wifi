import pytest
import numpy as np
from public_wifi import misc


@pytest.mark.parametrize("stamp", [
    11,
    np.ones((11, 11), dtype=float),
    misc.pd.Series({i: np.ones((11, 11)) for i in range(5)})
])
def test_get_stamp_center(stamp):
    center = misc.get_stamp_center(stamp)
    assert((center == np.array([5, 5])).all())

def test_coordinate_conversion():
    stamp_size = 15
    pos = (-3, -3)
    center = misc.get_stamp_center(stamp_size)
    ll_pos = misc.center_to_ll_coords(stamp_size, pos)
    center_pos = misc.ll_to_center_coords(stamp_size, ll_pos)
    assert(all(center_pos == pos))
