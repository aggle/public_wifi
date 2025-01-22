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
