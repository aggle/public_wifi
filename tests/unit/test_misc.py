import pytest
import numpy as np
from public_wifi import misc

def test_calc_stamp_center():
    stamp = np.ones((11, 11), dtype=int)
    center = misc.calc_stamp_center(stamp)
    assert(all(center == (5, 5)))

    stamp = np.ones((12, 12), dtype=int)
    center = misc.calc_stamp_center(stamp)
    assert(all(center == (6, 6)))
