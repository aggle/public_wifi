"""
This module contains tests for tr14.utils.shared_utils.py
"""
import pytest
from pathlib import Path
import random

import tr14.utils.header_utils as header_utils

@pytest.mark.parametrize(['num', 'test_flags'],
                         [(1,  [1]),
                          (4,  [4]),
                          (16+256, [16, 256])
                          (sum([2**i for i in range(0,15)]),
                           [2**i for i in range(0,15)])]
                         )
def test_parse_dq_binword(num, numstr, test_flags):
    """
    Test translating integers to binary words
    Given a number, its binary string, and the binary components,
    When you apply the integer parser to the number,
    It should give you back the binary components
    """
    flags = header_utils.parse_dq_binword(int(num))
    assert(sorted(flags) == sorted(test_flags))




