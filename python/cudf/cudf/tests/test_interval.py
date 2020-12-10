# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


# @pytest.mark.parametrize(
#     "data",
#     [1.0, 2.0
#     ]
# )

def test_create_struct_series():
    expect = pd.Series(pd.Interval(1,4), dtype='interval')
    got = cudf.Series(pd.Interval(1,4), dtype='interval')
    assert_eq(expect, got)
