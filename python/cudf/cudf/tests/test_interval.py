# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "data1, data2",
    [
        (1, 1.0, 3),
        (2, 2.0, 4.0),
        
    ]
)

def test_create_struct_series(data1, data2):
    expect = pd.Series(pd.Interval(data1,data2), dtype='interval')
    got = cudf.Series(pd.Interval(data1,data2), dtype='interval')
    assert_eq(expect, got)
