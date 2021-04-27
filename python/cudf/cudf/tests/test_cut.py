# Copyright (c) 2018-2021, NVIDIA CORPORATION.

"""
Test related to Index
"""

import numpy as np
import pandas as pd
from cudf.core.cut import cut
import pytest
from cudf.tests.utils import assert_eq

@pytest.mark.parametrize("x", [[1, 7, 5, 4, 6, 3], [1,7]])
@pytest.mark.parametrize("bins", [1,2,3])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("precision", [3])
def test_cut_basic(x, bins, right, include_lowest, ordered, precision):

    pcat = pd.cut(
        x = x,
        bins = bins,
        right = right,
        precision = precision,
        include_lowest = include_lowest,
        ordered = ordered
    )
    pindex= pd.CategoricalIndex(pcat)
    gindex = cut(
        x = x,
        bins = bins,
        right = right,
        precision = precision,
        include_lowest = include_lowest,
        ordered = ordered
    )

    assert_eq(pindex, gindex)