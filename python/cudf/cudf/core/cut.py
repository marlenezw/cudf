# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf._lib.label_bins import bin
from cudf.core.column import as_column
import cupy
import pandas as pd
import numpy as np


def cut( x,
        bins,
        right: bool = True,
        labels=None,
        retbins: bool = False,
        precision: int = 3,
        include_lowest: bool = False,
        duplicates: str = "raise",
        ordered: bool = True):

    """
    Bin that follows cudf cut 
    """
    left_inclusive = False
    right_inclusive = True

    #the inputs is a column of the values in the array x
    input_arr = as_column(x)

    #create the bins 
    x = cupy.asarray(x)
    sz = x.size
    rng = (x.min(), x.max())
    mn, mx = [mi + 0.0 for mi in rng]
    bins = cupy.linspace(mn, mx, bins + 1, endpoint=True)
    adj = (mx - mn) * 0.001
    adjust = lambda x: x - 10 ** (-3)
    breaks = [float(b) for b in bins]
    breaks[0] = adjust(breaks[0])
    interval_labels = pd.IntervalIndex.from_breaks(breaks)
    bins[0] = adjust(bins[0])
    #get the left and right edges of the bins as columns 
    left_edges = as_column(bins[:-1:])
    right_edges = as_column(bins[+1::])
    #the input arr must be changed to the same type as the edges
    input_arr = input_arr.astype(left_edges._dtype)
    #checking for the correct inclusivity values
    if not right:
        right_inclusive = False
    if include_lowest:
        left_inclusive = True
    labels = bin(input_arr,left_edges, left_inclusive,right_edges,right_inclusive)
    fin_labels = [interval_labels[labels[i]] for i in range(len(labels))]
    breakpoint()
    return fin_labels 
