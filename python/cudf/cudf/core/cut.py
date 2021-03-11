import numpy as np
import cupy
import pandas as pd
import cudf


def cut(
    x,
    bins,
    right: bool = True,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):

    x = cupy.asarray(x)
    sz = x.size
    rng = (x.min(), x.max())
    mn, mx = [mi + 0.0 for mi in rng]
    bins = cupy.linspace(mn, mx, bins + 1, endpoint=True)
    adj = (mx - mn) * 0.001
    side = "left" if right else "right"
    ids = cupy.searchsorted(bins, x, side=side)
    na_mask = cupy.isnan(x) | (ids == len(bins)) | (ids == 0)
    # np.putmask(ids, na_mask, 0) or cupy.putmask(ids,na_mask,0) if working
    adjust = lambda x: x - 10 ** (-3)
    breaks = [b.item() for b in bins]
    breaks[0] = adjust(breaks[0])
    labels = pd.IntervalIndex.from_breaks(breaks)
    labels = pd.Categorical(
        labels,
        categories=labels if len(set(labels)) == len(labels) else None,
        ordered=ordered,
    )

    return labels
