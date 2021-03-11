
from cudf._lib.binning import bin
from cudf.core.column import as_column
import cupy


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
    side = "left" if right else "right"
    ids = cupy.searchsorted(bins, x, side=side)
    adjust = lambda x: x - 10 ** (-3)
    breaks = [b for b in bins]
    breaks[0] = adjust(breaks[0])
    #get the left and right edges of the bins as columns 
    left_edges = as_column(breaks[:-1:])
    right_edges = as_column(breaks[+1::])

    if not right:
        right_inclusive = False
    
    if include_lowest:
        left_inclusive = True

    labels = bin(input_arr,left_edges, left_inclusive,right_edges,right_inclusive)
    return labels 
