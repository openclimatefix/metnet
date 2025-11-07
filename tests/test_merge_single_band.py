import sys
from pathlib import Path

import numpy as np
import xarray as xr

from metnet.data.merge_single_band import merge_two_arrays


def test_merge_two_arrays_single_present():
    gk = xr.DataArray([1.0, np.nan], dims=["x"])
    go = xr.DataArray([np.nan, 3.0], dims=["x"])
    out = merge_two_arrays(gk, go)
    assert float(out[0]) == 1.0
    assert float(out[1]) == 3.0


def test_merge_two_arrays_mean_both_present():
    gk = xr.DataArray([2.0], dims=["x"])
    go = xr.DataArray([4.0], dims=["x"])
    out = merge_two_arrays(gk, go)
    assert float(out[0]) == 3.0