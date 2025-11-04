import os
import sys

# ensure project root is on sys.path so metnet imports work without editable install
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import xarray as xr
from metnet.data.merge_single_band import merge_086um


def test_merge_086um_mean():
    gk = xr.DataArray([1.0, np.nan], dims=["x"])
    go = xr.DataArray([np.nan, 3.0], dims=["x"])
    out = merge_086um(gk, go)
    assert float(out[0]) == 1.0
    assert float(out[1]) == 3.0


def test_merge_086um_mean_two_values():
    gk = xr.DataArray([2.0], dims=["x"])
    go = xr.DataArray([4.0], dims=["x"])
    out = merge_086um(gk, go)
    assert float(out[0]) == 3.0
