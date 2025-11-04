import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Try import normally first (works when package installed or editable)
try:
    from metnet.data.merge_single_band import merge_086um
except ModuleNotFoundError:
    ROOT = str(Path(__file__).resolve().parents[1])
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
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
