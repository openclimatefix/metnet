import numpy as np
import xarray as xr
import pytest

from metnet.data.merge_single_band import (
    merge_two_arrays,
    merge_086um_band,
    get_band_data,
)


# ---------------------------------------------------------
# Helpers: create synthetic lat/lon grids
# ---------------------------------------------------------


def make_lat_lon_grid(shape=(4, 4)):
    """Create a simple synthetic lat/lon grid for gaussian tests."""
    h, w = shape
    lat = xr.DataArray(
        np.linspace(-10, 10, h).reshape(h, 1).repeat(w, axis=1),
        dims=("y", "x"),
    )
    lon = xr.DataArray(
        np.linspace(30, 50, w).reshape(1, w).repeat(h, axis=0),
        dims=("y", "x"),
    )
    return lat, lon


# ---------------------------------------------------------
# get_band_data tests
# ---------------------------------------------------------


def test_get_band_data_latest():
    times = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64")
    da = xr.DataArray([1, 2], dims=["time"], coords={"time": times})
    ds = xr.Dataset({"VI008": da})

    out = get_band_data(ds, "VI008", time=None)
    assert out.item() == 2


def test_get_band_data_exact_match_str_timestamp():
    times = np.array(["2020-01-01T00:00", "2020-01-01T01:00"], dtype="datetime64")
    da = xr.DataArray([1, 2], dims=["time"], coords={"time": times})
    ds = xr.Dataset({"VI008": da})

    out = get_band_data(ds, "VI008", time="2020-01-01T00:00")
    assert out.item() == 1


def test_get_band_data_missing_band_returns_none():
    ds = xr.Dataset({})
    assert get_band_data(ds, "C03") is None


# ---------------------------------------------------------
# merge_two_arrays: gaussian merge
# ---------------------------------------------------------


def test_merge_two_arrays_gaussian_basic():
    a = xr.DataArray(np.array([[10.0, 20.0]]), dims=["y", "x"])
    b = xr.DataArray(np.array([[30.0, 40.0]]), dims=["y", "x"])
    lat, lon = make_lat_lon_grid(shape=(1, 2))

    subpoints = {
        "gk2a": (0.0, 128.2),
        "goes_east": (0.0, -75.2),
    }

    out = merge_two_arrays(
        a,
        b,
        method="gaussian",
        lat=lat,
        lon=lon,
        sat_name_a="gk2a",
        sat_name_b="goes_east",
        sigma_deg=20.0,
        subpoints=subpoints,
    )

    # Outputs should fall between min(a,b) and max(a,b)
    assert np.all(out.values >= 10.0)
    assert np.all(out.values <= 40.0)


# ---------------------------------------------------------
# merge_086um_band: missing-band behavior
# ---------------------------------------------------------


def test_merge_086um_band_missing_one_band(monkeypatch):
    """When one satellite lacks the band, should return the other unchanged."""

    ds_gk = xr.Dataset({"VI008": xr.DataArray([1.0], dims=["time"])})
    ds_go = xr.Dataset({})  # missing C03

    # Patch icechunk loading
    monkeypatch.setattr(
        "metnet.data.merge_single_band.open_icechunk_store",
        lambda prefix: ds_gk if "gk2a" in prefix else ds_go,
    )

    # Patch get_band_data: return the dataset's band if it exists
    monkeypatch.setattr(
        "metnet.data.merge_single_band.get_band_data",
        lambda ds, band, time=None: ds.get(band, None),
    )

    merged, meta = merge_086um_band(time=None)

    assert float(merged.values.item()) == 1.0
    assert meta["method"] == "mean"


def test_merge_086um_band_both_missing(monkeypatch):
    """Should raise when both satellites lack the 0.86 μm band."""

    empty = xr.Dataset({})

    monkeypatch.setattr(
        "metnet.data.merge_single_band.open_icechunk_store",
        lambda prefix: empty,
    )
    monkeypatch.setattr(
        "metnet.data.merge_single_band.get_band_data",
        lambda ds, band, time=None: None,
    )

    with pytest.raises(RuntimeError):
        merge_086um_band(time=None)
