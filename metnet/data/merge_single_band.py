"""
Merge matching bands from multiple geostationary satellites.

Implements wavelength-based band matching across satellites as described in
the MetNet Global paper. This is a minimal proof-of-concept focusing on the
0.86 µm band from GK2A and GOES-East.

TODO: Add proper reprojection and azimuth-based blending.
"""

from datetime import datetime
from typing import Mapping, Optional, Union

import icechunk
import numpy as np
import xarray as xr


def open_icechunk_store(prefix: str):
    """Open an icechunk store (default points to Source Cooperative)."""
    storage = icechunk.s3_storage(
        bucket="bkr",
        prefix=prefix,
        region="us-east-1",
        endpoint_url="https://data.source.coop",
        anonymous=True,
        force_path_style=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch="main")
    return xr.open_zarr(session.store, consolidated=False)


def get_band_data(
    ds,
    band_name: str,
    time: Optional[Union[datetime, str]] = None,
):
    """
    Return one timestep of a band.

    Behaviour:
      - If `time` is None: return the most recent timestep.
      - If `time` is str/datetime: require an exact match on the `time` coordinate.

    Notes:
      - Returns `None` if `band_name` is not present in `ds`.
      - Raises `KeyError` if `time` is provided and the exact timestamp is not found.
    """
    if band_name not in ds.data_vars:
        return None

    var = ds[band_name]
    if time is None:
        return var.isel(time=-1)
    return var.sel(time=time)


def merge_two_arrays(
    satellite_one: xr.DataArray,
    satellite_two: xr.DataArray,
    method: str = "mean",
    *,
    name: Optional[str] = None,
) -> xr.DataArray:
    """
    Merge two DataArrays on a common grid by aligning their coordinates.

    This is pure (no I/O), so it can be unit-tested in isolation.

    Behaviour:
      - aligns both inputs by coords (outer join preserves union of grids)
      - if both pixels present → combine (default: mean)
      - if only one present → use that single value

    Args:
      satellite_one: first DataArray
      satellite_two: second DataArray
      method: merge strategy ("mean" or "first")
      name: name to assign to the merged DataArray (optional)

    Returns:
      A new DataArray containing merged values.
    """

    # Align to a common set of coords (outer join preserves union)
    a, b = xr.align(satellite_one, satellite_two, join="outer")

    if method == "mean":
        both = a.notnull() & b.notnull()
        merged = xr.where(both, 0.5 * (a + b), xr.where(a.notnull(), a, b))
    else:
        # Fallback: prefer first where available, else second
        merged = a.combine_first(b)

    merged = merged.copy()
    merged.name = name or satellite_one.name or satellite_two.name or "merged"
    return merged


def merge_086um_band(
    time: datetime | str | None = None,
    *,
    stores: Optional[Mapping[str, str]] = None,
):
    """
    Merge 0.86 µm band from GK2A and GOES-East satellites.

    Band matching:
      - GK2A VI008 ~ 0.863 µm
      - GOES-East C03 ~ 0.865 µm

    Args:
      time: timestamp to select (if None, uses latest available)
      stores: mapping of satellite→store prefix. Defaults to Source Coop demo
        stores.

    Example:
          {
            "gk2a": "geo/gk2a_1000m.icechunk",
            "goes_east": "geo/goes-east_1000m.icechunk",
          }

    Returns:
      (merged_dataarray, metadata_dict)
    """

    if stores is None:
        stores = {
            "gk2a": "geo/gk2a_1000m.icechunk",
            "goes_east": "geo/goes-east_1000m.icechunk",
        }

    # Open both stores (allow caller to override S3/endpoint kwargs)
    ds_gk2a = open_icechunk_store(stores["gk2a"])  # kwargs baked into helper
    ds_goes = open_icechunk_store(stores["goes_east"])  # same here

    # Select a single timestep for each using helper (prefers latest if time is None)
    band_gk2a = get_band_data(ds_gk2a, "VI008", time)
    band_goes = get_band_data(ds_goes, "C03", time)

    merged = merge_two_arrays(band_gk2a, band_goes, method="mean", name="reflectance_086um")

    metadata = {
        "wavelength": "0.86 µm",
        "satellites": ["GK2A", "GOES-East"],
        "bands": {"GK2A": "VI008", "GOES-East": "C03"},
        "method": "mean",
    }

    return merged, metadata


if __name__ == "__main__":
    merged_data, metadata = merge_086um_band(time=None)
    print(f"Merged {metadata['wavelength']} from {metadata['satellites']}")
    valid_pixels = int(np.sum(~np.isnan(merged_data.values)))
    print(f"Shape: {merged_data.shape}, " f"Valid pixels: {valid_pixels}")
