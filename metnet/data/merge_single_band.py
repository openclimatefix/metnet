"""
Merge matching bands from multiple geostationary satellites.

Implements wavelength-based band matching across satellites as described in
the MetNet Global paper. This is a minimal proof-of-concept focusing on the
0.86 µm band from GK2A and GOES-East.

TODO: Add proper reprojection and azimuth-based blending.
"""

from datetime import datetime

import icechunk
import numpy as np
import xarray as xr
from typing import Mapping, Optional, Union


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
    """Return one timestep of a band.

    If `time` is None -> use most recent. If str/datetime -> nearest match.
    """
    if band_name not in ds.data_vars:
        return None

    var = ds[band_name]
    if time is None:
        return var.isel(time=-1)
    return var.sel(time=time, method="nearest")


def merge_086um(
    da_gk2a: xr.DataArray,
    da_goes_east: xr.DataArray,
    method: str = "mean",
) -> xr.DataArray:
    """Merge two 0.86 µm reflectance DataArrays on a common grid.

    This is *pure* (no I/O), so it can be unit-tested in isolation.

    Behaviour:
      - aligns by coords (outer join)
      - if both pixels present → combine (default: mean)
      - if only one present → use that value
    """

    # Align to a common set of coords (outer join preserves union)
    a, b = xr.align(da_gk2a, da_goes_east, join="outer")

    if method == "mean":
        both = a.notnull() & b.notnull()
        merged = xr.where(
            both,
            0.5 * (a + b),
            xr.where(a.notnull(), a, b),
        )
    else:
        # Fallback: prefer GK2A where available, else GOES-East
        merged = a.combine_first(b)

    merged = merged.copy()
    merged.name = "reflectance_086um"
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
      stores: mapping of satellite→store prefix. Defaults to Source Coop demo stores.
              Example: {"gk2a": "geo/gk2a_1000m.icechunk", "goes_east": "geo/goes-east_1000m.icechunk"}

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

    merged = merge_086um(band_gk2a, band_goes, method="mean")

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
    print(
        f"Shape: {merged_data.shape}, Valid pixels: {np.sum(~np.isnan(merged_data.values))}"
    )
