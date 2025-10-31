"""
Merge matching bands from multiple geostationary satellites.

Implements wavelength-based band matching across satellites as described in
the MetNet Global paper. This is a minimal proof-of-concept focusing on the
0.86 µm band from GK2A and GOES-East.

TODO: Add proper reprojection and azimuth-based blending.
"""

import xarray as xr
import numpy as np
import icechunk


def open_icechunk_store(prefix: str):
    """Open an icechunk store from Source Cooperative."""
    storage = icechunk.s3_storage(
        bucket="bkr",
        prefix=prefix,
        region="us-east-1",
        endpoint_url="https://data.source.coop",
        anonymous=True,
        force_path_style=True
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch="main")
    return xr.open_zarr(session.store, consolidated=False)


def get_band_data(ds, band_name: str, time_idx: int = 0):
    """Extract a single timestep of a band."""
    if band_name not in ds.data_vars:
        return None
    
    # Get single time slice
    band = ds[band_name].isel(time=time_idx)
    
    return band


def merge_086um_band(time_idx: int = -1):
    """
    Merge 0.86 µm band from GK2A and GOES-East satellites.
    
    Band matching:
    - GK2A VI008: 0.863 µm
    - GOES-East C03: 0.865 µm
    
    Args:
        time_idx: Time index to extract (-1 for most recent)
        
    Returns:
        Tuple of (merged_data, metadata)
    """
    
    # Load GK2A data
    ds_gk2a = open_icechunk_store("geo/gk2a_1000m.icechunk")
    band_gk2a = ds_gk2a["VI008"].isel(time=time_idx)
    
    # Load GOES-East data
    ds_goes = open_icechunk_store("geo/goes-east_1000m.icechunk")
    band_goes = ds_goes["C03"].isel(time=time_idx)
    
    # Simple merge: average where both have data, otherwise use available
    if band_gk2a.shape == band_goes.shape:
        merged = xr.where(
            np.isnan(band_gk2a) & np.isnan(band_goes),
            np.nan,
            xr.where(
                np.isnan(band_gk2a),
                band_goes,
                xr.where(
                    np.isnan(band_goes),
                    band_gk2a,
                    (band_gk2a + band_goes) / 2
                )
            )
        )
    else:
        # Shapes don't match, use first satellite's grid
        merged = band_gk2a
    
    metadata = {
        "wavelength": "0.86 µm",
        "satellites": ["GK2A", "GOES-East"],
        "bands": {"GK2A": "VI008", "GOES-East": "C03"},
        "method": "mean",
    }
    
    return merged, metadata


if __name__ == "__main__":
    merged_data, metadata = merge_086um_band(time_idx=-1)
    print(f"Merged {metadata['wavelength']} from {metadata['satellites']}")
    print(f"Shape: {merged_data.shape}, Valid pixels: {np.sum(~np.isnan(merged_data.values))}")


