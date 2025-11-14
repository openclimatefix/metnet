"""
Gaussian center-weighted blending utilities for satellite mosaicing.

This file provides:
  • haversine-based angular distance (pure math)
  • Gaussian weight computation per satellite
  • Gaussian merging of N satellite arrays (N ≥ 2)

This is pure math and can be unit-tested with synthetic lat/lon grids.
"""

from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------
# Satellite sub-satellite points (lat, lon in degrees)
# Extend this dict as more satellites are added.
# ---------------------------------------------------------------------
SATELLITE_SUBPOINTS: Dict[str, Tuple[float, float]] = {
    "gk2a": (0.0, 128.2),  # GK2A sub-satellite point
    "goes_east": (0.0, -75.2),  # GOES-East
    "goes_west": (0.0, -137.2),
    "himawari": (0.0, 140.7),
    "meteosat": (0.0, 0.0),
}

# Geometry utilities


def haversine_rad(lat1, lon1, lat2, lon2):
    """
    Great-circle angular distance (RADIANS) between points in degrees.

    Supports scalars or numpy arrays.
    """
    φ1 = np.deg2rad(lat1)
    φ2 = np.deg2rad(lat2)
    Δφ = φ2 - φ1
    Δλ = np.deg2rad(lon2 - lon1)

    a = np.sin(Δφ / 2.0) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(a))


def gaussian_weight_from_subpoint(
    lat_grid,
    lon_grid,
    sat_name: str,
    sigma_deg: float = 20.0,
):
    """
    Compute Gaussian weights for one satellite across a lat/lon grid.

    Args:
      lat_grid: ndarray or DataArray (degrees)
      lon_grid: ndarray or DataArray (degrees)
      sat_name: key in SATELLITE_SUBPOINTS dict
      sigma_deg: Gaussian sigma (degrees)

    Returns:
      ndarray of weights, same shape as lat_grid.
    """

    if sat_name not in SATELLITE_SUBPOINTS:
        raise ValueError(f"Unknown satellite '{sat_name}'")

    lat0, lon0 = SATELLITE_SUBPOINTS[sat_name]

    d = haversine_rad(lat_grid, lon_grid, lat0, lon0)
    sigma_rad = np.deg2rad(sigma_deg)

    return np.exp(-(d**2) / (2.0 * sigma_rad**2))


# MULTI-SATELLITE GAUSSIAN MERGE (general case, N ≥ 2)


def gaussian_merge_multi(
    arrays: List[xr.DataArray],
    lat: xr.DataArray,
    lon: xr.DataArray,
    sat_names: List[str],
    sigma_deg: float = 20.0,
) -> xr.DataArray:
    """
    Gaussian center-weighted merge of N satellite DataArrays.

    Args:
      arrays: list[xr.DataArray], must all have spatial dims ("y", "x")
      lat: xr.DataArray (H, W) with latitude values for each pixel
      lon: xr.DataArray (H, W) with longitude values for each pixel
      sat_names: list[str] — same length as arrays
      sigma_deg: Gaussian sigma (degrees)

    Returns:
      xr.DataArray with merged values across all satellites.
    """
    if len(arrays) != len(sat_names):
        raise ValueError("arrays and sat_names must have same length")

    # Align all inputs on a shared union of coordinates
    aligned = xr.align(*arrays, lat, lon, join="outer")

    aligned_arrays = aligned[:-2]  # last two are lat/lon
    lat_aligned = aligned[-2]
    lon_aligned = aligned[-1]

    lat_vals = lat_aligned.values.astype(float)
    lon_vals = lon_aligned.values.astype(float)

    # Prepare containers
    arr_vals = []
    weight_maps = []

    # Compute weights + mask NaNs
    for arr, sat_name in zip(aligned_arrays, sat_names):
        vals = arr.values.astype(float)
        arr_vals.append(vals)

        weights = gaussian_weight_from_subpoint(lat_vals, lon_vals, sat_name, sigma_deg)
        weights *= ~np.isnan(vals)  # zero-out weights where data missing

        weight_maps.append(weights)

    arr_stack = np.stack(arr_vals, axis=0)  # (N, H, W)
    w_stack = np.stack(weight_maps, axis=0)  # (N, H, W)

    numerator = np.nansum(w_stack * arr_stack, axis=0)
    denominator = np.sum(w_stack, axis=0)

    merged = np.full_like(numerator, np.nan)

    # Weighted average where denom > 0
    valid = denominator > 0.0
    merged[valid] = numerator[valid] / denominator[valid]

    # Fallback where all weights = 0 (but some arrays may have data)
    zero_mask = ~valid
    if np.any(zero_mask):
        fallback = np.full_like(merged, np.nan)
        for vals in arr_vals:
            fallback = np.where(~np.isnan(vals) & np.isnan(fallback), vals, fallback)
        merged[zero_mask] = fallback[zero_mask]

    # Build output DataArray
    out = aligned_arrays[0].copy(deep=False)
    out.values = merged
    out.name = "merged"
    return out
