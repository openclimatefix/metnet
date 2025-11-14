"""
Gaussian center-weighted blending utilities for satellite mosaicing.

This file provides:
  • haversine-based angular distance (pure math)
  • Gaussian weight computation per satellite (lat0, lon0 provided by caller)
  • Gaussian merging of N satellite arrays (N ≥ 2)

This module contains no I/O and is fully testable with synthetic grids.
"""

from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------


def haversine_rad(lat1, lon1, lat2, lon2):
    """
    Great-circle angular distance (RADIANS) between points in degrees.

    Supports scalars or numpy arrays.
    """
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.deg2rad(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(a))


def gaussian_weight_from_subpoint(
    lat_grid,
    lon_grid,
    lat0: float,
    lon0: float,
    sigma_deg: float = 20.0,
) -> np.ndarray:
    """
    Compute Gaussian weights for one satellite across a lat/lon grid.

    Args:
      lat_grid: ndarray or DataArray (degrees)
      lon_grid: ndarray or DataArray (degrees)
      lat0: sub-satellite point latitude (degrees)
      lon0: sub-satellite point longitude (degrees)
      sigma_deg: Gaussian sigma (degrees)

    Returns:
      ndarray of weights, same shape as lat_grid.
    """
    d = haversine_rad(lat_grid, lon_grid, lat0, lon0)
    sigma_rad = np.deg2rad(sigma_deg)
    return np.exp(-(d**2) / (2.0 * sigma_rad**2))


# ---------------------------------------------------------------------
# MULTI-SATELLITE GAUSSIAN MERGE (general case, N ≥ 2)
# ---------------------------------------------------------------------


def gaussian_merge_multi(
    arrays: List[xr.DataArray],
    lat: xr.DataArray,
    lon: xr.DataArray,
    subpoints: Dict[str, Tuple[float, float]],
    sat_names: List[str],
    sigma_deg: float = 20.0,
) -> xr.DataArray:
    """
    Gaussian center-weighted merge of N satellite DataArrays.

    Args:
      arrays: list[xr.DataArray], all with spatial dims ("y", "x")
      lat: xr.DataArray (H, W) latitude grid
      lon: xr.DataArray (H, W) longitude grid
      subpoints: dict[sat_name -> (lat0, lon0)]
      sat_names: list[str], must match arrays in order
      sigma_deg: Gaussian sigma in degrees

    Returns:
      xr.DataArray containing the merged reflectance field.
    """
    if len(arrays) != len(sat_names):
        raise ValueError("arrays and sat_names must have same length")

    # Align arrays + grids
    aligned = xr.align(*arrays, lat, lon, join="outer")

    aligned_arrays = aligned[:-2]
    lat_aligned = aligned[-2].values.astype(float)
    lon_aligned = aligned[-1].values.astype(float)

    arr_vals = []
    weight_maps = []

    # Compute per-satellite weights and values
    for arr, sat_name in zip(aligned_arrays, sat_names):
        vals = arr.values.astype(float)
        arr_vals.append(vals)

        if sat_name not in subpoints:
            raise KeyError(f"Missing subpoint for satellite '{sat_name}'")

        lat0, lon0 = subpoints[sat_name]

        weights = gaussian_weight_from_subpoint(lat_aligned, lon_aligned, lat0, lon0, sigma_deg)
        weights *= ~np.isnan(vals)

        weight_maps.append(weights)

    arr_stack = np.stack(arr_vals, axis=0)
    weight_stack = np.stack(weight_maps, axis=0)

    numerator = np.nansum(weight_stack * arr_stack, axis=0)
    denominator = np.sum(weight_stack, axis=0)

    merged = np.full_like(numerator, np.nan)
    valid = denominator > 0.0
    merged[valid] = numerator[valid] / denominator[valid]

    zero_mask = ~valid
    if np.any(zero_mask):
        fallback = np.full_like(merged, np.nan)
        for vals in arr_vals:
            fallback = np.where(~np.isnan(vals) & np.isnan(fallback), vals, fallback)
        merged[zero_mask] = fallback[zero_mask]

    out = aligned_arrays[0].copy(deep=False)
    out.values = merged
    out.name = "merged"
    return out
