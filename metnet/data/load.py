"""Loaders for the real-world data sources used by MetNet-3."""

import gzip
from datetime import datetime

import eccodes
import fsspec
import numpy as np

MRMS_BUCKET = "s3://noaa-mrms-pds"
MRMS_SHAPE = (3500, 7000)  # fixed ~1 km CONUS grid


def load_mrms(
    timestamp: datetime,
    product: str = "PrecipRate",
    lat_min: float = 20.01,
    lat_max: float = 54.99,
    lon_min: float = -130.00,
    lon_max: float = -60.01,
    target_resolution_km: float = 4.0,
) -> np.ndarray:
    """
    Load a single MRMS product at a given timestamp from AWS S3.

    MRMS data is available at ~1km resolution on a fixed CONUS grid
    (3500 x 7000 pixels), published every 2 minutes. This function fetches the
    file, crops it to the requested domain, replaces fill values with NaN, and
    average-pools to target_resolution_km.

    Args:
        timestamp: UTC datetime to load, floored to the previous 2 minute step
        product: MRMS product name, e.g. 'PrecipRate'
        lat_min: Minimum latitude of domain
        lat_max: Maximum latitude of domain
        lon_min: Minimum longitude of domain (-180 to 180)
        lon_max: Maximum longitude of domain (-180 to 180)
        target_resolution_km: Target resolution in km. 1.0 keeps the native
            grid, 4.0 gives the MetNet-3 high-res path, 8.0 the low-res path.

    Returns:
        2D numpy array [height, width] cropped and resampled to the target
        domain, with MRMS fill values (negative, e.g. -3.0) replaced by NaN
    """
    # Floor timestamp to the 2 minute grid
    minute = (timestamp.minute // 2) * 2
    timestamp = timestamp.replace(minute=minute, second=0, microsecond=0)

    # Build S3 path
    date_str = timestamp.strftime("%Y%m%d")
    time_str = timestamp.strftime("%Y%m%d-%H%M%S")
    key = f"CONUS/{product}_00.00/{date_str}/MRMS_{product}_00.00_{time_str}.grib2.gz"

    # Download from S3 (anonymous access) and decompress in memory
    fs = fsspec.filesystem("s3", anon=True)
    with fs.open(f"{MRMS_BUCKET}/{key}", "rb") as f:
        grib_bytes = gzip.decompress(f.read())

    # Decode the GRIB2 message directly from bytes
    msg_id = eccodes.codes_new_from_message(grib_bytes)
    try:
        data = eccodes.codes_get_values(msg_id).astype(np.float32)
        lats = eccodes.codes_get_array(msg_id, "latitudes")
        lons = eccodes.codes_get_array(msg_id, "longitudes")
    finally:
        eccodes.codes_release(msg_id)

    data = data.reshape(MRMS_SHAPE)
    lats = lats.reshape(MRMS_SHAPE)
    lons = lons.reshape(MRMS_SHAPE)

    # Convert longitudes from 0-360 to -180 to 180
    lons = np.where(lons > 180, lons - 360, lons)

    # Replace MRMS fill values with NaN
    data = np.where(data < 0, np.nan, data)

    # Crop to domain. The grid is regular, so rows depend only on latitude
    # and columns only on longitude. eps keeps points exactly on a bound.
    eps = 1e-4
    lat_mask = (lats[:, 0] >= lat_min - eps) & (lats[:, 0] <= lat_max + eps)
    lon_mask = (lons[0, :] >= lon_min - eps) & (lons[0, :] <= lon_max + eps)
    data = data[np.ix_(lat_mask, lon_mask)]

    # Resample to target resolution via average pooling (native MRMS is ~1 km)
    factor = int(target_resolution_km)
    if factor > 1:
        h, w = data.shape
        h_new, w_new = h // factor, w // factor
        data = data[: h_new * factor, : w_new * factor]
        data = np.nanmean(data.reshape(h_new, factor, w_new, factor), axis=(1, 3))

    return data
