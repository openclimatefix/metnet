import fsspec
import gzip
import io
import os
import tempfile
import eccodes

import numpy as np
from datetime import datetime

def load_mrms(
    timestamp: datetime,
    product: str = "PrecipRate",
    lat_min: float = 25.0,
    lat_max: float = 50.0,
    lon_min: float = -125.0,
    lon_max: float = -65.0,
    target_resolution_km: float = 4.0,
) -> np.ndarray:
    """
    Load a single MRMS product at a given timestamp from AWS S3.

    MRMS data is available at ~1km resolution on a fixed CONUS grid
    (3500 x 7000 pixels). This function fetches, crops to the domain,
    replaces fill values with NaN, and resamples to target_resolution_km.

    Args:
        timestamp: UTC datetime to load — rounded to nearest 2 minutes
        product: MRMS product name, 'PrecipRate' or 'PrecipFlag'
        lat_min: Minimum latitude of domain
        lat_max: Maximum latitude of domain
        lon_min: Minimum longitude of domain
        lon_max: Maximum longitude of domain
        target_resolution_km: Target resolution in km, 4.0 for high-res
            path, 8.0 for low-res path per MetNet-3 paper

    Returns:
        2D numpy array [height, width] cropped and resampled to target domain,
        with MRMS fill values (-3.0) replaced by NaN
    """

    # Round timestamp to nearest 2 minutes
    minute = (timestamp.minute // 2) * 2
    timestamp = timestamp.replace(minute=minute, second=0, microsecond=0)

    # Build S3 path
    date_str = timestamp.strftime("%Y%m%d")
    time_str = timestamp.strftime("%Y%m%d-%H%M%S")
    key = f"CONUS/{product}_00.00/{date_str}/MRMS_{product}_00.00_{time_str}.grib2.gz"

    # Download from S3 — no auth needed
    fs = fsspec.filesystem("s3", anon=True)
    with fs.open(f"s3://noaa-mrms-pds/{key}", "rb") as f:
        compressed = f.read()

    # Decompress
    grib_bytes = gzip.decompress(compressed)

    # Read directly from in-memory bytes using eccodes
    data, lats, lons = None, None, None
    with eccodes.GribFile(io.BytesIO(grib_bytes)) as grib:
        msg = next(grib)  # gets the first GRIB message
        data = msg["values"].astype(np.float32)
        lats = msg["latitudes"]
        lons = msg["longitudes"]

    # Convert longitudes from 0-360 to -180 to 180
    lons = np.where(lons > 180, lons - 360, lons)

    # Replace MRMS fill value with NaN
    data = np.where(data < 0, np.nan, data)

    # Crop to domain
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    data = data[np.ix_(lat_mask, lon_mask)]

    # Resample to target resolution via average pooling
    # Native MRMS is ~1km, factor = target_resolution_km
    factor = int(target_resolution_km)
    if factor > 1:
        h, w = data.shape
        h_new = h // factor
        w_new = w // factor
        data = data[:h_new * factor, :w_new * factor]
        data = np.nanmean(
            data.reshape(h_new, factor, w_new, factor),
            axis=(1, 3)
        )

    return data

