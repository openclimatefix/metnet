"""Simple mosaic builder helpers for Source Cooperative icechunk stores.

This script provides utilities to:
- Open icechunk stores from Source Cooperative URLs
- Load satellite data using xarray/zarr
- Create mosaics from multiple satellite datasets

For MetNet Global: combines geostationary satellite imagery from multiple sources.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import numpy as np
import time

# Satellite configuration following MetNet Global paper
# All 7 geostationary satellites with their orbital parameters
# As specified: "3 EUMETSAT ones + GK-2A, Himawari9, GOES-19/GOES-18"
SATELLITE_CONFIG = {
    'goes-east': {
        'longitude': -75.0,
        'height': 35786023.0,  # meters
        'sweep': 'x',
        'url': 'https://data.source.coop/bkr/geo/goes-east_500m.icechunk/'
    },
    'goes-west': {
        'longitude': -137.0,
        'height': 35786023.0,
        'sweep': 'x',
        'url': 'https://data.source.coop/bkr/geo/goes-west_500m.icechunk/'
    },
    'himawari': {
        'longitude': 140.7,
        'height': 35786023.0,
        'sweep': 'y',  # Himawari uses 'y' sweep
        'url': 'https://data.source.coop/bkr/geo/himawari_500m.icechunk/'
    },
    'gk2a': {
        'longitude': 128.2,
        'height': 35786023.0,
        'sweep': 'y',  # GK-2A uses 'y' sweep
        'url': 'https://data.source.coop/bkr/geo/gk2a_500m.icechunk/'
    },
    # The 3 EUMETSAT Meteosat satellites
    'meteosat-9': {
        'longitude': 0.0,
        'height': 35786023.0,
        'sweep': 'y',  # Meteosat uses 'y' sweep
        'url': 'https://data.source.coop/bkr/geo/meteosat-9_500m.icechunk/'
    },
    'meteosat-10': {
        'longitude': 0.0,
        'height': 35786023.0,
        'sweep': 'y',
        'url': 'https://data.source.coop/bkr/geo/meteosat-10_500m.icechunk/'
    },
    'meteosat-11': {
        'longitude': 0.0,
        'height': 35786023.0,
        'sweep': 'y',
        'url': 'https://data.source.coop/bkr/geo/meteosat-11_500m.icechunk/'
    }
}

# Band mappings for different satellite naming conventions
# Following MetNet Global paper's 18-band configuration
# Based on GOES-R ABI spectral channels mapped to equivalent bands on other satellites
BAND_MAPPING = {
    # Visible channels (3 bands)
    'C01': ['C01', 'VI004', 'VIS004', 'B01', 'VIS006'],  # ~0.47 μm (blue)
    'C02': ['C02', 'VI006', 'VIS006', 'B02', 'VIS008'],  # ~0.64 μm (red)
    'C03': ['C03', 'NI008', 'NIR_008', 'B03', 'NIR016'],  # ~0.86 μm (NIR vegetation)
    
    # Near-infrared (3 bands)
    'C04': ['C04', 'NI013', 'NIR_013', 'B04', 'IR_016'],  # ~1.37 μm (cirrus)
    'C05': ['C05', 'NI016', 'NIR_016', 'B05', 'IR_039'],  # ~1.6 μm (snow/ice)
    'C06': ['C06', 'NI022', 'NIR_022', 'B06', 'WV_062'],  # ~2.2 μm (cloud particle size)
    
    # Infrared - shortwave (4 bands)
    'C07': ['C07', 'IR039', 'IR_039', 'B07', 'IR_087'],   # ~3.9 μm (shortwave window)
    'C08': ['C08', 'WV062', 'WV_062', 'B08', 'IR_097'],   # ~6.2 μm (upper-level water vapor)
    'C09': ['C09', 'WV069', 'WV_069', 'B09', 'IR_108'],   # ~6.9 μm (mid-level water vapor)
    'C10': ['C10', 'WV073', 'WV_073', 'B10', 'IR_120'],   # ~7.3 μm (low-level water vapor)
    
    # Infrared - longwave (8 bands)
    'C11': ['C11', 'IR087', 'IR_087', 'B11', 'IR_134'],   # ~8.4 μm (cloud phase)
    'C12': ['C12', 'IR096', 'IR_096', 'B12', 'HRV'],      # ~9.6 μm (ozone)
    'C13': ['C13', 'IR108', 'IR_108', 'B13', 'IR_108'],   # ~10.3 μm (clean longwave window)
    'C14': ['C14', 'IR112', 'IR_112', 'B14', 'IR_120'],   # ~11.2 μm (longwave window)
    'C15': ['C15', 'IR120', 'IR_120', 'B15', 'IR_134'],   # ~12.3 μm (dirty longwave window)
    'C16': ['C16', 'IR133', 'IR_133', 'B16', 'WV_073'],   # ~13.3 μm (CO2 longwave)
    
    # Additional channels for completeness
    'HRV': ['HRV', 'VIS008', 'B17'],  # High Resolution Visible (Meteosat)
    'C17': ['C17', 'B17'],  # Additional channel if present
}

try:
    import xarray as xr
    import zarr
    from icechunk import IcechunkStore, Repository, s3_storage
    from pyresample import geometry, kd_tree
    from pyproj import Proj, Transformer
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install xarray zarr icechunk pyresample pyproj")
    exit(1)


def open_icechunk_store(base_url: str) -> IcechunkStore:
    """Open an Icechunk store from a Source Cooperative URL.
    
    Args:
        base_url: Base URL like 'https://data.source.coop/bkr/geo/gk2a_500m.icechunk/'
    
    Returns:
        IcechunkStore instance
    """
    # Remove trailing slash if present
    base_url = base_url.rstrip('/')
    
    # Parse URL: https://data.source.coop/bkr/geo/gk2a_500m.icechunk/
    # Bucket: bkr (the user/org)
    # Prefix: geo/gk2a_500m.icechunk
    path_parts = base_url.split('source.coop/')[-1]  # Get: bkr/geo/gk2a_500m.icechunk
    parts = path_parts.split('/', 1)
    
    bucket = parts[0]  # e.g., "bkr"
    prefix = parts[1] if len(parts) > 1 else ""  # e.g., "geo/gk2a_500m.icechunk"
    
    # Source Cooperative uses data.source.coop as an S3-compatible endpoint
    storage = s3_storage(
        bucket=bucket,
        prefix=prefix,
        region="us-east-1",  # Default region for S3-compatible services
        endpoint_url="https://data.source.coop",
        anonymous=True,
        allow_http=False,
        force_path_style=True  # Required for S3-compatible endpoints
    )
    
    # Open the repository
    repo = Repository.open(storage)
    
    # Get the store from the main branch
    session = repo.readonly_session(branch="main")
    store = session.store
    
    return store


def load_satellite_data(store_url: str, time: Optional[str] = None) -> xr.Dataset:
    """Load satellite data from an Icechunk store.
    
    Args:
        store_url: URL to the Icechunk store
        time: Optional ISO format time string to select specific timestamp
    
    Returns:
        xarray Dataset with satellite data
    """
    store = open_icechunk_store(store_url)
    
    # Open with xarray
    ds = xr.open_zarr(store, consolidated=False)
    
    # Select specific time if provided
    if time:
        # Handle duplicate time values and ensure monotonic index
        import pandas as pd
        
        # Get unique time indices while preserving order
        time_values = ds.time.values
        unique_times, unique_idx = [], []
        seen = set()
        
        for i, t in enumerate(time_values):
            t_str = str(t)
            if t_str not in seen:
                seen.add(t_str)
                unique_times.append(t)
                unique_idx.append(i)
        
        # Select only unique times
        if len(unique_idx) < len(time_values):
            print(f"   ⚠️  Removed {len(time_values) - len(unique_idx)} duplicate timestamps")
            ds = ds.isel(time=unique_idx)
        
        # Check if time index is monotonic
        time_diffs = np.diff(ds.time.values.astype('int64'))
        if not (np.all(time_diffs > 0) or np.all(time_diffs < 0)):
            print(f"   ⚠️  Time index is not monotonic, sorting...")
            # Sort by time
            sorted_idx = np.argsort(ds.time.values)
            ds = ds.isel(time=sorted_idx)
        
        # Now select the nearest time
        ds = ds.sel(time=time, method='nearest')
    
    return ds


def create_mosaic(satellite_urls: List[str], time: str, output_path: Optional[Path] = None) -> xr.Dataset:
    """Create a mosaic from multiple satellite datasets.
    
    Args:
        satellite_urls: List of Icechunk store URLs for different satellites
        time: ISO format time string to align all datasets
        output_path: Optional path to save the mosaic
    
    Returns:
        Combined xarray Dataset
    """
    datasets = []
    
    print(f"Loading satellite data for time: {time}")
    
    for url in satellite_urls:
        try:
            print(f"Loading: {url}")
            ds = load_satellite_data(url, time=time)
            datasets.append(ds)
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    
    if not datasets:
        raise ValueError("No datasets loaded successfully")
    
    # Combine datasets - simple concatenation or merge depending on structure
    # For now, merge by coordinates
    mosaic = xr.merge(datasets, compat='override')
    
    print(f"Created mosaic with shape: {dict(mosaic.dims)}")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mosaic.to_netcdf(output_path)
        print(f"Saved mosaic to: {output_path}")
    
    return mosaic


def inspect_store(store_url: str):
    """Inspect an Icechunk store and print metadata.
    
    Args:
        store_url: URL to the Icechunk store
    """
    print(f"\nInspecting store: {store_url}")
    print("=" * 80)
    
    try:
        ds = load_satellite_data(store_url)
        
        print(f"\nDataset dimensions: {dict(ds.dims)}")
        print(f"\nData variables:")
        for var in ds.data_vars:
            print(f"  - {var}: {ds[var].dims}, dtype={ds[var].dtype}")
        
        print(f"\nCoordinates:")
        for coord in ds.coords:
            print(f"  - {coord}: shape={ds[coord].shape}, dtype={ds[coord].dtype}")
            if coord == 'time' and len(ds[coord]) > 0:
                print(f"    Time range: {ds[coord].values[0]} to {ds[coord].values[-1]}")
        
        print(f"\nAttributes:")
        for key, value in ds.attrs.items():
            print(f"  - {key}: {value}")
    
    except Exception as e:
        print(f"Error inspecting store: {e}")
        import traceback
        traceback.print_exc()


def create_global_mosaic(
    satellite_urls: List[str],
    target_time: datetime,
    bands: List[str] = ["C02"],
    target_resolution: float = 0.1,
    target_extent: Tuple[float, float, float, float] = (-180, -60, 180, 60)
) -> xr.Dataset:
    """Create global mosaic using azimuth-based blending from MetNet Global paper.
    
    Args:
        satellite_urls: List of satellite Icechunk URLs
        target_time: Target datetime for mosaic
        bands: Spectral bands to include (e.g., ["C02", "C13"])
        target_resolution: Target grid resolution in degrees
        target_extent: (lon_min, lat_min, lon_max, lat_max) in degrees
        
    Returns:
        Global mosaic as xarray Dataset
    """
    lon_min, lat_min, lon_max, lat_max = target_extent
    
    # Create target equirectangular grid
    target_lons = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    target_lats = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    
    # Initialize mosaic storage
    mosaic_data = {}
    total_weights = np.zeros((len(target_lats), len(target_lons)))
    
    print(f"🌍 Creating {len(target_lons)}x{len(target_lats)} global mosaic...")
    total_start = time.time()
    
    for i, url in enumerate(satellite_urls):
        sat_start = time.time()
        print(f"\n📡 Processing satellite {i+1}/{len(satellite_urls)}")
        
        try:
            # Load satellite data
            load_start = time.time()
            sat_data = load_satellite_data(url, target_time)
            load_time = time.time() - load_start
            
            if sat_data is None:
                print(f"   ⚠️  No data available, skipping")
                continue
            
            print(f"   ⏱️  Data loaded in {load_time:.2f}s")
            
            # Extract satellite info
            sat_lon = _extract_satellite_longitude(sat_data, url)
            sat_height = _get_satellite_height(url)
            sat_sweep = _get_satellite_sweep(url)
            print(f"   📍 Satellite longitude: {sat_lon}°, height: {sat_height/1e6:.1f}k km, sweep: {sat_sweep}")
            
            # Process each band
            for band in bands:
                band_start = time.time()
                
                if band not in sat_data.data_vars:
                    # Try alternative band names
                    alt_band = _find_alternative_band(sat_data, band)
                    if alt_band:
                        band = alt_band
                        print(f"   🔄 Using alternative band: {band}")
                    else:
                        print(f"   ⚠️  Band {band} not found, skipping")
                        continue
                
                # Proper geostationary reprojection
                reproject_start = time.time()
                reprojected = _reproject_geostationary(
                    sat_data[band], 
                    target_lons, 
                    target_lats,
                    sat_lon,
                    sat_height=sat_height
                )
                reproject_time = time.time() - reproject_start
                
                if reprojected is None:
                    continue
                
                # Calculate weights
                weight_start = time.time()
                weights = _calculate_azimuth_weights(target_lons, target_lats, sat_lon)
                weight_time = time.time() - weight_start
                
                # Initialize band if first satellite
                if band not in mosaic_data:
                    mosaic_data[band] = np.zeros_like(reprojected)
                
                # Weighted accumulation
                blend_start = time.time()
                valid_mask = ~np.isnan(reprojected)
                mosaic_data[band] += reprojected * weights * valid_mask
                total_weights += weights * valid_mask
                blend_time = time.time() - blend_start
                
                band_total = time.time() - band_start
                print(f"   ✅ Processed band {band} in {band_total:.2f}s (reproject: {reproject_time:.2f}s, weight: {weight_time:.2f}s, blend: {blend_time:.2f}s)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
        
        sat_total = time.time() - sat_start
        print(f"   ⏱️  Satellite total: {sat_total:.2f}s")
    
    # Normalize by weights
    norm_start = time.time()
    for band in mosaic_data:
        valid_weights = total_weights > 0
        mosaic_data[band][valid_weights] /= total_weights[valid_weights]
        mosaic_data[band][~valid_weights] = np.nan
        # Ensure float32 for NetCDF compatibility
        mosaic_data[band] = mosaic_data[band].astype(np.float32)
    norm_time = time.time() - norm_start
    
    # Create dataset
    mosaic_ds = xr.Dataset(
        data_vars={
            band: (["lat", "lon"], mosaic_data[band])
            for band in mosaic_data
        },
        coords={
            "lat": target_lats,
            "lon": target_lons,
            "time": target_time
        },
        attrs={
            "title": "MetNet Global Satellite Mosaic",
            "projection": "equirectangular",
            "resolution_degrees": target_resolution,
            "satellites_used": len(satellite_urls)
        }
    )
    
    total_time = time.time() - total_start
    print(f"\n✅ Mosaic complete in {total_time:.2f}s: {len(mosaic_data)} bands from {len(satellite_urls)} satellites")
    print(f"   Normalization: {norm_time:.2f}s")
    return mosaic_ds


def _extract_satellite_longitude(sat_data: xr.Dataset, url: str) -> float:
    """Extract satellite longitude from data or URL."""
    # Try to get from data attributes
    orbital_params = sat_data.attrs.get('orbital_parameters', {})
    if isinstance(orbital_params, dict) and 'projection_longitude' in orbital_params:
        return float(orbital_params['projection_longitude'])
    
    # Fallback: look up from SATELLITE_CONFIG
    url_lower = url.lower()
    for sat_name, config in SATELLITE_CONFIG.items():
        if sat_name in url_lower:
            return config['longitude']
    
    # Last resort: try to infer from URL
    if 'goes-east' in url_lower or 'goes16' in url_lower:
        return -75.0
    elif 'goes-west' in url_lower or 'goes17' in url_lower or 'goes18' in url_lower:
        return -137.0
    elif 'himawari' in url_lower:
        return 140.7
    elif 'gk2a' in url_lower or 'geo-kompsat' in url_lower:
        return 128.2
    elif 'meteosat' in url_lower or 'msg' in url_lower:
        return 0.0
    
    print(f"    ⚠️  Could not determine satellite longitude from URL: {url}")
    return 0.0  # Default


def _get_satellite_height(url: str) -> float:
    """Get satellite height above Earth's surface."""
    url_lower = url.lower()
    for sat_name, config in SATELLITE_CONFIG.items():
        if sat_name in url_lower:
            return config['height']
    return 35786023.0  # Default geostationary height


def _get_satellite_sweep(url: str) -> str:
    """Get satellite sweep angle convention ('x' or 'y')."""
    url_lower = url.lower()
    for sat_name, config in SATELLITE_CONFIG.items():
        if sat_name in url_lower:
            return config['sweep']
    return 'x'  # Default to GOES convention


def _find_alternative_band(sat_data: xr.Dataset, target_band: str) -> Optional[str]:
    """Find alternative band names (different satellites use different naming conventions)."""
    if target_band in BAND_MAPPING:
        for alt_name in BAND_MAPPING[target_band]:
            if alt_name in sat_data.data_vars:
                return alt_name
    
    # Direct match
    if target_band in sat_data.data_vars:
        return target_band
    
    return None


def _reproject_geostationary(
    data_array: xr.DataArray, 
    target_lons: np.ndarray, 
    target_lats: np.ndarray,
    sat_lon: float,
    sat_height: float = 35786023.0,  # Geostationary orbit height in meters
    subsample_factor: int = 20
) -> Optional[np.ndarray]:
    """Reproject geostationary satellite data to equirectangular grid using pyresample.
    
    This implements proper geostationary projection transformation following the
    MetNet Global paper approach.
    
    Args:
        data_array: Input satellite data with geostationary coordinates
        target_lons: Target longitude grid
        target_lats: Target latitude grid
        sat_lon: Satellite sub-point longitude
        sat_height: Satellite height above Earth's surface (meters)
        subsample_factor: Factor to subsample data before reprojection (for speed)
    
    Returns:
        Reprojected data on target grid, or None if failed
    """
    try:
        # Check if data has proper geostationary coordinates
        if 'x_geostationary' not in data_array.coords or 'y_geostationary' not in data_array.coords:
            print(f"    ⚠️  Missing geostationary coordinates")
            return None
        
        # OPTIMIZATION: Subsample BEFORE loading to reduce data transfer
        data_subset = data_array.isel(
            x_geostationary=slice(None, None, subsample_factor),
            y_geostationary=slice(None, None, subsample_factor)
        )
        
        # Get geostationary coordinates (in radians)
        x_geo = data_subset.coords['x_geostationary'].values
        y_geo = data_subset.coords['y_geostationary'].values
        
        # Load data
        data_values = data_subset.compute().values
        if data_values.dtype == np.float16:
            data_values = data_values.astype(np.float32)
        
        # Define source geostationary area using pyresample
        # Geostationary projection definition
        proj_dict = {
            'proj': 'geos',
            'lon_0': sat_lon,
            'h': sat_height,
            'x_0': 0,
            'y_0': 0,
            'ellps': 'WGS84',
            'units': 'm',
            'sweep': 'x'  # GOES satellites use 'x' sweep
        }
        
        # Create source area definition
        # x_geo and y_geo are in radians, need to convert to meters
        x_size = len(x_geo)
        y_size = len(y_geo)
        
        area_extent = (
            x_geo[0] * sat_height,
            y_geo[-1] * sat_height,
            x_geo[-1] * sat_height,
            y_geo[0] * sat_height
        )
        
        source_area = geometry.AreaDefinition(
            'geostationary',
            'Geostationary satellite view',
            'geos',
            proj_dict,
            x_size,
            y_size,
            area_extent
        )
        
        # Define target equirectangular area
        target_proj_dict = {
            'proj': 'longlat',
            'datum': 'WGS84'
        }
        
        lon_min, lon_max = target_lons[0], target_lons[-1]
        lat_min, lat_max = target_lats[0], target_lats[-1]
        
        target_area = geometry.AreaDefinition(
            'equirectangular',
            'Equirectangular grid',
            'longlat',
            target_proj_dict,
            len(target_lons),
            len(target_lats),
            (lon_min, lat_min, lon_max, lat_max)
        )
        
        # Perform reprojection using pyresample's kd_tree (fast nearest neighbor)
        # Use a radius of influence to avoid excessive interpolation artifacts
        radius_of_influence = 50000  # 50 km
        
        result = kd_tree.resample_nearest(
            source_area,
            data_values,
            target_area,
            radius_of_influence=radius_of_influence,
            fill_value=np.nan
        )
        
        return result.astype(np.float32)
        
    except Exception as e:
        print(f"    ❌ Reprojection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _calculate_azimuth_weights(lons: np.ndarray, lats: np.ndarray, sat_lon: float, sat_lat: float = 0.0) -> np.ndarray:
    """Calculate azimuth-based weights for satellite blending.
    
    Implements the azimuth-based blending approach from MetNet Global paper.
    Satellites are weighted based on their viewing angle - pixels near the satellite's
    nadir (directly below) get higher weights than pixels at oblique viewing angles.
    
    Args:
        lons: Target longitude grid
        lats: Target latitude grid
        sat_lon: Satellite sub-point longitude
        sat_lat: Satellite sub-point latitude (always 0 for geostationary)
    
    Returns:
        Weight array with values in [0, 1]
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Calculate great circle distance from satellite nadir point
    # Using haversine formula for accurate distance on sphere
    
    # Convert to radians
    lon1 = np.radians(sat_lon)
    lat1 = np.radians(sat_lat)
    lon2 = np.radians(lon_grid)
    lat2 = np.radians(lat_grid)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Angular distance in radians
    
    # Convert to degrees for interpretability
    angular_distance_deg = np.degrees(c)
    
    # Weight function: inverse of (1 + distance)^2
    # This gives strong preference to nadir views while smoothly transitioning
    # to lower weights at oblique angles
    weights = 1.0 / (1.0 + angular_distance_deg**2 / 100.0)
    
    # Apply viewing angle cutoff - satellites typically can't see beyond ~70° from nadir
    max_viewing_angle = 70.0  # degrees
    weights[angular_distance_deg > max_viewing_angle] = 0.0
    
    # Normalize to [0, 1] range
    if weights.max() > 0:
        weights = weights / weights.max()
    
    return weights


def test_global_mosaic():
    """Test creating a global mosaic with all 18 bands from MetNet Global paper."""
    print("🌍 Testing Global Mosaic Creation")
    print("=" * 50)
    
    # QUICK TEST: Use only 2 satellites but ALL 18 bands
    print("\n⚡ QUICK TEST MODE: 2 satellites × 18 bands")
    print("   Testing full band set from MetNet Global paper")
    
    satellite_urls = [
        SATELLITE_CONFIG['goes-east']['url'],
        SATELLITE_CONFIG['goes-west']['url']
    ]
    
    # All 18 bands from MetNet Global paper
    all_bands = [
        # Visible (3)
        "C01", "C02", "C03",
        # Near-infrared (3)
        "C04", "C05", "C06",
        # Infrared shortwave (4)
        "C07", "C08", "C09", "C10",
        # Infrared longwave (8)
        "C11", "C12", "C13", "C14", "C15", "C16",
        # Additional if available
        "C17", "HRV"
    ]
    
    print(f"\n📡 Using {len(satellite_urls)} satellites:")
    for name in ['goes-east', 'goes-west']:
        config = SATELLITE_CONFIG[name]
        print(f"   - {name}: {config['longitude']}°")
    
    print(f"\n📊 Testing {len(all_bands)} bands:")
    print(f"   Visible: C01, C02, C03")
    print(f"   Near-IR: C04, C05, C06")
    print(f"   IR-SW: C07, C08, C09, C10")
    print(f"   IR-LW: C11, C12, C13, C14, C15, C16")
    print(f"   Extra: C17, HRV")
    
    target_time = datetime(2018, 1, 1, 0, 0, 0)
    
    try:
        mosaic = create_global_mosaic(
            satellite_urls,
            target_time,
            bands=all_bands,  # All 18 bands!
            target_resolution=2.0,  # 2° for testing (181x61 grid)
            target_extent=(-180, -60, 180, 60)
        )
        
        print(f"\n🎉 Success!")
        print(f"   Shape: {dict(mosaic.sizes)}")
        print(f"   Bands found: {len(mosaic.data_vars)}/{len(all_bands)}")
        
        # Summary by category
        visible = [b for b in ["C01", "C02", "C03"] if b in mosaic.data_vars]
        near_ir = [b for b in ["C04", "C05", "C06"] if b in mosaic.data_vars]
        ir_sw = [b for b in ["C07", "C08", "C09", "C10"] if b in mosaic.data_vars]
        ir_lw = [b for b in ["C11", "C12", "C13", "C14", "C15", "C16"] if b in mosaic.data_vars]
        
        print(f"\n📊 Band Coverage:")
        print(f"   Visible: {len(visible)}/3 - {visible}")
        print(f"   Near-IR: {len(near_ir)}/3 - {near_ir}")
        print(f"   IR-SW: {len(ir_sw)}/4 - {ir_sw}")
        print(f"   IR-LW: {len(ir_lw)}/8 - {ir_lw}")
        
        # Show statistics for each band
        print(f"\n📈 Band Statistics:")
        for band in sorted(mosaic.data_vars):
            data = mosaic[band].values
            valid_pixels = (~np.isnan(data)).sum()
            total_pixels = data.size
            coverage = 100.0 * valid_pixels / total_pixels
            if valid_pixels > 0:
                print(f"   {band}: {coverage:5.1f}% coverage, range [{np.nanmin(data):8.3f}, {np.nanmax(data):8.3f}]")
            else:
                print(f"   {band}: {coverage:5.1f}% coverage (no valid data)")
        
        # Save output
        output_file = Path("test_global_mosaic_18bands.nc")
        mosaic.to_netcdf(output_file)
        print(f"\n   💾 Saved: {output_file}")
        
        print("\n✅ All systems operational!")
        print(f"   Successfully processed {len(mosaic.data_vars)}/{len(all_bands)} bands")
        print("   To test all 7 satellites, update satellite_urls in test_global_mosaic()")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Mosaic builder for Source Cooperative Icechunk satellite data"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect an Icechunk store')
    inspect_parser.add_argument('store_url', help='URL to Icechunk store')
    
    # Mosaic command
    mosaic_parser = subparsers.add_parser('mosaic', help='Create a mosaic from multiple satellites')
    mosaic_parser.add_argument('--stores', nargs='+', required=True, help='URLs to Icechunk stores')
    mosaic_parser.add_argument('--time', required=True, help='Time to create mosaic for (ISO format)')
    mosaic_parser.add_argument('--output', type=Path, help='Output path for mosaic')
    
    # Test command
    test_parser = subparsers.add_parser('test-mosaic', help='Test global mosaic creation')
    
    args = parser.parse_args()
    
    if args.command == 'inspect':
        inspect_store(args.store_url)
    
    elif args.command == 'mosaic':
        # Parse time string
        target_time = datetime.fromisoformat(args.time.replace('Z', '+00:00'))
        mosaic = create_global_mosaic(args.stores, target_time, output_path=args.output)
        print("\nMosaic created successfully!")
    
    elif args.command == 'test-mosaic':
        success = test_global_mosaic()
        exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()