#!/usr/bin/env python3
"""
Example: Create a global satellite mosaic using MetNet Global methodology.

This example demonstrates creating a global mosaic from multiple geostationary
satellites with azimuth-based blending, following the MetNet Global paper approach.

Current Status:
- ✅ Supports all 7 satellites (GOES-East, GOES-West, Himawari, GK-2A, 3x Meteosat)
- ✅ Equirectangular projection
- ✅ Azimuth-based blending with 70° viewing angle cutoff
- ✅ pyresample for proper geostationary reprojection
- ⚠️  Currently only C02 band available in Source Cooperative dataset
- 🔜 Will support all 18 bands when data becomes available

Usage:
    python example_mosaic.py
"""

import sys
sys.path.insert(0, '/Users/sagarpillai/metnet/.venv_py312/lib/python3.12/site-packages')

from datetime import datetime
from mosaic_builder import create_global_mosaic, SATELLITE_CONFIG

def main():
    """Create a global mosaic from available geostationary satellites."""
    
    print("="*80)
    print("MetNet Global - Satellite Mosaicing Example")
    print("="*80)
    
    # Configure satellites (all 7 from the paper)
    satellite_urls = [
        SATELLITE_CONFIG["goes-east"]["url"],
        SATELLITE_CONFIG["goes-west"]["url"],
        SATELLITE_CONFIG["himawari"]["url"],
        SATELLITE_CONFIG["gk2a"]["url"],
        SATELLITE_CONFIG["meteosat-9"]["url"],
        SATELLITE_CONFIG["meteosat-10"]["url"],
        SATELLITE_CONFIG["meteosat-11"]["url"],
    ]
    
    # Currently only C02 (visible red, ~0.64 μm) is available
    # The pipeline is ready for all 18 bands once data becomes available
    bands = ["C02"]
    
    # Select timestamp (2018-01-01 has good coverage)
    target_time = datetime(2018, 1, 1, 0, 0)
    
    # Target grid: 0.1° resolution, -180 to 180 lon, -60 to 60 lat
    # (Matches MetNet Global paper configuration)
    target_resolution = 0.1  # degrees
    target_extent = (-180, -60, 180, 60)  # (lon_min, lat_min, lon_max, lat_max)
    
    print(f"\n📋 Configuration:")
    print(f"   Satellites: {len(satellite_urls)} (GOES-E, GOES-W, Himawari, GK-2A, 3x Meteosat)")
    print(f"   Bands: {bands}")
    print(f"   Time: {target_time}")
    print(f"   Resolution: {target_resolution}° ({target_resolution*111:.1f} km at equator)")
    print(f"   Extent: {target_extent}")
    print(f"   Output grid: {int((target_extent[2]-target_extent[0])/target_resolution)+1} × "
          f"{int((target_extent[3]-target_extent[1])/target_resolution)+1} pixels")
    
    print(f"\n🚀 Creating mosaic...")
    print(f"   Note: First run may take ~30 minutes due to S3 data transfer")
    
    # Create the mosaic
    mosaic = create_global_mosaic(
        satellite_urls=satellite_urls,
        target_time=target_time,
        bands=bands,
        target_resolution=target_resolution,
        target_extent=target_extent
    )
    
    print(f"\n✅ Mosaic created successfully!")
    print(f"\n📊 Output Summary:")
    print(f"   Dimensions: {dict(mosaic.dims)}")
    print(f"   Variables: {list(mosaic.data_vars)}")
    print(f"   Coordinates: {list(mosaic.coords)}")
    
    # Display data statistics
    import numpy as np
    print(f"\n📈 Data Statistics:")
    for var in mosaic.data_vars:
        data = mosaic[var].values
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(f"   {var}:")
            print(f"      Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]")
            print(f"      Mean: {valid_data.mean():.3f}")
            print(f"      Std: {valid_data.std():.3f}")
            print(f"      Coverage: {len(valid_data)/data.size*100:.1f}%")
        else:
            print(f"   {var}: No valid data")
    
    print(f"\n💾 Output saved to: test_global_mosaic.nc")
    print(f"\n🎯 Next Steps:")
    print(f"   - Visualize the mosaic (e.g., using xarray.plot or matplotlib)")
    print(f"   - Verify spatial coverage and blending quality")
    print(f"   - Test with different timestamps")
    print(f"   - Add more bands as they become available")


if __name__ == "__main__":
    main()
