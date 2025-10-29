#!/usr/bin/env python3
"""Quick test: 2 satellites × 2 bands to validate production code."""

import sys

sys.path.insert(0, "/Users/sagarpillai/metnet/.venv_py312/lib/python3.12/site-packages")

from datetime import datetime

from mosaic_builder import SATELLITE_CONFIG, create_global_mosaic


def main():
    """Run quick validation test."""
    print("\n" + "=" * 80)
    print("QUICK VALIDATION TEST: 2 Satellites × 2 Bands")
    print("=" * 80)

    # Select 2 satellites that have data in 2018
    test_satellites = [
        SATELLITE_CONFIG["goes-east"]["url"],
        SATELLITE_CONFIG["goes-west"]["url"],
    ]

    # Select 2 common bands
    test_bands = ["C02", "C13"]  # Visible red and IR longwave

    target_time = datetime(2018, 1, 1, 0, 0)

    print("\n📋 Configuration:")
    print(f"  Satellites: {len(test_satellites)}")
    for url in test_satellites:
        sat_name = url.split("/")[-2].replace("_500m.icechunk", "")
        print(f"    - {sat_name}")
    print(f"  Bands: {test_bands}")
    print(f"  Time: {target_time}")
    print("  Resolution: 0.1°")
    print("  Extent: [-180, -60, 180, 60]")

    print("\n🚀 Starting mosaic creation...")

    try:
        mosaic = create_global_mosaic(
            satellite_urls=test_satellites,
            target_time=target_time,
            bands=test_bands,
            target_resolution=0.1,
            target_extent=(-180, -60, 180, 60),
        )

        print("\n" + "=" * 80)
        print("✅ TEST PASSED!")
        print("=" * 80)
        print("\n📊 Mosaic Summary:")
        print(f"  Dimensions: {dict(mosaic.dims)}")
        print(f"  Variables: {list(mosaic.data_vars)}")
        print(f"  Coordinates: {list(mosaic.coords)}")

        # Check data ranges
        print("\n📈 Data Ranges:")
        for var in mosaic.data_vars:
            data = mosaic[var].values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                print(
                    f"  {var}: [{valid_data.min():.3f}, {valid_data.max():.3f}], "
                    f"mean={valid_data.mean():.3f}, "
                    f"coverage={len(valid_data)/data.size*100:.1f}%"
                )
            else:
                print(f"  {var}: No valid data")

        print("\n💾 Saved to: test_global_mosaic.nc")
        print("\nNext step: Update configuration for 7 satellites × 18 bands")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import numpy as np

    exit(main())
