#!/usr/bin/env python3
"""Check available bands in satellite datasets."""

import sys

sys.path.insert(0, "/Users/sagarpillai/metnet/.venv_py312/lib/python3.12/site-packages")

from datetime import datetime

from mosaic_builder import SATELLITE_CONFIG, load_satellite_data


def check_available_bands():
    """Check what bands are available in the satellite data."""
    print("\n" + "=" * 80)
    print("CHECKING AVAILABLE BANDS")
    print("=" * 80)

    # Check a few satellites
    satellites_to_check = ["goes-east", "goes-west", "himawari"]
    target_time = datetime(2018, 1, 1, 0, 0)

    for sat_name in satellites_to_check:
        if sat_name not in SATELLITE_CONFIG:
            print(f"\n❌ {sat_name} not in config")
            continue

        url = SATELLITE_CONFIG[sat_name]["url"]
        print(f"\n📡 {sat_name}")
        print(f"   URL: {url}")

        try:
            ds = load_satellite_data(url, target_time)
            if ds is None:
                print(f"   ⚠️  No data available at {target_time}")
                continue

            print("   ✅ Data loaded")
            print(f"   Variables: {list(ds.data_vars)}")
            print(f"   Dimensions: {dict(ds.dims)}")

        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    check_available_bands()
