# MetNet Global - Satellite Mosaic Builder

This directory contains tools for creating global satellite mosaics from geostationary satellite data for the MetNet Global project.

## Overview

MetNet Global uses imagery from multiple geostationary satellites to provide global coverage for precipitation nowcasting. This implementation accesses pre-processed satellite data from Source Cooperative's Icechunk archives.

**Related Issue**: [#74 - MetNet Global Paper Implementation](https://github.com/openclimatefix/metnet/issues/74)

## Available Satellites

Data is sourced from [source.coop/bkr/geo](https://source.coop/bkr/geo):

- **GOES-East** (GOES-16/17) - Americas, Atlantic
- **GOES-West** (GOES-18) - Eastern Pacific, Western Americas
- **Himawari-9** - Asia-Pacific, Indian Ocean
- **GK-2A** - East Asia (Korea, Japan, China)
- **Meteosat** (multiple) - Europe, Africa, Indian Ocean

All data is:
- ✅ Processed by Satpy (calibrated L1b)
- ✅ Available at 500m, 1km, 2km resolutions
- ✅ Stored in Icechunk format for efficient access
- ✅ Publicly accessible via S3-compatible API

## Installation

### Requirements

- Python 3.12+ (Python 3.14 has compatibility issues with icechunk)
- icechunk library (requires compilation)

### Setup

```bash
# Create Python 3.12 environment
python3.12 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install xarray zarr icechunk
```

## Usage

### 1. Inspect a Satellite Dataset

```bash
python mosaic_builder.py inspect https://data.source.coop/bkr/geo/goes-east_500m.icechunk/
```

### 2. Load Data for a Specific Time

```python
from mosaic_builder import load_satellite_data

# Load GOES-East data for 2018-01-01
ds = load_satellite_data(
    "https://data.source.coop/bkr/geo/goes-east_500m.icechunk/",
    time="2018-01-01T00:00:00"
)
print(ds)
```

### 3. Create a Mosaic from Multiple Satellites

```bash
python mosaic_builder.py mosaic \
  --stores \
    https://data.source.coop/bkr/geo/goes-east_500m.icechunk/ \
    https://data.source.coop/bkr/geo/goes-west_500m.icechunk/ \
  --time 2018-01-01T00:00:00 \
  --output /tmp/mosaic.nc
```

### 4. Run the Demonstration

```bash
python demo_mosaic_access.py
```

This will:
- Access multiple satellite datasets
- Load data for 2018-01-01
- Display coverage information
- Show projection details

## Current Status

### ✅ Completed
- [x] Access Icechunk stores from Source Cooperative
- [x] Load geostationary satellite data via xarray/zarr
- [x] Handle duplicate timestamps in satellite data
- [x] Query data by time with nearest-neighbor selection
- [x] Basic multi-satellite data loading
- [x] Command-line interface for inspection and mosaicing

### 🚧 In Progress
- [ ] Spatial reprojection to common grid
- [ ] Overlap handling and blending
- [ ] Visualization of coverage and mosaics
- [ ] Integration with all spectral bands

### ❓ Questions for Maintainer

1. **Target Projection**: What should the final mosaic projection be?
   - Equirectangular (lat/lon grid)?
   - Equal-area projection?
   - Keep native geostationary projections?

2. **Overlap Handling**: How should overlapping regions be merged?
   - Priority by satellite?
   - Temporal blending?
   - Quality-based weighting?

3. **Reprojection Tool**: Should we use:
   - `pyresample` for custom reprojection?
   - `satpy.Scene.resample()` for consistency?
   - Keep data in native projections?

## Files

- `mosaic_builder.py` - Main module for accessing and combining satellite data
- `demo_mosaic_access.py` - Demonstration script showing data access
- `test_mosaic.py` - Test suite for mosaicing functionality
- `README.md` - This file

## Data Structure

Each satellite dataset contains:
- **Dimensions**: `time`, `y_geostationary`, `x_geostationary`
- **Variables**: Spectral bands (e.g., `C02` for GOES, `VI006` for GK-2A)
- **Coordinates**: Time, x/y in geostationary projection
- **Metadata**: Orbital parameters, projection info, calibration data

Example:
```
Dataset dimensions: {'time': 6853, 'y_geostationary': 21696, 'x_geostationary': 21696}
Projection center: -75.0° (GOES-East)
Resolution: 500 meters at nadir
Time range: 2018-01-01 to 2022-01-14
```

## Technical Notes

### Python Version
- **Use Python 3.12** - icechunk has compilation issues with Python 3.14
- The `.venv` directory should use Python 3.12

### Duplicate Timestamps
Some satellite datasets have duplicate timestamps from overlapping scans. The code handles this by:
1. Detecting duplicate time values
2. Keeping only unique timestamps
3. Selecting nearest time after deduplication

### S3 Configuration
Source Cooperative uses S3-compatible storage:
- Endpoint: `https://data.source.coop`
- Region: `us-east-1`
- Access: Anonymous (no credentials required)
- Path style: Force path-style addressing

## Contributing

This work is part of the MetNet Global implementation. Contributions welcome!

1. Test with different dates and satellites
2. Implement spatial reprojection
3. Add visualization tools
4. Optimize for large-scale processing

## References

- **MetNet Global Paper**: https://arxiv.org/pdf/2510.13050
- **Source Cooperative**: https://source.coop/bkr/geo
- **GitHub Issue**: https://github.com/openclimatefix/metnet/issues/74
- **Satpy Documentation**: https://satpy.readthedocs.io/

## License

Same as parent metnet repository.
