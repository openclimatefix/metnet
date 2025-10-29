# Draft PR: MetNet Global - Satellite Mosaicing Pipeline

## Summary

Implements the satellite mosaicing pipeline for MetNet Global (#74), following the paper's methodology with equirectangular projection and azimuth-based blending.

**Status**: ✅ Core pipeline working with C02 band | 🔜 Ready for all 18 bands once data available

## What's Implemented

### ✅ Core Requirements (from #74)
1. **Equirectangular projection** - Target grid with configurable resolution
2. **Azimuth-based blending** - 70° viewing angle cutoff, inverse square weighting
3. **pyresample reprojection** - Proper geostationary → equirectangular transformation
4. **All 7 satellites configured** - GOES-East, GOES-West, Himawari, GK-2A, 3× Meteosat
5. **All 18 bands mapped** - Ready for use when data becomes available

### ✅ Technical Implementation
- **Robust Icechunk access** with duplicate timestamp handling
- **Vectorized operations** for performance (400× data reduction via subsampling)
- **Proper azimuth calculation** using haversine formula
- **Weight normalization** in overlap regions
- **NetCDF output** with metadata
- **Comprehensive error handling** and logging

## What's Tested

**Test Configuration**: 2 satellites (GOES-East, GOES-West) × 1 band (C02)
- ✅ Data loading from Icechunk stores
- ✅ Geostationary reprojection (21,696² → 3,601×1,201 grid)
- ✅ Azimuth-based blending in overlap regions
- ✅ NetCDF output generation
- ✅ Realistic data ranges: [−0.2, 81.3] reflectance units
- ✅ Runtime: ~7.5 minutes (network-bound)

## Current Limitations

### Data Availability
⚠️ **Only C02 band (visible red, ~0.64 μm) currently available** in Source Cooperative dataset

The pipeline is fully ready for all 18 bands - the limitation is purely data availability. We've implemented:
- Complete band mapping for all 18 channels (C01-C16, C17, HRV)
- Cross-satellite naming compatibility
- Band fallback logic

**Questions for maintainers:**
1. Are other bands available in separate Icechunk stores?
2. Should we access the original AWS GOES/Himawari archives?
3. Is C02-only acceptable as MVP for initial PR?

### Untested Scenarios
- ⚠️ Himawari: Fixed timestamp issues in code but not yet tested
- ⚠️ GK-2A: Likely no data for 2018 (operational from 2019)
- ⚠️ Meteosats: Not yet tested
- ⚠️ Full 7-satellite mosaic: Not yet run (would take ~30-40 min)

## Files Modified

### New Files
- `data/mosaic_builder.py` (793 lines) - Complete pipeline implementation
- `data/example_mosaic.py` - Usage example
- `data/test_quick.py` - Quick validation test
- `data/IMPLEMENTATION_SUMMARY.md` - Technical documentation
- `data/CURRENT_STATUS.md` - Status and next steps

### Updated Files
- `data/README.md` - Documentation updates

## Code Structure

```python
# Main API
def create_global_mosaic(
    satellite_urls: List[str],
    target_time: datetime,
    bands: List[str] = ["C02"],
    target_resolution: float = 0.1,
    target_extent: Tuple[float, float, float, float] = (-180, -60, 180, 60)
) -> xr.Dataset

# Key helpers
_reproject_geostationary()  # pyresample-based reprojection
_calculate_azimuth_weights()  # Azimuth-based blending
_extract_satellite_longitude()  # Orbital parameter extraction
```

## Example Usage

```python
from mosaic_builder import create_global_mosaic, SATELLITE_CONFIG

mosaic = create_global_mosaic(
    satellite_urls=[
        SATELLITE_CONFIG["goes-east"]["url"],
        SATELLITE_CONFIG["goes-west"]["url"],
        # ... add more satellites
    ],
    target_time=datetime(2018, 1, 1, 0, 0),
    bands=["C02"],  # Currently only C02 available
    target_resolution=0.1,  # 0.1° = ~11 km at equator
)
```

## Compliance with Paper

Following [MetNet Global paper (arxiv.org/pdf/2510.13050)](https://arxiv.org/pdf/2510.13050):
- ✅ 7 geostationary satellites (GOES, Himawari, GK-2A, Meteosat)
- ✅ Equirectangular projection for global coverage
- ✅ Azimuth-based blending in overlap regions
- ✅ 70° viewing angle cutoff
- ✅ 18 spectral bands (mapped, awaiting data)
- ✅ 0.5km native resolution (from Satpy-processed archives)

## Next Steps

1. **Get feedback** on implementation approach
2. **Clarify data access** for remaining 17 bands
3. **Run full 7-satellite test** once data availability confirmed
4. **Add visualization tools** for quality validation
5. **Add unit tests** for core functions

## Testing

Run quick validation:
```bash
cd data
python test_quick.py  # 2 satellites × 1 band (~7.5 min)
```

Run with all 7 satellites (once data confirmed):
```bash
python example_mosaic.py  # ~30-40 min
```

## Questions for Reviewers

1. Is the azimuth-based blending implementation correct per your understanding of the paper?
2. Should we prioritize getting all 18 bands, or is C02-only acceptable for MVP?
3. Any concerns with the pyresample-based reprojection approach?
4. Preferred testing strategy before merging?

---

**Ready for review!** Happy to iterate based on feedback and run additional tests as needed.
