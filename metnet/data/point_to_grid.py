"""Point-source to grid translator for MetNet-3 weather station data.

Translates sparse weather station observations (lat/lon point measurements)
to a dense 2D grid, as described in the MetNet-3 paper:

'For surface variables, we take the OMO station point measurements and map
them to a 4km by 4km pixel in which the station lies. If there are multiple
stations in a given region, we take the average of their measurements.'

The 14 variables used are based on the OMO variable list at:
https://madis.ncep.noaa.gov/sfc_OMO_variable_list.shtml
"""

import numpy as np
import torch
from pyproj import Transformer
from typing import List, Dict

# Best approximation of the 14 OMO input variables using MADIS ASOS
OMO_VARIABLES = [
    "T",        # air temperature (K)
    "TD",       # dewpoint temperature (K)
    "RH",       # relative humidity (%)
    "P",        # station pressure (Pa)
    "DD",       # wind direction (deg)
    "FF",       # wind speed (m/s)
    "U",        # u wind component (m/s)
    "V",        # v wind component (m/s)
    "FFGUST",   # wind gust (m/s)
    "VIS",      # visibility (m)
    "PCPRATE",  # precipitation rate (kg/m²/s)
    "PCP1H",    # accumulated precip 1h (m)
    "Q",        # specific humidity (kg/kg)
    "ALTSE",    # altimeter pressure (Pa)
]


class PointToGridTranslator:
    """
    Translates sparse weather station observations to a dense grid.

    Supports Lambert Conformal Conic projection, matches CONUS setup
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution_km: float = 4.0,
        variables: List[str] = OMO_VARIABLES,
        fill_value: float = 0.0,
    ):
        """
        Initialize the translator.

        Args:
            grid_height: Number of grid points in y direction
            grid_width: Number of grid points in x direction
            lat_min: Minimum latitude of grid domain
            lat_max: Maximum latitude of grid domain
            lon_min: Minimum longitude of grid domain
            lon_max: Maximum longitude of grid domain
            resolution_km: Grid resolution in km, 4.0 as per MetNet-3
            variables: List of variable names to extract from station data
            fill_value: Value for grid cells with no station, default 0.0
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution_km = resolution_km
        self.variables = variables
        self.fill_value = fill_value
        self.num_variables = len(variables)

        self.transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 lat/lon
            "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=40 +lon_0=-96",  # CONUS LCC
            always_xy=True,
        )
        self.x_min, self.y_min = self.transformer.transform(lon_min, lat_min)

    def _latlon_to_pixel(self, lat: float, lon: float):
        """Convert lat/lon to grid pixel coordinates."""
        x, y = self.transformer.transform(lon, lat)
        col = int((x - self.x_min) / (self.resolution_km * 1000))
        row = int((y - self.y_min) / (self.resolution_km * 1000))
        return row, col

    def translate(
        self,
        stations: List[Dict],
    ) -> torch.Tensor:
        """
        Translate station observations to a dense grid.

        Args:
            stations: List of dicts, each with keys:
                - 'lat': station latitude
                - 'lon': station longitude
                - variable names matching self.variables with float values

        Returns:
            Dense grid tensor of shape [num_variables, grid_height, grid_width]
            with fill_value where no station is present
        """
        # Initialize grid and count grids for averaging
        grid = np.full(
            (self.num_variables, self.grid_height, self.grid_width),
            self.fill_value,
            dtype=np.float32,
        )
        counts = np.zeros(
            (self.grid_height, self.grid_width),
            dtype=np.int32,
        )

        for station in stations:
            lat = station["lat"]
            lon = station["lon"]

            # Skip stations outside grid bounds
            if not (self.lat_min <= lat <= self.lat_max):
                continue
            if not (self.lon_min <= lon <= self.lon_max):
                continue

            row, col = self._latlon_to_pixel(lat, lon)

            # Skip if outside grid
            if not (0 <= row < self.grid_height and 0 <= col < self.grid_width):
                continue

            # Add station values
            for i, var in enumerate(self.variables):
                if var in station and station[var] is not None:
                    value = station[var]
                    if np.isnan(value):
                        continue
                    if counts[row, col] == 0:
                        grid[i, row, col] = value
                    else:
                        # Running sum for averaging
                        grid[i, row, col] += value

            counts[row, col] += 1

        # Average where multiple stations fall in same pixel
        multiple = counts > 1
        if np.any(multiple):
            grid[:, multiple] /= counts[multiple]

        return torch.from_numpy(grid)