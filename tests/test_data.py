import torch
from metnet.data.point_to_grid import PointToGridTranslator


def make_translator(
    grid_height=10,
    grid_width=10,
    # Domain sized to match a 10x10 grid at the default 4km resolution
    # (40km x 40km), with SAMPLE_STATION (40.0, -95.0) landing near the
    # center. A domain much larger than grid_height/width * resolution_km
    # will place stations outside the array, so translate() silently
    # drops them instead of raising.
    lat_min=39.8,
    lat_max=40.2,
    lon_min=-95.25,
    lon_max=-94.5,
    fill_value=0.0,
):
    return PointToGridTranslator(
        grid_height=grid_height,
        grid_width=grid_width,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        fill_value=fill_value,
    )


SAMPLE_STATION = {
    "lat": 40.0,
    "lon": -95.0,
    "T": 295.0,
    "TD": 288.0,
    "RH": 70.0,
    "P": 101325.0,
    "DD": 180.0,
    "FF": 5.0,
    "U": 0.0,
    "V": -5.0,
    "FFGUST": 8.0,
    "VIS": 10000.0,
    "PCPRATE": 0.0,
    "PCP1H": 0.0,
    "Q": 0.01,
    "ALTSE": 101325.0,
}


def test_output_shape():
    output = make_translator().translate([SAMPLE_STATION])
    assert output.shape == (14, 10, 10)


def test_fill_value():
    output = make_translator().translate([])
    assert (output == 0.0).all()


def test_nan_handling():
    station = {**SAMPLE_STATION, "T": float("nan")}
    output = make_translator().translate([station])
    assert output[0].max() == 0.0


def test_outside_bounds():
    station = {**SAMPLE_STATION, "lat": 60.0}
    output = make_translator().translate([station])
    assert (output == 0.0).all()


def test_averaging():
    translator = make_translator(fill_value=-999.0)  # distinct fill value
    station1 = {**SAMPLE_STATION, "T": 290.0}
    station2 = {**SAMPLE_STATION, "T": 300.0}  # exact same lat/lon
    output = translator.translate([station1, station2])
    non_fill = output[0][output[0] != -999.0]
    assert torch.isclose(non_fill, torch.tensor(295.0)).any()
