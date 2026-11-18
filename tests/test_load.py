import numpy as np

from metnet.data.load import _crop_and_resample


def _grid(h=8, w=8):
    lats, lons = np.meshgrid(np.linspace(40, 33, h), np.linspace(260, 267, w), indexing="ij")
    return lats, lons


def test_fill_values_become_nan():
    lats, lons = _grid()
    data = np.full((8, 8), -3.0)
    data[4:] = 1.0
    out = _crop_and_resample(data, lats, lons, 30, 45, -110, -90, 1.0)
    assert np.isnan(out[:4]).all()
    assert (out[4:] == 1.0).all()


def test_longitudes_converted_and_cropped():
    lats, lons = _grid()  # 260..267 east == -100..-93
    out = _crop_and_resample(np.ones((8, 8)), lats, lons, 30, 45, -100, -97, 1.0)
    assert out.shape[1] == 4  # lons -100, -99, -98, -97


def test_average_pooling():
    lats, lons = _grid()
    data = np.arange(64, dtype=float).reshape(8, 8)
    out = _crop_and_resample(data, lats, lons, 30, 45, -110, -90, 4.0)
    assert out.shape == (2, 2)
    assert out[0, 0] == data[:4, :4].mean()
