import numpy as np
import pytest
import rioxarray as rx
import xarray as xr

from tw_wbt.wbt import resample


@pytest.fixture
def data_array():
    xmin, ymin, xmax, ymax = -121, 35, -111, 45
    resolution = 1
    crs = "EPSG:4326"

    x = np.arange(xmin + resolution / 2, xmax, resolution)
    y = np.arange(ymax - resolution / 2, ymin, -resolution)  # flip y downward

    # Create empty 2D array (all zeros)
    data = np.zeros((len(y), len(x)), dtype=np.float32)

    return xr.DataArray(
        data, dims=("y", "x"), coords={"x": x, "y": y}, name="band1"
    ).rio.write_crs(crs)


def test_resample_types(data_array):
    output = resample(data_array, cell_size=2)
    assert isinstance(output, xr.DataArray)
