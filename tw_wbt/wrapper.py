"""Functions to wrap existing WhiteboxTools functions."""

from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import geopandas as gpd
import rioxarray as rx
import xarray as xr


def wrap_wbt(wbt_function: Callable) -> Callable:
    """Wrap the given function so it will accept Xarray & GeoPandas objects.

    Args:
        wbt_function:

    Returns:
        The wrapper function.

    """

    @wraps(wbt_function)
    def _wrapper(*args: str, **kwargs: str) -> xr.DataArray:
        with TemporaryDirectory() as temp_dir:
            processed_args = [
                save_object_to_file(arg, Path(temp_dir) / str(i))
                for i, arg in enumerate(args)
            ]
            processed_kwargs = {
                k: save_object_to_file(v, Path(temp_dir) / k) for k, v in kwargs.items()
            }

            # Assuming all outputs are tiff?
            temp_output_file = Path(temp_dir) / "output.tif"

            wbt_function(*processed_args, output=temp_output_file, **processed_kwargs)
            return xr.DataArray(rx.open_rasterio(temp_output_file))

    return _wrapper


def save_object_to_file(
    object: xr.DataArray | xr.Dataset | gpd.GeoDataFrame | Path | str, temp_stem: Path
) -> Path | str:
    """Save the given object to a file.

    Args:
        object:
        temp_stem:

    Returns:
        The path of the object on disk.

    """
    if isinstance(object, xr.Dataset) or isinstance(object, xr.DataArray):
        output_file = temp_stem.with_suffix(".tif")
        save_xarray_to_wbt_tiff(object, output_file)
        return output_file
    elif isinstance(object, gpd.GeoDataFrame):
        output_file = temp_stem.with_suffix(".shp")
        object.to_file(output_file)
        return output_file
    else:
        return object


def save_xarray_to_wbt_tiff(xarr: xr.DataArray | xr.Dataset, path: Path) -> Path:
    """Save an xarray DataArray or Dataset to a TIFF.

    Args:
        xarr: A 2 or 3 dimensional DataArray or 2-d Dataset.
        path: The path to save the file.

    Returns:
        The input path

    """
    wbt_opts = {
        "driver": "GTiff",
        "compress": "none",  # No compression
        "tiled": False,  # Strip-based, not tiled
        "interleave": "band",  # Band interleave
        "BIGTIFF": "IF_NEEDED",  # Use BigTIFF only if necessary
    }
    xarr.rio.to_raster(path, **wbt_opts)
    return path
