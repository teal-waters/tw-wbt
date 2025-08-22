"""Wrappers for WhiteBoxTools functions."""

import geopandas as gpd
import xarray as xr
from whitebox import WhiteboxTools

from tw_wbt.wrapper import wrap_wbt

wbt = WhiteboxTools()


resample = wrap_wbt(wbt.resample)
gaussian_filter = wrap_wbt(wbt.gaussian_filter)
fill_burn = wrap_wbt(wbt.fill_burn)
breach_depressions_least_cost = wrap_wbt(wbt.breach_depressions_least_cost)
fill_depressions = wrap_wbt(wbt.fill_depressions)
quinn_flow_accumulation = wrap_wbt(wbt.quinn_flow_accumulation)
d_inf_flow_accumulation = wrap_wbt(wbt.d_inf_flow_accumulation)
d8_flow_accumulation = wrap_wbt(wbt.d8_flow_accumulation)
slope = wrap_wbt(wbt.slope)


def burn(
    xarr: xr.DataArray | xr.Dataset, lines: gpd.GeoDataFrame
) -> xr.DataArray | xr.Dataset:
    """Burn lines into a Xarray object.

    Args:
        xarr: Input DataArray or Dataset
        lines: Lines to burn

    Returns:
        The input with lines burned in.

    """
    burned_xarr = fill_burn(xarr, lines)
    return xarr.where(burned_xarr >= xarr, burned_xarr)
