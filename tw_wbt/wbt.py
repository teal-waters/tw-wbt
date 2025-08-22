"""Wrappers for WhiteBoxTools functions."""

from whitebox import WhiteboxTools

from tw_wbt.wrapper import wrap_wbt

# from xarray import DataArray
# from xarray import Dataset

wbt = WhiteboxTools()


resample = wrap_wbt(wbt.resample)


# def resample(
#    input: DataArray | Dataset | list[DataArray] | list[Dataset], **kwargs: dict
# ) -> DataArray:
#    return _resample(input, **kwargs)
