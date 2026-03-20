import io
from typing import Callable
from typing import Iterator

import earthkit.data as ekd
import xarray as xr
from meteodatalab import data_source
from meteodatalab import grib_decoder


class FieldListDataSource(data_source.DataSource):
    """Adapter exposing an earthkit FieldList as a meteodatalab DataSource."""

    def __init__(self, fieldlist: ekd.FieldList):
        """Initialize the data source.

        Args:
            fieldlist (ekd.FieldList): FieldList to serve via the retrieval API.
        """
        self.fieldlist = fieldlist

    def _retrieve(self, request: dict) -> Iterator[object]:
        """Yield fields matching the request.

        Args:
            request (dict): Selection arguments passed to FieldList.sel(**request).

        Returns:
            Iterator[object]: Iterator over matching fields from the underlying FieldList.
        """
        yield from self.fieldlist.sel(**request)


def meteodatalab_wrapper(
    func: Callable[..., dict[str, xr.DataArray]],
) -> Callable[[ekd.FieldList], ekd.FieldList]:
    """Decorator to wrap a function that processes an ekd.FieldList."""

    def inner(fieldlist: ekd.FieldList) -> ekd.FieldList:
        source = FieldListDataSource(fieldlist)
        result = func(source)
        return _meteodalab_ds_to_fieldlist(result)

    return inner


def to_meteodatalab(fieldlist: ekd.FieldList) -> dict[str, xr.DataArray]:
    """Convert an ekd.FieldList to a dictionary of xarray DataArrays."""
    source = FieldListDataSource(fieldlist)
    return grib_decoder.load(source, {})


def from_meteodatalab(ds: dict[str, xr.DataArray]) -> ekd.FieldList:
    """Convert a dictionary of xarray DataArrays to an ekd.FieldList."""
    return _meteodalab_ds_to_fieldlist(ds)


def _meteodalab_ds_to_fieldlist(ds: dict[str, xr.DataArray]) -> ekd.FieldList:
    with io.BytesIO() as buffer:
        # write data to the buffer
        for da in ds.values():
            if "z" in da.dims and da["z"].size == 1 and bool(da["z"].values[0] is None):
                da = da.squeeze("z", drop=True)
            grib_decoder.save(
                da, buffer, bits_per_value=32
            )  # TODO: find out why we need 32 and 16 leads to precision loss

        # reset the buffer position to the beginning
        buffer.seek(0)

        # read data from the buffer into a FieldList
        fs = ekd.from_source("stream", buffer, read_all=True, lazily=False)

        # somehow read_all does not work correctly, so we need to convert to FieldList
        # to actually have all data loaded in memory and not get IO errors later
        fl = ekd.FieldList.from_fields(fs)

    return fl
