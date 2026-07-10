import io

import earthkit.data as ekd
from anemoi.transform.filter import Filter
from meteodatalab import data_source
from meteodatalab import grib_decoder
from meteodatalab.operators import destagger


def to_meteodatalab(fieldlist: ekd.FieldList) -> dict:
    source = _FieldListDataSource(fieldlist)
    return grib_decoder.load(source, {})


def from_meteodatalab(ds: dict) -> ekd.FieldList:
    return _meteodatalab_ds_to_fieldlist(ds)


class _FieldListDataSource(data_source.DataSource):
    def __init__(self, fieldlist: ekd.FieldList):
        self.fieldlist = fieldlist

    def _retrieve(self, request: dict):
        yield from self.fieldlist.sel(**request)


def _meteodatalab_ds_to_fieldlist(ds: dict) -> ekd.FieldList:
    with io.BytesIO() as buffer:
        for da in ds.values():
            if "z" in da.dims and da["z"].size == 1 and bool(da["z"].values[0] is None):
                da = da.squeeze("z", drop=True)
            grib_decoder.save(da, buffer, bits_per_value=32)
        buffer.seek(0)
        fs = ekd.from_source("stream", buffer, read_all=True, lazily=False)
        fl = ekd.FieldList.from_fields(fs)
    return fl


class Destagger(Filter):
    """A filter to destagger fields using meteodata-lab."""

    def __init__(self, param_dim: dict[str, str]):
        """Initialize the filter.

        Parameters
        ----------
        param_dim:
            Dictionary mapping parameter names to dimensions along which to destagger.
        """
        self.param_dim = param_dim
        self.param = list(param_dim.keys())

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        ds = to_meteodatalab(data)
        for name, dim in self.param_dim.items():
            if name not in ds:
                raise ValueError(f"Field {name} not found in dataset.")
            ds[name] = destagger.destagger(ds[name], dim)
        data = from_meteodatalab(ds)
        return data

    def backward_transform(self):
        raise NotImplementedError("Destagger is not reversible.")
