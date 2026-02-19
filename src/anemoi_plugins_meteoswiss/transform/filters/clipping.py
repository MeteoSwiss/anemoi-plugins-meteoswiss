import earthkit.data as ekd
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter


class ClipLateralBoundaries(Filter):
    """A filter to clip fields of ICON-CH to a specified lateral boundary."""

    def __init__(self, strip_idx: int, gridfile: str):
        """Initialize the filter.

        Parameters
        ----------
        strip_idx:
            The maximum lateral boundary strip index to keep.
        gridfile:
            The path to the grid descriptor file.
        """
        self.strip_idx = strip_idx
        self.gridfile = gridfile

        ds = xr.open_dataset(self.gridfile)
        if "refin_c_ctrl" not in ds:
            raise ValueError(
                f"Grid descriptor file {self.gridfile} does not contain 'refin_c_ctrl' variable."
            )

        self.idx = ds["refin_c_ctrl"].assign_attrs(uuidOfHGrid=ds.attrs["uuidOfHGrid"])

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        result = []
        for field in data:
            _field = _clip_field_lateral_boundaries(field, self.strip_idx, self.idx)
            field = new_field_from_numpy(
                _field.values,
                template=field,
                uuidOfHGrid=_field.metadata.get("uuidOfHGrid"),
                numberOfDataPoints=_field.metadata.get("numberOfDataPoints"),
            )
            result.append(field)
        return new_fieldlist_from_list(result)


def _clip_field_lateral_boundaries(
    field: ekd.Field,
    strip_idx: int,
    idx: xr.DataArray,
) -> xr.DataArray:
    from meteodatalab.grib_decoder import _FieldBuffer
    from meteodatalab.grib_decoder import _is_ensemble
    from meteodatalab.operators.clip import clip_lateral_boundary_strip

    buffer = _FieldBuffer(_is_ensemble(field))
    buffer.load(field, None)
    da = buffer.to_xarray()
    return clip_lateral_boundary_strip(da, strip_idx, idx=idx)
