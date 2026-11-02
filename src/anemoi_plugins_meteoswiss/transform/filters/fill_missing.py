import earthkit.data as ekd
import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter


class FillMissing(Filter):
    """Replace missing values (NaN) of selected fields with a constant or with another field.

    Meant for diagnostics that are undefined at some points, e.g. the height of the 0 degC
    isotherm (HZEROCL) where the whole column above the ground is colder than 0 degC: filling
    with the surface height (HSURF) means "the level is at the ground". The fill field is read
    once from a GRIB file (e.g. the constants file) and must be on the same grid. Fields of
    other parameters pass through unchanged.

    Args:
        param (str | list[str]): Short names of the fields to fill.
        value (float, optional): Constant fill value.
        path (str, optional): GRIB file containing the fill field (instead of ``value``).
        fill_param (str, optional): Short name of the fill field in ``path``. Defaults to ``HSURF``.
    """

    def __init__(
        self,
        param: str | list[str],
        value: float | None = None,
        path: str | None = None,
        fill_param: str = "HSURF",
    ):
        if (value is None) == (path is None):
            raise ValueError("Give exactly one of 'value' or 'path'.")
        self.param = [param] if isinstance(param, str) else list(param)
        self.value = value
        self.path = path
        self.fill_param = fill_param
        self._fill_values = None

    def fill_values(self) -> float | np.ndarray:
        if self.value is not None:
            return self.value
        if self._fill_values is None:
            fields = ekd.from_source("file", self.path).sel(shortName=self.fill_param)
            if len(fields) != 1:
                raise ValueError(
                    f"Expected one {self.fill_param} field in {self.path}, found {len(fields)}."
                )
            self._fill_values = np.asarray(fields[0].values)
        return self._fill_values

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        result = []
        for field in data:
            if field.metadata("shortName") not in self.param:
                result.append(field)
                continue
            values = np.array(field.values)  # flat copy
            missing = np.isnan(values)
            if missing.any():
                fill = self.fill_values()
                if np.ndim(fill) and np.shape(fill) != values.shape:
                    raise ValueError(
                        f"Fill field {self.fill_param} has shape {np.shape(fill)}, "
                        f"{field.metadata('shortName')} has {values.shape}."
                    )
                values[missing] = fill[missing] if np.ndim(fill) else fill
            result.append(new_field_from_numpy(values, template=field))
        return new_fieldlist_from_list(result)
