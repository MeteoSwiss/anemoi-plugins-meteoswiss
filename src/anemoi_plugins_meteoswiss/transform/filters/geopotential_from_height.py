import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter

G = 9.80665  # m/s^2

GEOPOTENTIAL_UNITS = "m2 s-2"


class GeopotentialFromHeight(Filter):
    """A filter to convert height fields to geopotential fields."""

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = []
        for field in data:
            # the parameter component, not the raw GRIB shortName: grids and flavours mask the raw GRIB keys
            name = field.get("parameter.variable")
            if name in ["HHL", "h"]:  # h is the destaggered version of HHL for which we have no definition
                out.append(self._geopotential(field, "FI"))
            elif name == "HSURF":
                out.append(self._geopotential(field, "FIS"))
            else:
                out.append(field)
        return new_fieldlist_from_list(out)

    @staticmethod
    def _geopotential(field: ekd.Field, name: str) -> ekd.Field:
        # param sets the parameter component (the variable name); shortName is kept as a label
        return new_field_from_numpy(
            field.values * G, template=field, shortName=name, param=name, units=GEOPOTENTIAL_UNITS
        )
