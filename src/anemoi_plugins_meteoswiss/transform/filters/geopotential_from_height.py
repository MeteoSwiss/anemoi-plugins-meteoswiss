import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.filter import Filter

G = 9.80665  # m/s^2


class GeopotentialFromHeight(Filter):
    """A filter to convert height fields to geopotential fields."""

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = ekd.SimpleFieldList()
        for field in data:
            if field.metadata("shortName") in [
                "HHL",
                "h",
            ]:  # h is the destaggered version of HHL for which we have no definition
                out.append(
                    new_field_from_numpy(
                        field.values * G, template=field, shortName="FI", param="FI"
                    )
                )
            elif field.metadata("shortName") == "HSURF":
                # somehow this is not updating the values...
                out.append(
                    new_field_from_numpy(
                        field.values * G, template=field, shortName="FIS", param="FIS"
                    )
                )
            else:
                out.append(field)
        return out
