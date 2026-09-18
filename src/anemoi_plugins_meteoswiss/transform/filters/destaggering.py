from collections import defaultdict

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.filter import Filter


class Destagger(Filter):
    """Destagger fields from half levels to full levels in the vertical.

    Full level ``k`` is the mean of half levels ``k`` and ``k + 1``, so both must be in
    the input. Fields of other parameters pass through unchanged; the half-level inputs
    of destaggered parameters are dropped.

    Args:
        param (list[str]): Short names of the fields on half levels, e.g. ``[W]``.
        levels (list[int], optional): Full levels to output. Defaults to every level ``k``
            for which half levels ``k`` and ``k + 1`` are both in the input.
    """

    def __init__(self, param: str | list[str], levels: list[int] | None = None):
        self.param = [param] if isinstance(param, str) else list(param)
        self.levels = None if levels is None else {int(k) for k in levels}

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = ekd.SimpleFieldList()
        half_levels = defaultdict(dict)
        for field in data:
            name = field.metadata("shortName")
            if name in self.param:
                key = (name, field.metadata("valid_datetime"))
                half_levels[key][int(field.metadata("level"))] = field
            else:
                out.append(field)

        for (name, valid_datetime), fields in half_levels.items():
            levels = sorted(k for k in fields if k + 1 in fields)
            if self.levels is not None:
                missing = self.levels - set(levels)
                if missing:
                    raise ValueError(
                        f"Cannot destagger {name} at {valid_datetime} to levels {sorted(missing)}: "
                        f"half levels k and k+1 are needed, got {sorted(fields)}"
                    )
                levels = sorted(self.levels)
            for k in levels:
                values = 0.5 * (fields[k].values + fields[k + 1].values)
                out.append(
                    new_field_from_numpy(
                        values, template=fields[k], typeOfLevel="generalVerticalLayer"
                    )
                )
        return out
