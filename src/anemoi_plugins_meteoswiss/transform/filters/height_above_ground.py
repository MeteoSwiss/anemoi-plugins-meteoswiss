from collections import defaultdict

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.filter import Filter


class HeightAboveGround(Filter):
    """Convert heights above sea level to heights above ground.

    The surface height field (``surface``, e.g. ``HSURF``) is subtracted from every field of
    ``param`` with the same valid time. The surface field is consumed (not passed on); other
    fields pass through unchanged.

    Args:
        param (list[str]): Short names of the height fields above sea level, e.g. ``[HHL]``.
        surface (str): Short name of the surface height field. Defaults to ``HSURF``.
        name (str): Format of the output short name, with ``{param}`` replaced by the input
            short name. Defaults to ``{param}_AGL``.
        keep (bool): Also pass the input heights above sea level on. Defaults to ``True``.
    """

    def __init__(
        self,
        param: str | list[str],
        surface: str = "HSURF",
        name: str = "{param}_AGL",
        keep: bool = True,
    ):
        self.param = [param] if isinstance(param, str) else list(param)
        self.surface = surface
        self.name = name
        self.keep = keep
        if keep and any(name.format(param=p) == p for p in self.param):
            raise ValueError(
                f"keep=True needs an output name different from the input, got name={name!r}"
            )

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        surfaces = {}
        heights = defaultdict(list)
        out = ekd.SimpleFieldList()
        for field in data:
            short_name = field.metadata("shortName")
            valid_datetime = field.metadata("valid_datetime")
            if short_name == self.surface:
                if valid_datetime in surfaces:
                    raise ValueError(
                        f"More than one {self.surface} field at {valid_datetime}"
                    )
                surfaces[valid_datetime] = field
                continue
            if short_name in self.param:
                heights[valid_datetime].append(field)
                if not self.keep:
                    continue
            out.append(field)

        for valid_datetime, fields in heights.items():
            if valid_datetime not in surfaces:
                raise ValueError(
                    f"No {self.surface} field at {valid_datetime} to subtract from {self.param}"
                )
            surface = surfaces[valid_datetime].values
            for field in fields:
                new_name = self.name.format(param=field.metadata("shortName"))
                out.append(
                    new_field_from_numpy(
                        field.values - surface,
                        template=field,
                        shortName=new_name,
                        param=new_name,
                    )
                )
        return out
