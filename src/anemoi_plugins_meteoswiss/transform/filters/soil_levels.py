import earthkit.data as ekd
from anemoi.transform.fields import NewMetadataField
from anemoi.transform.filter import Filter

SOIL_LEVEL_TYPES = ("depthBelowLand", "depthBelowLandLayer")


class _SoilLevelField(NewMetadataField):
    """Field with soil depth metadata, also in the ``mars`` namespace.

    The ``mars`` namespace of soil fields from GRIB has no ``levelist`` key, and
    ``NewMetadataField`` only overrides keys that are already there.
    """

    def metadata(self, *args, **kwargs):
        result = super().metadata(*args, **kwargs)
        if kwargs.get("namespace") == "mars" and not args:
            result = dict(result)
            for key in ("levtype", "levelist"):
                result[key] = self.kwargs[key]
        return result


def soil_depth_mm(field: ekd.Field) -> int:
    """Depth of a soil level (top of the layer for layers) in millimetres."""
    value = field.metadata("scaledValueOfFirstFixedSurface")
    factor = field.metadata("scaleFactorOfFirstFixedSurface")
    depth = value * 10.0 ** (3 - factor)
    if abs(depth - round(depth)) > 1e-6:
        raise ValueError(
            f"Soil depth {value}e-{factor} m is not a whole number of millimetres"
        )
    return int(round(depth))


class SoilDepthLevels(Filter):
    """Label soil levels with their depth in millimetres.

    eccodes truncates the depth of soil levels to whole metres in ``level`` (0.005, 0.02,
    0.06 and 0.18 m all become 0), and the flavour of the ``fdb`` source sets ``levelist``
    to null for them, so all levels of a soil parameter end up with the same variable name.
    This filter sets ``level`` and ``levelist`` to the depth in mm (top of the layer for
    ``depthBelowLandLayer``) and ``levtype`` to ``dp``, so the variables are named
    ``<param>_<depth in mm>``, e.g. ``T_SO_5`` for 0.005 m. Other fields pass through.

    Args:
        param (list[str], optional): Short names to relabel. Defaults to every field on a
            soil level type.
    """

    def __init__(self, param: str | list[str] | None = None):
        self.param = (
            None
            if param is None
            else ([param] if isinstance(param, str) else list(param))
        )

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = ekd.SimpleFieldList()
        for field in data:
            if field.metadata("typeOfLevel", default=None) not in SOIL_LEVEL_TYPES or (
                self.param is not None and field.metadata("shortName") not in self.param
            ):
                out.append(field)
                continue
            depth = soil_depth_mm(field)
            out.append(
                _SoilLevelField(field, level=depth, levelist=depth, levtype="dp")
            )
        return out
