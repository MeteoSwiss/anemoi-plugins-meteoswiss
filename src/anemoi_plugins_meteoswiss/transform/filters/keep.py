import logging

import earthkit.data as ekd
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)


class Keep(Filter):
    """Keep only fields matching the given parameters, dropping everything else.

    The inverse of anemoi-transform's ``drop`` filter: instead of naming what
    to remove, name what to keep.

    Parameters
    ----------
    param:
        Names of the parameters to keep. Fields whose ``parameter.variable``
        metadata (the GRIB ``shortName`` for GRIB fields) is not in this list are dropped.
    """

    def __init__(self, param: str | list[str]):
        self.param = [param] if isinstance(param, str) else list(param)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        params = [f.get("parameter.variable") for f in data]
        kept = [f for f, p in zip(data, params) if p in self.param]

        dropped_params = sorted({p for p in params if p not in self.param})
        LOG.info("Dropping %d fields, param=%s", len(dropped_params), dropped_params)

        missing_params = sorted(set(self.param) - set(params))
        if missing_params:
            LOG.warning("Requested params=%s are missing", missing_params)
        return new_fieldlist_from_list(kept)
