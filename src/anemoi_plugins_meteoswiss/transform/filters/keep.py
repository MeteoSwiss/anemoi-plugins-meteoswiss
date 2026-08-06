import earthkit.data as ekd
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter


class Keep(Filter):
    """Keep only fields matching the given parameters, dropping everything else.

    The inverse of anemoi-transform's ``drop`` filter: instead of naming what
    to remove, name what to keep.

    Parameters
    ----------
    param:
        Names of the parameters to keep. Fields whose ``param`` metadata is
        not in this list are dropped.
    """

    def __init__(self, param: str | list[str]):
        self.param = [param] if isinstance(param, str) else list(param)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        return new_fieldlist_from_list(
            [field for field in data if field.metadata("param") in self.param]
        )
