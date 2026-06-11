import earthkit.data as ekd
from anemoi.transform.filter import Filter


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
        from meteodatalab.operators import destagger
        from anemoi_plugins_meteoswiss.helpers import from_meteodatalab
        from anemoi_plugins_meteoswiss.helpers import to_meteodatalab
        
        ds = to_meteodatalab(data)
        for name, dim in self.param_dim.items():
            if name not in ds:
                raise ValueError(f"Field {name} not found in dataset.")
            ds[name] = destagger.destagger(ds[name], dim)
        data = from_meteodatalab(ds)
        return data

    def backward_transform(self):
        raise NotImplementedError("Destagger is not reversible.")
