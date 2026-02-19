import earthkit.data as ekd
import xarray as xr
from anemoi.transform.filter import Filter

from mch_anemoi_plugins.helpers import from_meteodatalab
from mch_anemoi_plugins.helpers import to_meteodatalab

G = 9.80665
R_D = 287.053


class OmegaFromW(Filter):
    """A filter to convert vertical velocity in m/s to omega in Pa/s."""

    def __init__(self):
        """Initialize the filter.

        Parameters
        ----------
        strip_idx:
            The maximum lateral boundary strip index to keep.
        gridfile:
            The path to the grid descriptor file.
        """
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        ds = to_meteodatalab(data)
        ds["OMEGA"] = omega_from_w(ds.pop("W"), ds["T"], ds["P"])
        return from_meteodatalab(ds)


def omega_from_w(w: xr.DataArray, t: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    """Convert vertical velocity in m/s to omega in Pa/s."""
    from meteodatalab import metadata

    rho = p / (R_D * t)
    out = -w * rho * G
    out.attrs = metadata.override(w.attrs["metadata"], shortName="OMEGA")
    return out
