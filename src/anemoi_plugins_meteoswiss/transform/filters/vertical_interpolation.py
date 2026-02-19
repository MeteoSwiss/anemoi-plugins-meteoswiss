import logging

import earthkit.data as ekd
import xarray as xr
from anemoi.transform.filter import Filter
from meteodatalab.operators import vertical_extrapolation
from meteodatalab.operators import vertical_interpolation

from mch_anemoi_plugins.helpers import from_meteodatalab
from mch_anemoi_plugins.helpers import to_meteodatalab

SFC_VCOORD_TYPES = [
    "surface",
    "heightAboveGround",
    "meanSea",
]

LOG = logging.getLogger(__name__)


class InterpK2P(Filter):
    """A filter to perform vertical interpolation from model to pressure levels."""

    def __init__(
        self,
        levels: list[float],
        ext_levels: list[float] = [],
    ):
        """Initialize the filter.

        Parameters
        ----------
        levels: list of numbers
            The pressure levels to interpolate to, in hPa.
        ext_levels: list of numbers, optional
            The pressure levels to extrapolate to below the surface, in hPa.
        """

        super().__init__()

        self.levels = levels
        self.ext_levels = ext_levels

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        ds = to_meteodatalab(data)

        # make sure constant variables have same time coordinates as other fields
        time_coords = {
            k: ds["P"].coords[k] for k in ["ref_time", "lead_time", "valid_time"]
        }
        for var in [
            "HSURF",
            "HHL",
            "h",
        ]:  # for some reason 'HHL' is sometimes decoded as 'h'
            if var not in ds:
                continue
            ds[var] = xr.DataArray(
                ds[var].values.repeat(ds["P"].shape[2], axis=2),
                dims=ds[var].dims,
                coords=dict(ds[var].coords) | time_coords,
                attrs=ds[var].attrs,
            )

        ds = _interpolate_to_pressure_levels(
            ds,
            ds.pop("P"),
            self.levels,
            self.ext_levels,
        )

        data = from_meteodatalab(ds)

        return data


def _interpolate_to_pressure_levels(
    ds: dict[str, xr.DataArray],
    pressure: xr.DataArray,
    p_lev: list[float],
    p_ex_lev: list[float],
) -> dict[str, xr.DataArray]:
    """Interpolate to pressure levels and extrapolate below the surface where needed."""

    # ensure all values at the top-most level are below 5000 hPa
    # else the interpolation will leave NaNs at the top
    pressure[{"z": 0}] = pressure[{"z": 0}].where(
        pressure[{"z": 0}] < 5000, 5000 - 1e-5
    )

    for name, field in ds.items():
        if field.attrs.get("vcoord_type", "") != "model_level":
            continue
        LOG.info("Interpolating %s to pressure levels %s", name, p_lev)

        res = vertical_interpolation.interpolate_k2p(
            field, "linear_in_lnp", pressure, p_lev, "hPa"
        )
        for p in p_ex_lev:
            idx = {"z": p_lev.index(p)}
            if name == "T":
                extrap_res = vertical_extrapolation.extrapolate_temperature_sfc2p(
                    ds["T_2M"], ds["HSURF"], ds["PS"], p * 100
                )
            elif name == "FI":
                extrap_res = vertical_extrapolation.extrapolate_geopotential_sfc2p(
                    ds["HSURF"], ds["T_2M"], ds["PS"], p * 100
                )
            else:
                extrap_res = vertical_extrapolation.extrapolate_k2p(field, p * 100)
            res[idx] = res[idx].where(
                res[idx].notnull(), extrap_res.squeeze().assign_coords(z=p)
            )
        ds[name] = res

    # remove surface fields used for extrapolation
    del ds["HSURF"]
    del ds["PS"]
    del ds["T_2M"]

    return ds
