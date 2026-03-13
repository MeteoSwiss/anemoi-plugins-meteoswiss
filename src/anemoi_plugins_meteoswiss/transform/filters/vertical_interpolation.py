import logging

import earthkit.data as ekd
import xarray as xr
from anemoi.transform.filter import Filter
from earthkit.meteo import vertical
from meteodatalab.operators import vertical_extrapolation
from meteodatalab.operators import vertical_interpolation


SFC_VCOORD_TYPES = [
    "surface",
    "heightAboveGround",
    "meanSea",
]

LOG = logging.getLogger(__name__)


class InterpK2P(Filter):
    """
    A filter to perform vertical interpolation from model to pressure levels.

    Some parameters need to be in the data: P.

    """

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
        # metadata of pressure field (needed to make time info consistent)
        p_md = data.sel(shortName="P")[0].metadata()

        # pressure field
        pressure = data.sel(shortName="P").to_xarray()["P"]
        # ensure all values at the top-most level are below 5000 hPa
        # else the interpolation will leave NaNs at the top
        pressure[{"level": 0}] = pressure[{"level": 0}].where(
            pressure[{"level": 0}] < 5000, 5000 - 1e-5
        )

        out = ekd.SimpleFieldList()

        for fields in data.group_by("shortName"):
            _field = fields[0]
            param = _field.metadata().get("shortName")

            # make sure constant variables have same time coordinates as other fields
            if param in ("HHL", "h", "HSURF"):
                _field = _field.copy(
                    metadata=_field.metadata().override(
                        date=p_md.get("date"),
                        time=p_md.get("time"),
                        step=p_md.get("step"),
                    )
                )

            # skip pressure and fields on unsuited vertical levels
            if param == "P" or _field.metadata().get("typeOfLevel") != "generalVerticalLayer":
                continue

            da = fields.to_xarray()[param]
            LOG.info("Interpolating %s to pressure levels %s", param, self.levels)
            pinterp = vertical.interpolate_to_pressure_levels(
                da,
                pressure,
                self.levels,
                "hPa",
                "log",
                "level",
            )

            # reconstruct fields
            _construct_fields(out, pinterp, _field)

        return out

def _construct_fields(
    out: ekd.SimpleFieldList,
    da: xr.DataArray,
    template: ekd.core.fieldlist.Field,
):
    for i, lev in enumerate(da.level.values):
        field_da = da.isel(level=i)
        out.append(
            template.clone(
                values=field_da.values,
                typeOfLevel="isobaricInhPa",
                level=int(lev),
                levelist=int(lev),
            )
        )


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
