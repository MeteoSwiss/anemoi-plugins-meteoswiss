import logging
from typing import Literal

import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.transform.filter import Filter
from earthkit.meteo import vertical

SFC_VCOORD_TYPES = [
    "surface",
    "heightAboveGround",
    "meanSea",
]

LOG = logging.getLogger(__name__)


class InterpK2P(Filter):
    """
    A filter to perform vertical interpolation from model to pressure levels.

    Some parameters need to be in the data: P, PS, HSURF/h, T_2M.

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
        # topography height
        hsurf = data.sel(shortName="HSURF").to_xarray()["HSURF"]
        # 2m temperature
        t2m = data.sel(shortName="T_2M").to_xarray()["T_2M"]
        # surface pressure
        ps = data.sel(shortName="PS").to_xarray()["PS"]

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
            if (
                param == "P"
                or _field.metadata().get("typeOfLevel") != "generalVerticalLayer"
            ):
                continue

            da = fields.to_xarray()[param]
            LOG.info("Interpolating %s to pressure levels %s", param, self.levels)
            interp = vertical.interpolate_to_pressure_levels(
                da,
                pressure,
                self.levels,
                "hPa",
                "log",
                "level",
            )

            LOG.info(
                "Extrapolating %s below surface for pressure levels %s",
                param,
                self.ext_levels,
            )
            for p in self.ext_levels:
                idx = {"level": self.levels.index(p)}
                if param == "T":
                    extrap = extrapolate_temperature_sfc2p(t2m, hsurf, ps, p * 100)
                elif param == "FI":
                    extrap = extrapolate_geopotential_sfc2p(hsurf, t2m, ps, p * 100)
                else:
                    extrap = extrapolate_k2p(da, p * 100)
                interp[idx] = interp[idx].where(
                    interp[idx].notnull(), extrap.squeeze().assign_coords(level=p)
                )

            # reconstruct fields
            _construct_fields(out, interp, _field)

        return out


def _construct_fields(
    out: ekd.SimpleFieldList,
    da: xr.DataArray,
    template: ekd.core.fieldlist.Field,
):
    for i, lev in enumerate(da.level.values):
        field_da = da.isel(level=i)
        # clone() allows to set inconsistent metadata: switching to isobaricInPa
        # means that eccodes exposes level (not the mars keyword levelist) in the
        # metadata - but for some reason anemoi-datasets uses levelist downstream
        # to determine the levels (can we change that? is it configurable?)
        out.append(
            template.clone(
                values=field_da.values,
                typeOfLevel="isobaricInhPa",
                level=int(lev),
                levelist=int(lev),
            )
        )


###
### Extrapolation functions copied from
###    https://github.com/MeteoSwiss/meteodata-lab/blob/main/src/meteodatalab/operators/vertical_extrapolation.py
###
### Will be moved to using earthkit-meteo as soon as functionality is implemented there
###
### --->

LAPSE_RATE = 0.0065  # K m^-1
H1 = 2000.0
H2 = 2500.0
T1 = 298.0
g = 9.80665  # [m s-2]
r_d = 287.05  # Specific gas constant for dry air [J kg-1 K-1]


def extrapolate_temperature_sfc2p(
    t_sfc: xr.DataArray,
    h_sfc: xr.DataArray,
    p_sfc: xr.DataArray,
    p_target: float,
) -> xr.DataArray:
    """Extrapolate temperature to a target pressure level.

    Implements the algorithm described in [1]_. The algorithm extrapolates
    temperature from the surface to a target pressure level using a
    polynomial expression of a dimensionless variable y, which is a function of
    the surface temperature, surface pressure, and height. It assumes
    a constant lapse rate of 0.0065 K m^-1 and dry air gas constant.

    .. caution :
        This extrapolation should be used with caution. Its intended use is to
        extrapolate temperature to pressure levels below the surface, where
        values are undefined. This is useful for applications where no missing values
        are allowed, such as when training data-driven models. Results of the
        extrapolation are not physically meaningful.

    Parameters
    ----------
    t_sfc : xr.DataArray
        Surface temperature [K].
    h_sfc : xr.DataArray
        Surface height [m].
    p_sfc : xr.DataArray
        Surface pressure [Pa].
    p_target : float
        Target pressure level [Pa].

    Returns
    -------
    xr.DataArray
        Extrapolated temperature at the target pressure level.

    References
    ----------
    .. [1] https://www.umr-cnrm.fr/gmapdoc/IMG/pdf/ykfpos46t1r1.pdf

    """
    y = _vertical_extrapolation_y_term(t_sfc, p_sfc, h_sfc, p_target)
    res = t_sfc * (1 + y + (y**2) / 2 + (y**3) / 6)
    # res.attrs = metadata.override(
    #    t_sfc.metadata, shortName="T", typeOfLevel="isobaricInPa"
    # )
    res = _assign_vcoord(res, p_target)
    return res


def extrapolate_geopotential_sfc2p(
    h_sfc: xr.DataArray,
    t_sfc: xr.DataArray,
    p_sfc: xr.DataArray,
    p_target: float,
) -> xr.DataArray:
    """Extrapolate geopotential to a target pressure level.

    Implements the algorithm described in [1]_. The algorithm extrapolates
    geopotential from the surface to a target pressure level using a
    polynomial expression of a dimensionless variable y, which is a function of
    the surface temperature, surface pressure, and height. It assumes
    a constant lapse rate of 0.0065 K m^-1 and dry air gas constant.

    .. caution :
        This extrapolation should be used with caution. Its intended use is to
        extrapolate geopotential to pressure levels below the surface, where
        values are undefined. This is useful for applications where no missing values
        are allowed, such as when training data-driven models. Results of the
        extrapolation are not physically meaningful.

    Parameters
    ----------
    h_sfc : xr.DataArray
        Surface height [m].
    t_sfc : xr.DataArray
        Surface temperature [K].
    p_sfc : xr.DataArray
        Surface pressure [Pa].
    p_target : float
        Target pressure level [Pa].

    Returns
    -------
    xr.DataArray
        Extrapolated geopotential at the target pressure level.

    References
    ----------
    .. [1] https://www.umr-cnrm.fr/gmapdoc/IMG/pdf/ykfpos46t1r1.pdf

    """
    y = _vertical_extrapolation_y_term(
        t_sfc, p_sfc, h_sfc, p_target, lapse_rate=LAPSE_RATE
    )
    res = h_sfc * g - r_d * t_sfc * np.log(p_target / p_sfc) * (1 + y / 2 + (y**2) / 6)
    #    res.attrs = metadata.override(
    #        t_sfc.metadata, shortName="FI", typeOfLevel="isobaricInPa"
    #    )
    res = _assign_vcoord(res, p_target)
    return res


def extrapolate_k2p(
    field: xr.DataArray,
    p_target: float,
    mode: Literal["constant"] = "constant",
) -> xr.DataArray:
    """Extrapolate a field to a target pressure level.

    Currently, only the 'constant' extrapolation mode is implemented, where
    the extrapolation is done by simply extending the values of the
    lowermost model level to the target pressure level.

    .. caution :
        This extrapolation should be used with caution. Its intended use is to
        extrapolate temperature to pressure levels below the surface, where
        values are undefined. This is useful for applications where no missing values
        are allowed, such as when training data-driven models. Results of the
        extrapolation are not physically meaningful.

    Parameters
    ----------
    field : xr.DataArray
        Field to extrapolate.
    p_target : float
        Target pressure level [Pa].
    mode : str, optional
        Extrapolation mode. Currently only 'constant' is implemented.

    Returns
    -------
    xr.DataArray
        Extrapolated field at the target pressure level.

    References
    ----------
    .. [1] https://www.umr-cnrm.fr/gmapdoc/IMG/pdf/ykfpos46t1r1.pdf

    """
    return _assign_vcoord(field[{"level": [-1]}], p_target)  # .assign_attrs(


#        metadata.override(field.metadata, typeOfLevel="isobaricInPa")
#    )


def _vertical_extrapolation_t_zero_prime(t_sfc, h_sfc):
    t = t_sfc + LAPSE_RATE * h_sfc
    t_min = np.minimum(t, T1)
    return xr.where(h_sfc > H2, t_min, t_min * 0.5 + t * 0.5)


def _vertical_extrapolation_lapse_rate(h_sfc, t_sfc):
    t_zero_prime = _vertical_extrapolation_t_zero_prime(t_sfc, h_sfc)
    return xr.where(
        h_sfc < H1,
        LAPSE_RATE,
        1 / h_sfc * np.maximum(t_zero_prime - t_sfc, 0.0),
    )


def _vertical_extrapolation_y_term(
    t_sfc, p_sfc, h_sfc, p_target, lapse_rate=None
) -> xr.DataArray:
    if lapse_rate is None:
        lapse_rate = _vertical_extrapolation_lapse_rate(h_sfc, t_sfc)
    return lapse_rate * r_d / g * np.log(p_target / p_sfc)


def _assign_vcoord(x: xr.DataArray, p_target: float) -> xr.DataArray:
    attrs = {
        "units": "Pa",
        "positive": "down",
        "standard_name": "air_pressure",
        "long_name": "pressure",
    }
    try:
        x = x.assign_coords(level=[p_target])
    except xr.core.coordinates.CoordinateValidationError:
        x = x.expand_dims(level=[p_target])
    x["level"].attrs = attrs
    return x
