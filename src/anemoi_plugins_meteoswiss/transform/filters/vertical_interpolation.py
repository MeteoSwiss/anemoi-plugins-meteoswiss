import logging
from datetime import datetime
from typing import Literal

import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.transform.filter import Filter
from earthkit.meteo.vertical import interpolate_to_pressure_levels

SFC_VCOORD_TYPES = [
    "surface",
    "heightAboveGround",
    "meanSea",
]

LOG = logging.getLogger(__name__)


BASE_REQUEST = {
    "stream": "reanl",
    "class": "rd",
    "expver": "r001",
    "model": "icon-rea-l-ch1",
    "type": "cf",
}

# silence logs from 'anemoi.transform'
# logging.getLogger("anemoi.transform").setLevel(logging.CRITICAL)


class ModelToPressureLevel(Filter):
    """
    A filter to perform vertical interpolation from model to pressure levels.
    Also performs extrapolation for values below orography that would be otherwise
    undefined. Auxiliary variables are needed:
        - HHL: model levels height
        - HSURF: surface height
        - T_2M: 2-meter temperature
        - PS: surface pressure
    """

    def __init__(
        self,
        interpolate_levels: list[float],
        extrapolate_levels: list[float] = [],
        add_geopotential: bool = True,
    ):
        """Initialize the filter.

        Parameters
        ----------
        interpolate_levels : list of float
            The pressure levels to interpolate to, in hPa.
        extrapolate_levels : list of float, optional
            The pressure levels to extrapolate to below the surface, in hPa.
        add_geopotential : bool, optional
            Whether to add geopotential (FI) to the output FieldList.

        """

        super().__init__()

        self.interpolate_levels = interpolate_levels
        self.extrapolate_levels = extrapolate_levels
        self.add_geopotential = add_geopotential

        # constant auxiliary variables
        constant_time_keys = {"date": "20200101", "time": "0000", "step": "0"}

        hhl_constant_keys = {
            "param": "HHL",
            "levtype": "ml",
            "levelist": "1/to/81",
        } | constant_time_keys
        hhl = self.load_auxiliary(hhl_constant_keys).to_fieldlist()
        _fi_values = (destagger_z(hhl.to_xarray()["HHL"]) * 9.80665).values
        _fi_md = [
            md.override(shortName="FI", typeOfLevel="generalVerticalLayer")
            for md in hhl[:-1].metadata()
        ]
        self.fi = ekd.FieldList.from_array(_fi_values, _fi_md).to_xarray()["FI"]

        # surface height (orography)
        hsurf_constant_keys = {"param": "HSURF", "levtype": "sfc"} | constant_time_keys
        self.hsurf = self.load_auxiliary(hsurf_constant_keys).to_xarray()["HSURF"]

    def _construct_time_request(self, date: datetime | None):
        if date is None:
            return {}
        base = date.replace(hour=0, minute=0, second=0, microsecond=0)
        step = int((date - base).total_seconds() // 3600)
        return {"date": base.strftime("%Y%m%d"), "time": "0000", "step": str(step)}

    def load_auxiliary(
        self, extra_request_keys: dict, date: datetime | None = None
    ) -> ekd.FieldList:
        # Load auxiliary data based on the request keys and date
        # TODO: if date is None, assume it's static keys
        request = BASE_REQUEST | extra_request_keys
        request |= self._construct_time_request(date)
        return ekd.from_source("fdb", request)

    def surface_temperature(self, date: datetime) -> xr.DataArray:
        extra_request_keys = {"param": "T_2M", "levtype": "sfc"}
        ds = self.load_auxiliary(extra_request_keys, date).to_xarray()
        return ds["T_2M"]

    def surface_pressure(self, date: datetime) -> xr.DataArray:
        extra_request_keys = {"param": "PS", "levtype": "sfc"}
        ds = self.load_auxiliary(extra_request_keys, date).to_xarray()
        return ds["PS"]

    def pressure(self, date: datetime) -> xr.DataArray:
        extra_request_keys = {"param": "P", "levtype": "ml", "levelist": "1/to/81"}
        da = self.load_auxiliary(extra_request_keys, date).to_xarray()["P"]
        da[{"level": 0}] = da[{"level": 0}].where(da[{"level": 0}] < 5000, 5000 - 1e-5)
        return da

    def interpolate_extrapolate(
        self,
        da: xr.DataArray,
        p: xr.DataArray,
        t2m: xr.DataArray,
        ps: xr.DataArray,
        param: str,
    ) -> xr.DataArray:
        LOG.info(
            "Interpolating %s to pressure levels %s",
            param,
            self.interpolate_levels,
        )
        interp = interpolate_to_pressure_levels(
            da, p, self.interpolate_levels, "hPa", "log", "level"
        )

        LOG.info(
            "Extrapolating %s below surface for pressure levels %s",
            param,
            self.extrapolate_levels,
        )
        for p_level in self.extrapolate_levels:
            idx = {"level": [el for el in interp.level].index(p_level * 100)}
            if param == "T":
                extrap = extrapolate_temperature_sfc2p(
                    t2m, self.hsurf, ps, p_level * 100
                )
            elif param == "FI":
                extrap = extrapolate_geopotential_sfc2p(
                    self.hsurf, t2m, ps, p_level * 100
                )
            else:
                extrap = extrapolate_k2p(da, p_level * 100)
            interp[idx] = interp[idx].where(
                interp[idx].notnull(),
                extrap.squeeze().assign_coords(level=p_level * 100),
            )
        return interp

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = ekd.FieldList()
        for time_group in data.group_by("validityTime"):
            valid_datetime = time_group[0].metadata("valid_datetime")
            valid_datetime = datetime.strptime(valid_datetime, "%Y-%m-%dT%H:%M:%S")
            t2m = self.surface_temperature(valid_datetime)
            ps = self.surface_pressure(valid_datetime)
            p = self.pressure(valid_datetime)
            for param_group in time_group.group_by("shortName"):
                template_field = param_group[0]
                param = template_field.metadata("shortName")
                da = param_group.to_xarray()[param]

                if param == "W":
                    da = destagger_z(da)

                out += self.interpolate_extrapolate(
                    da, p, t2m, ps, param
                ).earthkit.to_fieldlist()
            if self.add_geopotential:
                time_metadata = time_group[0].metadata(namespace="time")
                _fi = _override_time_metadata_on_constant_auxiliary(
                    self.fi, time_metadata
                )
                out += self.interpolate_extrapolate(
                    _fi, p, t2m, ps, "FI"
                ).earthkit.to_fieldlist()
        return _override_pressure_level_units(out)


def _override_time_metadata_on_constant_auxiliary(
    da: xr.DataArray, time_metadata: dict
) -> xr.DataArray:
    """Metadata override for constant auxiliary fields."""
    del time_metadata["validityDate"]  # read-only
    del time_metadata["validityTime"]  # read-only
    md = da.earthkit.metadata.override(**time_metadata)
    da.attrs["_earthkit"]["message"] = md._handle.get_buffer()
    return da


def _override_pressure_level_units(fields):
    out = ekd.SimpleFieldList()
    for field in fields:
        level_hpa = int(int(field.metadata("level")) / 100)
        overrides = {
            "typeOfLevel": "isobaricInhPa",
            "level": level_hpa,
            "levelist": level_hpa,
        }
        out.append(field.clone(**overrides))
    return out


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


def destagger_z(field: xr.DataArray) -> xr.DataArray:
    """Destagger a field in the vertical (z) direction."""
    dims = list(field.sizes.keys())
    out = (
        xr.apply_ufunc(
            lambda a: 0.5 * (a[..., :-1] + a[..., 1:]),
            field,
            input_core_dims=[["level"]],
            output_core_dims=[["level"]],
            exclude_dims={"level"},
            keep_attrs=True,
        )
        .transpose(*dims)
        .assign_coords(level=field.level[:-1])
    )
    return out
