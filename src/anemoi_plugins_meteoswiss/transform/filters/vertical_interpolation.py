import logging
from datetime import datetime
from typing import Literal

import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

SFC_VCOORD_TYPES = [
    "surface",
    "heightAboveGround",
    "meanSea",
]

BASE_REQUEST = {
    "stream": "reanl",
    "class": "rd",
    "expver": "r001",
    "model": "icon-rea-l-ch1",
    "type": "cf",
}

CONSTANT_TIME_KEYS = {"date": "20200101", "time": "0000", "step": "0"}

AUXILIARY_VARIABLES = {
    "T_2M": {"levtype": "sfc"},
    "PS": {"levtype": "sfc"},
    "P": {"levtype": "ml", "levelist": "1/to/81"},
    "HSURF": {"levtype": "sfc", "constant": True},
    "HHL": {"levtype": "ml", "levelist": "1/to/81", "constant": True},
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

        self._fdb_cache: dict[str, ekd.FieldList] = {}

    def _fetch_from_fdb(self, shortname: str, time_group: ekd.FieldList) -> ekd.FieldList:
        """Fetch ``shortname`` from FDB per its ``AUXILIARY_VARIABLES`` spec, for the
        timestep ``time_group`` belongs to.

        Constant variables (HHL/HSURF) are fetched once for a fixed
        reference date and cached, then restamped with this timestep's own
        time metadata on every call; the rest are fetched fresh each time.
        """
        spec = AUXILIARY_VARIABLES[shortname]
        if spec.get("constant"):
            if shortname not in self._fdb_cache:
                self._fdb_cache[shortname] = self._request_fdb(shortname, spec, CONSTANT_TIME_KEYS)
            time_metadata = time_group[0].metadata(namespace="time")
            return _restamp_time(self._fdb_cache[shortname], time_metadata)

        date = datetime.strptime(time_group[0].metadata("valid_datetime"), "%Y-%m-%dT%H:%M:%S")
        return self._request_fdb(shortname, spec, self._construct_time_request(date))

    def _request_fdb(self, shortname: str, spec: dict, time_keys: dict) -> ekd.FieldList:
        extra_keys = {"param": shortname, "levtype": spec["levtype"]}
        if "levelist" in spec:
            extra_keys["levelist"] = spec["levelist"]
        request = BASE_REQUEST | extra_keys | time_keys
        return ekd.from_source("fdb", request).to_fieldlist()

    def _construct_time_request(self, date: datetime) -> dict:
        base = date.replace(hour=0, minute=0, second=0, microsecond=0)
        step = int((date - base).total_seconds() // 3600)
        return {"date": base.strftime("%Y%m%d"), "time": "0000", "step": str(step)}

    def _get_field(self, fieldlist: ekd.FieldList, shortname: str) -> ekd.FieldList:
        selected = fieldlist.sel(shortName=shortname)
        if len(selected) > 0:
            return selected
        return self._fetch_from_fdb(shortname, fieldlist)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = ekd.FieldList()
        for time_group in data.group_by("valid_datetime"):
            t2m = self._get_field(time_group, "T_2M").to_xarray()["T_2M"]
            ps = self._get_field(time_group, "PS").to_xarray()["PS"]
            p = self._get_field(time_group, "P").to_xarray()["P"]
            hsurf = self._get_field(time_group, "HSURF").to_xarray()["HSURF"]

            p[{"level": 0}] = p[{"level": 0}].where(p[{"level": 0}] < 5000, 5000 - 1e-5)

            for param_group in time_group.group_by("shortName"):
                template_field = param_group[0]
                param = template_field.metadata("shortName")

                if param in ["P", "HHL"]:
                    continue

                if template_field.metadata("typeOfLevel") in SFC_VCOORD_TYPES:
                    out += param_group  # 2D surface field: passthrough, not interpolated
                    continue

                da = param_group.to_xarray()[param]

                if param == "W":
                    da = destagger_z(da)

                out += interpolate_extrapolate(
                    da, p, t2m, ps, hsurf, param, self.interpolate_levels, self.extrapolate_levels
                ).earthkit.to_fieldlist()

            if self.add_geopotential:
                hhl = self._get_field(time_group, "HHL")
                fi = _geopotential_from_hhl(hhl)
                out += interpolate_extrapolate(
                    fi, p, t2m, ps, hsurf, "FI", self.interpolate_levels, self.extrapolate_levels
                ).earthkit.to_fieldlist()
        return _override_pressure_level_units(out)


def _geopotential_from_hhl(hhl: ekd.FieldList) -> xr.DataArray:
    """Geopotential (FI) from destaggered model-level heights (HHL)."""
    fi_values = (destagger_z(hhl.to_xarray()["HHL"]) * 9.80665).values
    fi_md = [
        md.override(shortName="FI", typeOfLevel="generalVerticalLayer")
        for md in hhl[:-1].metadata()
    ]
    return ekd.FieldList.from_array(fi_values, fi_md).to_xarray()["FI"]


def _restamp_time(fields: ekd.FieldList, time_metadata: dict) -> ekd.FieldList:
    """Restamp a cached constant auxiliary FieldList (fetched once, for a fixed
    reference date) with the current timestep's own time metadata."""
    time_metadata = dict(time_metadata)
    del time_metadata["validityDate"]  # read-only
    del time_metadata["validityTime"]  # read-only
    return ekd.SimpleFieldList([field.clone(**time_metadata) for field in fields])


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


def interpolate_extrapolate(
    da: xr.DataArray,
    p: xr.DataArray,
    t2m: xr.DataArray,
    ps: xr.DataArray,
    hsurf: xr.DataArray,
    param: str,
    interpolate_levels: list[float],
    extrapolate_levels: list[float],
) -> xr.DataArray:
    LOG.info(
        "Interpolating %s to pressure levels %s",
        param,
        interpolate_levels,
    )
    from earthkit.meteo.vertical import interpolate_to_pressure_levels

    interp = interpolate_to_pressure_levels(
        da, p, interpolate_levels, "hPa", "log", "level"
    )

    LOG.info(
        "Extrapolating %s below surface for pressure levels %s",
        param,
        extrapolate_levels,
    )
    for p_level in extrapolate_levels:
        idx = {"level": [el for el in interp.level].index(p_level * 100)}
        if param == "T":
            extrap = extrapolate_temperature_sfc2p(
                t2m, hsurf, ps, p_level * 100
            )
        elif param == "FI":
            extrap = extrapolate_geopotential_sfc2p(
                hsurf, t2m, ps, p_level * 100
            )
        else:
            extrap = extrapolate_k2p(da, p_level * 100)
        interp[idx] = interp[idx].where(
            interp[idx].notnull(),
            extrap.squeeze().assign_coords(level=p_level * 100),
        )
    return interp

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
