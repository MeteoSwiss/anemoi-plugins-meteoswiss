import logging
from datetime import datetime
from typing import Literal

import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

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

# earthkit-data level type of ICON full model levels (GRIB typeOfLevel generalVerticalLayer).
# W is on half levels (generalVertical) and is destaggered to full levels before interpolation.
MODEL_LEVEL_TYPE = "general"
HALF_LEVEL_PARAMS = ("W",)

FI_UNITS = "m**2 s**-2"

# Precision of the computations, as with ``to_xarray(dtype="float32")`` before earthkit-data 1.0.
DTYPE = np.float32

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
            return _override_time_metadata_on_constant_auxiliary(self._fdb_cache[shortname], time_group[0])

        date = time_group[0].get("time.valid_datetime")
        return self._request_fdb(shortname, spec, self._construct_time_request(date))

    def _request_fdb(self, shortname: str, spec: dict, time_keys: dict) -> ekd.FieldList:
        extra_keys = {"param": shortname, "levtype": spec["levtype"]}
        if "levelist" in spec:
            extra_keys["levelist"] = spec["levelist"]
        request = BASE_REQUEST | extra_keys | time_keys
        return ekd.from_source("fdb", request).to_fieldlist(read_all=True)

    def _construct_time_request(self, date: datetime) -> dict:
        base = date.replace(hour=0, minute=0, second=0, microsecond=0)
        step = int((date - base).total_seconds() // 3600)
        return {"date": base.strftime("%Y%m%d"), "time": "0000", "step": str(step)}

    def _get_field(self, fieldlist: ekd.FieldList, shortname: str) -> ekd.FieldList:
        selected = fieldlist.sel({"parameter.variable": shortname})
        if len(selected) > 0:
            return selected
        return self._fetch_from_fdb(shortname, fieldlist)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = []
        passthrough = []
        for time_group in data.group_by("time.valid_datetime"):
            t2m = _to_dataarray(self._get_field(time_group, "T_2M"), vertical=False)
            ps = _to_dataarray(self._get_field(time_group, "PS"), vertical=False)
            p = _to_dataarray(self._get_field(time_group, "P"))
            hsurf = _to_dataarray(self._get_field(time_group, "HSURF"), vertical=False)

            p[{"level": 0}] = p[{"level": 0}].where(p[{"level": 0}] < 5000, 5000 - 1e-5)

            for param_group in time_group.group_by("parameter.variable"):
                template_field = param_group[0]
                param = template_field.get("parameter.variable")

                if param in ["P", "HHL"]:
                    continue

                if not _is_model_level_field(template_field):
                    passthrough.extend(param_group)  # 2D surface field: passthrough, not interpolated
                    continue

                da = _to_dataarray(param_group)

                if param in HALF_LEVEL_PARAMS:
                    da = destagger_z(da)

                interp = interpolate_extrapolate(
                    da,
                    p,
                    t2m,
                    ps,
                    hsurf,
                    param,
                    self.interpolate_levels,
                    self.extrapolate_levels,
                )

                out.extend(_to_pressure_fields(interp, template_field))

            if self.add_geopotential:
                hhl = self._get_field(time_group, "HHL")
                fi = _geopotential_from_hhl(hhl)
                interp = interpolate_extrapolate(
                    fi,
                    p,
                    t2m,
                    ps,
                    hsurf,
                    "FI",
                    self.interpolate_levels,
                    self.extrapolate_levels,
                )
                out.extend(_to_pressure_fields(interp, hhl[0], param="FI", units=FI_UNITS))

        return new_fieldlist_from_list(out + passthrough)


def _is_model_level_field(field: ekd.Field) -> bool:
    """Whether ``field`` is on ICON model levels and is interpolated (full levels, or a half-level parameter)."""
    return field.get("vertical.level_type") == MODEL_LEVEL_TYPE or field.get("parameter.variable") in HALF_LEVEL_PARAMS


def _to_dataarray(fieldlist: ekd.FieldList, vertical: bool = True) -> xr.DataArray:
    """Values of ``fieldlist`` as a DataArray with dims ``(level, values)``, sorted by model level.

    With ``vertical=False``, ``fieldlist`` must hold a single field, returned with dims ``(values,)``.
    """
    fields = list(fieldlist)
    if not vertical:
        if len(fields) != 1:
            raise ValueError(f"Expected a single field, got {len(fields)}")
        return xr.DataArray(fields[0].to_numpy(flatten=True, dtype=DTYPE), dims=("values",))

    fields.sort(key=lambda f: f.get("vertical.level"))
    levels = [f.get("vertical.level") for f in fields]
    if len(set(levels)) != len(levels):
        raise ValueError(f"Duplicate levels for {fields[0].get('parameter.variable')}: {levels}")
    values = np.stack([f.to_numpy(flatten=True, dtype=DTYPE) for f in fields])
    return xr.DataArray(values, dims=("level", "values"), coords={"level": levels})


def _geopotential_from_hhl(hhl: ekd.FieldList) -> xr.DataArray:
    """Geopotential (FI) from destaggered model-level heights (HHL)."""
    return destagger_z(_to_dataarray(hhl)) * 9.80665


def _override_time_metadata_on_constant_auxiliary(fieldlist: ekd.FieldList, template: ekd.Field) -> ekd.FieldList:
    """Copy of constant auxiliary fields with the time metadata of ``template``; the cached fields are untouched."""
    return new_fieldlist_from_list([field.set(time=template.time) for field in fieldlist])


def _to_pressure_fields(da: xr.DataArray, template: ekd.Field, **metadata) -> list[ekd.Field]:
    """One field per pressure level of ``da`` (levels in Pa), from ``template``, with level in hPa."""
    fields = []
    for level in da["level"].values:
        level_hpa = int(int(level) / 100)
        values = np.ascontiguousarray(da.sel(level=level).values)
        fields.append(new_field_from_numpy(values, template=template, levtype="pl", levelist=level_hpa, **metadata))
    return fields


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
    from earthkit.meteo.vertical.xarray import interpolate_to_pressure_levels

    interp = interpolate_to_pressure_levels(da, p, interpolate_levels, "hPa", "log", "level")

    LOG.info(
        "Extrapolating %s below surface for pressure levels %s",
        param,
        extrapolate_levels,
    )
    for p_level in extrapolate_levels:
        idx = {"level": [el for el in interp.level].index(p_level * 100)}
        if param == "T":
            extrap = extrapolate_temperature_sfc2p(t2m, hsurf, ps, p_level * 100)
        elif param == "FI":
            extrap = extrapolate_geopotential_sfc2p(hsurf, t2m, ps, p_level * 100)
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
    y = _vertical_extrapolation_y_term(t_sfc, p_sfc, h_sfc, p_target, lapse_rate=LAPSE_RATE)
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


def _vertical_extrapolation_y_term(t_sfc, p_sfc, h_sfc, p_target, lapse_rate=None) -> xr.DataArray:
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
