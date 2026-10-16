"""Computes SP_10M (wind speed), DD_10M (wind direction), RELHUM_2M,
TW_2M (2 m wet-bulb temperature), HZEROCL (freezing level), SNOWLMT
(snowfall limit), mixed-layer and most-unstable CAPE/CIN (CAPE_ML, CIN_ML, CAPE_MU, CIN_MU),
precipitation type (PREC_TYPE) and accumulated snowfall (TOT_SNOW) in a
single pass. Fields are added to Varda inference output.

Usage::

    typed_variables:
      SP_10M:
        mars: {param: SP_10M, levtype: sfc}
      DD_10M:
        mars: {param: DD_10M, levtype: sfc}
      RELHUM_2M:
        mars: {param: RELHUM_2M, levtype: sfc}
      TW_2M:
        mars: {param: TW_2M, levtype: sfc}
      HZEROCL:
        mars: {param: HZEROCL, levtype: sfc}
      SNOWLMT:
        mars: {param: SNOWLMT, levtype: sfc}
      CAPE_ML:
        mars: {param: CAPE_ML, levtype: sfc}
      CIN_ML:
        mars: {param: CIN_ML, levtype: sfc}
      CAPE_MU:
        mars: {param: CAPE_MU, levtype: sfc}
      CIN_MU:
        mars: {param: CIN_MU, levtype: sfc}
      PREC_TYPE:
        mars: {param: PREC_TYPE, levtype: sfc}
      TOT_SNOW:
        mars: {param: TOT_SNOW, levtype: sfc}

    output:
      tee:
      - grib:
          post_processors:
          - forward_transform_filter:
              surface-diagnostics:
                u_component: U_10M
                v_component: V_10M
                temperature: T_2M
                dewpoint: TD_2M
                surface_pressure: PS
                temperature_levels: T
                specific_humidity_levels: QV
                geopotential_levels: FI
                precipitation: TOT_PREC
                orography_file: /path/to/external_parameter_icon_grid.nc
                variables: [SP_10M, DD_10M, RELHUM_2M, TW_2M, HZEROCL, SNOWLMT,
                            CAPE_ML, CIN_ML, CAPE_MU, CIN_MU, PREC_TYPE, TOT_SNOW]

Only the inputs needed by the requested ``variables`` must be given (and
present in the state), e.g. ``SP_10M`` alone needs only the wind components.

Notes
-----
- Each requested output needs a ``typed_variables`` + GRIB template entry
  of its own, since the checkpoint never produced it.
- Wind direction: meteorological convention (FROM, clockwise from North).
- Temperature/dewpoint in Kelvin; RH returned in percent.
- TW_2M: earthkit wet-bulb temperature (Normand's rule) from T_2M, TD_2M, PS.
- HZEROCL (INCA Z0), SNOWLMT (ZS) and the top of the melting layer
  (MELTLAYER_TOP, INCA ZM; internal only, see PREC_TYPE) follow INCA's
  ``heightlevel`` (``inca_diagnostics.py``): the *uppermost* crossing of
  the profile through the threshold (0 degC for HZEROCL, wet-bulb +1.3 degC
  for SNOWLMT, wet-bulb +0.2 degC for MELTLAYER_TOP), linear in geometric
  height between the two bracketing levels. Profiles are the pressure
  levels above ground (p < PS) plus a surface point at orography + 2 m
  (T_2M resp. TW_2M).
- Without a crossing, the level is extrapolated from the surface point with
  a lapse rate (0.65 K/100 m for HZEROCL, 0.5 K/100 m for SNOWLMT and
  MELTLAYER_TOP; MELTLAYER_TOP is orography + 150 m if only it has no
  crossing), so it can lie below ground, down to -100 m. MELTLAYER_TOP is
  kept at least 75 m above SNOWLMT.
- Unlike INCA, which runs on model levels, the profile is built from
  pressure levels, so the crossings are interpolated over much coarser
  layers.
- INCA computes wet-bulb with a different approximation than earthkit; the
  difference is a few tenths of a kelvin, i.e. some tens of metres in
  SNOWLMT/MELTLAYER_TOP.
- CAPE/CIN (J/kg), as ICON: CAPE_ML/CIN_ML lift a parcel with the mean
  potential temperature and specific humidity of the lowest 50 hPa
  (pressure-weighted), starting at the surface; CAPE_MU/CIN_MU lift the
  column point within 300 hPa of the surface with the highest equivalent
  potential temperature, starting there. The parcel rises along the dry
  adiabat to its LCL (earthkit ``lcl``), then along the pseudo-adiabat of
  its equivalent potential temperature (earthkit, Bolton 1980 eq. 39).
  Buoyancy uses virtual temperature for parcel and environment, on the same
  column as the height levels plus the LCL as an extra point, and is
  integrated in geometric height from the parcel's starting point. The LFC
  is the lowest point at or above the LCL where the parcel becomes buoyant,
  the EL the highest point above it where it stops being buoyant (or the
  column top). CAPE is the area from the LFC to the EL, CIN the negative
  area below the LFC (<= 0). Without an LFC, CAPE is 0 and CIN missing
  (NaN). With only a handful of pressure levels near the ground, the mixed
  layer usually contains the 2 m point and at most one level, and the most
  unstable parcel is picked from a few candidates: both are rough estimates.
- PREC_TYPE and TOT_SNOW follow INCA's ``precipitation_type`` and
  ``precip_snow_water_equivalent``, using SNOWLMT and MELTLAYER_TOP from
  this filter (MELTLAYER_TOP is not an output: it has no GRIB code). The
  melting-layer depth is MELTLAYER_TOP - SNOWLMT, clamped to [150, 375] m. Where it rains or snows (> 0.1 mm/h), PREC_TYPE is
  3 (snow) if the ground is at least that depth above SNOWLMT, 1 (rain) if
  at least that depth below, 2 (mixed) in between, and 4 (freezing rain)
  for rain with T_2M below 0.1 degC; 0 without precipitation. INCA's ground
  temperature is replaced by T_2M (Varda has none) and its hail category
  (5) is left out (no hail field). TOT_SNOW accumulates, from the start of
  the forecast, the precipitation of each step times the snow fraction: 0
  at or below SNOWLMT, 1 at or above SNOWLMT + the melting-layer depth,
  ``snow_fraction: linear`` (default) or ``logistic`` in between.
- The precipitation rate is the difference of ``precipitation``
  (accumulated from the start of the forecast) with the previous step, so
  the filter keeps state between steps and needs them in order. At the
  first step there is none: PREC_TYPE is missing and TOT_SNOW 0. Negative
  differences (where ``copy-prognostic-from-forecaster`` swaps in the
  forecaster's precipitation) are set to 0.
- The orography file must be on the same grid and cell order as the state
  (native ICON grid), so this filter must run before any regridding.
- With the height levels or CAPE/CIN the filter groups the pressure-level
  inputs vertically and re-yields them itself, so they stay in the output.
- Must be the last transform filter in its post_processors list: chaining
  separate filters fails because ``TransformFilter.process()`` re-wraps
  the whole state each time, and a field invented by an earlier filter
  isn't in the checkpoint's variable catalog, so the next wrap raises
  KeyError. One filter computing everything avoids that.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Literal

import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec
from earthkit.meteo import constants
from earthkit.meteo.thermo import dewpoint_from_specific_humidity
from earthkit.meteo.thermo import ept_from_dewpoint
from earthkit.meteo.thermo import ept_from_specific_humidity
from earthkit.meteo.thermo import lcl
from earthkit.meteo.thermo import potential_temperature
from earthkit.meteo.thermo import relative_humidity_from_dewpoint
from earthkit.meteo.thermo import saturation_specific_humidity
from earthkit.meteo.thermo import specific_humidity_from_dewpoint
from earthkit.meteo.thermo import temperature_from_potential_temperature
from earthkit.meteo.thermo import temperature_on_dry_adiabat
from earthkit.meteo.thermo import temperature_on_moist_adiabat
from earthkit.meteo.thermo import virtual_temperature
from earthkit.meteo.thermo import wet_bulb_temperature_from_dewpoint
from earthkit.meteo.thermo import wet_bulb_temperature_from_specific_humidity
from earthkit.meteo.vertical import geometric_height_from_geopotential

from .copy_prognostic_from_forecaster import _leadtime

LOG = logging.getLogger(__name__)

# INCA thresholds and fallbacks (heightlevel in inca_diagnostics.py).
HZEROCL_THRESHOLD = 273.15  # Freezing level in K
SNOWLMT_THRESHOLD = 273.15 + 1.3  # height where wet-bulb crosses 1.3 deg
MELTLAYER_TOP_THRESHOLD = 273.15 + 0.2  # where falling snow starts to melt
T_LAPSE_RATE = 0.65 / 100.0  # K/m, extrapolates HZEROCL
TW_LAPSE_RATE = 0.5 / 100.0  # K/m, extrapolates SNOWLMT/MELTLAYER_TOP
MIN_HEIGHT = -100.0  # m
MELTLAYER_TOP_OFFSET = (
    150.0  # When MELTLAYER_TOP has no crossing but SNOWLMT does, MELTLAYER_TOP is set to orography + 150 m.
)
MIN_MELTLAYER_DEPTH = 75.0  # m, minimum MELTLAYER_TOP - SNOWLMT

# Pseudo-adiabat for CAPE/CIN: Bolton (1980) eq. 39. earthkit's default "ifs"
# formula gives a parcel up to ~2 K colder aloft, i.e. ~15% less CAPE.
EPT_METHOD = "bolton39"
ML_DEPTH = 5000.0  # Pa, depth of the mixed layer averaged for CAPE_ML/CIN_ML
MU_DEPTH = 30000.0  # Pa, depth searched for the most unstable parcel (CAPE_MU/CIN_MU)

# INCA precipitation type and snow fraction (precipitation_type and
# precip_snow_water_equivalent in inca_diagnostics.py).
PREC_RATE_MIN = 0.1  # mm/h, minimum rate to assign a precipitation type
MELTLAYER_HALF_DEPTH_MIN = 150.0  # m, MELTLAYER_TOP - SNOWLMT is clamped to [min, 2.5 * min]
MELTLAYER_HALF_DEPTH_MAX = 2.5 * MELTLAYER_HALF_DEPTH_MIN
FREEZING_GROUND_T = 273.15 + 0.1  # K, rain on colder ground is freezing rain
FREEZING_MAX_AIR_T = 273.15 + 0.5  # K, ...unless the air is warmer than this
PREC_TYPE_NONE, PREC_TYPE_RAIN, PREC_TYPE_MIXED, PREC_TYPE_SNOW, PREC_TYPE_FREEZING_RAIN = 0, 1, 2, 3, 4

_WIND = ("u_component", "v_component")
_HUMIDITY = ("temperature", "dewpoint")
_MOIST_PROFILE = (
    *_HUMIDITY,
    "surface_pressure",
    "temperature_levels",
    "specific_humidity_levels",
    "geopotential_levels",
)
_SUPPORTED_VARIABLES: dict[str, tuple[str, ...]] = {
    "SP_10M": _WIND,
    "DD_10M": _WIND,
    "RELHUM_2M": _HUMIDITY,
    "TW_2M": (*_HUMIDITY, "surface_pressure"),
    "HZEROCL": ("temperature", "surface_pressure", "temperature_levels", "geopotential_levels"),
    "SNOWLMT": _MOIST_PROFILE,
    "CAPE_ML": _MOIST_PROFILE,
    "CIN_ML": _MOIST_PROFILE,
    "CAPE_MU": _MOIST_PROFILE,
    "CIN_MU": _MOIST_PROFILE,
    "PREC_TYPE": (*_MOIST_PROFILE, "precipitation"),
    "TOT_SNOW": (*_MOIST_PROFILE, "precipitation"),
}
_VERTICAL_VARIABLES = {
    "HZEROCL",
    "SNOWLMT",
    "CAPE_ML",
    "CIN_ML",
    "CAPE_MU",
    "CIN_MU",
    "PREC_TYPE",
    "TOT_SNOW",
}
# Need SNOWLMT and the internal MELTLAYER_TOP, whether or not SNOWLMT is output.
_MELTING_LAYER_VARIABLES = {"PREC_TYPE", "TOT_SNOW"}
_ALL_INPUTS = (
    "u_component",
    "v_component",
    "temperature",
    "dewpoint",
    "surface_pressure",
    "temperature_levels",
    "specific_humidity_levels",
    "geopotential_levels",
    "precipitation",
)


class SurfaceDiagnostics(MatchingFieldsFilter):
    """Compute surface diagnostics, INCA height levels and ML/MU CAPE/CIN."""

    MATCHING = MatchingSpec(
        select="param",
        forward=_ALL_INPUTS,
        return_inputs="all",
    )

    def __init__(
        self,
        *,
        variables: str | list[str],
        u_component: str | None = None,
        v_component: str | None = None,
        temperature: str | None = None,
        dewpoint: str | None = None,
        surface_pressure: str | None = None,
        temperature_levels: str | None = None,
        specific_humidity_levels: str | None = None,
        geopotential_levels: str | None = None,
        orography_file: str | None = None,
        orography_variable: str = "topography_c",
        precipitation: str | None = None,
        snow_fraction: Literal["linear", "logistic"] = "linear",
    ) -> None:
        """Field names are internal anemoi names (``*_levels`` are the
        pressure-level params, e.g. ``t``). ``variables`` selects one or more
        of ``SP_10M``, ``DD_10M``, ``RELHUM_2M``, ``TW_2M``, ``HZEROCL``,
        ``SNOWLMT``, ``CAPE_ML``, ``CIN_ML``, ``CAPE_MU``, ``CIN_MU``,
        ``PREC_TYPE``, ``TOT_SNOW`` to compute.
        ``orography_file`` (NetCDF, variable ``orography_variable`` in m) is
        required for all but the first four. ``PREC_TYPE`` and ``TOT_SNOW``
        also need ``precipitation``, accumulated from the start of the
        forecast (kg m-2); ``snow_fraction`` selects the snow fraction profile
        across the melting layer for ``TOT_SNOW``."""
        if isinstance(variables, str):
            variables = [variables]
        variables = list(variables)
        if not variables:
            raise ValueError(f"`variables` must list at least one of: {sorted(_SUPPORTED_VARIABLES)}.")
        unknown = set(variables) - set(_SUPPORTED_VARIABLES)
        if unknown:
            raise ValueError(f"Unsupported variable(s) {sorted(unknown)}; supported: {sorted(_SUPPORTED_VARIABLES)}.")

        self.u_component = u_component
        self.v_component = v_component
        self.temperature = temperature
        self.dewpoint = dewpoint
        self.surface_pressure = surface_pressure
        self.temperature_levels = temperature_levels
        self.specific_humidity_levels = specific_humidity_levels
        self.geopotential_levels = geopotential_levels
        self.precipitation = precipitation
        if snow_fraction not in ("linear", "logistic"):
            raise ValueError(f"`snow_fraction` must be 'linear' or 'logistic', got {snow_fraction!r}.")
        self.snow_fraction = snow_fraction
        self.variables = variables
        # Precipitation at the previous step and the snow accumulated so far,
        # carried from one call (output step) to the next.
        self._previous_precipitation: tuple[timedelta, np.ndarray] | None = None
        self._snow_total: np.ndarray | None = None

        required = {name for v in variables for name in _SUPPORTED_VARIABLES[v]}
        missing = sorted(name for name in required if getattr(self, name) is None)
        if missing:
            raise ValueError(f"Variables {variables} require the input(s) {missing}.")

        vertical = bool(_VERTICAL_VARIABLES & set(variables))
        self._orography = None
        if vertical:
            if orography_file is None:
                raise ValueError(f"{sorted(_VERTICAL_VARIABLES & set(variables))} require `orography_file`.")
            self._orography = _load_orography(Path(orography_file), orography_variable)

        # Match only the inputs the requested variables need. With vertical
        # grouping a pressure-level input arrives as a FieldList, which the
        # base class can't return, so forward_transform re-yields all inputs.
        self.MATCHING = replace(
            self.MATCHING,
            forward=tuple(name for name in _ALL_INPUTS if name in required),
            return_inputs="none" if vertical else "all",
            vertical=vertical,
        )

        LOG.info(
            "surface-diagnostics: will add %s to the output, computed from %s",
            self.variables,
            {name: getattr(self, name) for name in self.MATCHING.forward},
        )
        super().__init__()

    def forward_transform(
        self,
        u_component: ekd.Field | None = None,
        v_component: ekd.Field | None = None,
        temperature: ekd.Field | None = None,
        dewpoint: ekd.Field | None = None,
        surface_pressure: ekd.Field | None = None,
        temperature_levels: ekd.FieldList | None = None,
        specific_humidity_levels: ekd.FieldList | None = None,
        geopotential_levels: ekd.FieldList | None = None,
        precipitation: ekd.Field | None = None,
    ) -> Iterator[ekd.Field]:
        if self.MATCHING.vertical:
            inputs = (
                u_component,
                v_component,
                temperature,
                dewpoint,
                surface_pressure,
                temperature_levels,
                specific_humidity_levels,
                geopotential_levels,
                precipitation,
            )
            for x in inputs:
                if isinstance(x, ekd.Field):
                    yield x
                elif x is not None:
                    yield from x
            u_component, v_component, temperature, dewpoint, surface_pressure = (_single(x) for x in inputs[:5])
            precipitation = _single(precipitation)

        timings: dict[str, float] = {}

        if "SP_10M" in self.variables or "DD_10M" in self.variables:
            u = u_component.to_numpy()
            v = v_component.to_numpy()

        if "SP_10M" in self.variables:
            with _timer(timings, "SP_10M"):
                speed = np.sqrt(u**2 + v**2)
            yield self.new_field_from_numpy(
                speed,
                template=u_component,
                name="SP_10M",
                param="SP_10M",
                shortName="SP_10M",
            )

        if "DD_10M" in self.variables:
            with _timer(timings, "DD_10M"):
                direction = np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)
            yield self.new_field_from_numpy(
                direction,
                template=u_component,
                name="DD_10M",
                param="DD_10M",
                shortName="DD_10M",
            )

        if "RELHUM_2M" in self.variables:
            with _timer(timings, "RELHUM_2M"):
                rh = relative_humidity_from_dewpoint(temperature.to_numpy(), dewpoint.to_numpy())
            yield self.new_field_from_numpy(
                rh,
                template=temperature,
                name="RELHUM_2M",
                param="RELHUM_2M",
                shortName="RELHUM_2M",
            )

        melting_layer = bool(_MELTING_LAYER_VARIABLES & set(self.variables))
        wet_bulb_profile = melting_layer or "SNOWLMT" in self.variables
        if "TW_2M" in self.variables or wet_bulb_profile:
            with _timer(timings, "TW_2M"):
                tw_2m = wet_bulb_temperature_from_dewpoint(
                    temperature.to_numpy(flatten=True),
                    dewpoint.to_numpy(flatten=True),
                    surface_pressure.to_numpy(flatten=True),
                )

        if "TW_2M" in self.variables:
            yield self.new_field_from_numpy(
                tw_2m.reshape(temperature.shape),
                template=temperature,
                name="TW_2M",
                param="TW_2M",
                shortName="TW_2M",
            )

        if not self.MATCHING.vertical:
            _log_timings(timings, temperature or u_component)
            return

        t_2m = temperature.to_numpy(flatten=True)
        if self._orography.size != t_2m.size:
            raise ValueError(
                f"Orography has {self._orography.size} points but the state has {t_2m.size}; "
                "surface-diagnostics must run on the native grid of the orography file."
            )
        sp = surface_pressure.to_numpy(flatten=True)

        parcels = [kind for kind in ("ML", "MU") if {f"CAPE_{kind}", f"CIN_{kind}"} & set(self.variables)]
        level_inputs = {"t": temperature_levels, "z": geopotential_levels}
        if wet_bulb_profile or parcels:
            level_inputs["q"] = specific_humidity_levels
        with _timer(timings, "pressure-level profiles"):
            pressure, profiles = _stack_levels(level_inputs)
            level_heights = geometric_height_from_geopotential(profiles["z"])
            above_ground = pressure[:, None] < sp[None, :]

        height_variables = [*self.variables, "SNOWLMT", "MELTLAYER_TOP"] if melting_layer else self.variables
        height_label = "/".join(v for v in ("HZEROCL", "SNOWLMT", "MELTLAYER_TOP") if v in height_variables)
        with _timer(timings, height_label) if height_label else nullcontext():
            outputs = _height_levels(
                height_variables,
                t_2m=t_2m,
                tw_2m=tw_2m if wet_bulb_profile else None,
                orography=self._orography,
                t_levels=profiles["t"],
                tw_levels=_wet_bulb_levels(profiles, pressure) if wet_bulb_profile else None,
                level_heights=level_heights,
                above_ground=above_ground,
            )
        if parcels:
            with _timer(timings, "CAPE/CIN column"):
                column = _environment_column(
                    t_2m=t_2m,
                    td_2m=dewpoint.to_numpy(flatten=True),
                    sp=sp,
                    orography=self._orography,
                    pressure=pressure,
                    t_levels=profiles["t"],
                    q_levels=profiles["q"],
                    level_heights=level_heights,
                    above_ground=above_ground,
                )
        if melting_layer:
            with _timer(timings, "/".join(v for v in ("PREC_TYPE", "TOT_SNOW") if v in self.variables)):
                outputs.update(
                    self._precipitation_type_and_snow(
                        precipitation,
                        t_2m=t_2m,
                        snowlmt=outputs["SNOWLMT"],
                        meltlayer_top=outputs["MELTLAYER_TOP"],
                    )
                )
            outputs = {name: values for name, values in outputs.items() if name in self.variables}
        for kind in parcels:
            with _timer(timings, f"CAPE_{kind}/CIN_{kind}"):
                start = _mixed_layer_parcel(*column) if kind == "ML" else _most_unstable_parcel(*column)
                cape, cin = _parcel_cape_cin(*column, *start)
            for name, values in ((f"CAPE_{kind}", cape), (f"CIN_{kind}", cin)):
                if name in self.variables:
                    outputs[name] = values
        _log_timings(timings, temperature)
        for name, values in outputs.items():
            yield self.new_field_from_numpy(
                values.reshape(temperature.shape),
                template=temperature,
                name=name,
                param=name,
                shortName=name,
            )

    def _precipitation_type_and_snow(
        self, precipitation: ekd.Field, *, t_2m: np.ndarray, snowlmt: np.ndarray, meltlayer_top: np.ndarray
    ) -> dict[str, np.ndarray]:
        """PREC_TYPE and TOT_SNOW from the precipitation since the previous step.

        The first step seen (or a lead time not after the previous one, i.e.
        a new forecast) has no previous step: PREC_TYPE is missing there and
        TOT_SNOW starts from 0.
        """
        lead = _leadtime(precipitation)
        if lead is None:
            raise ValueError("surface-diagnostics: can't determine the lead time of the precipitation field.")
        total = precipitation.to_numpy(flatten=True)
        previous = self._previous_precipitation
        if previous is None or lead <= previous[0]:
            amount = np.full_like(total, np.nan)
            self._snow_total = np.zeros_like(total)
        else:
            # Accumulations should only grow; clip the jumps where the
            # forecaster's precipitation is swapped in at the overlap steps.
            amount = np.maximum(total - previous[1], 0.0)
        self._previous_precipitation = (lead, total)
        hours = 1.0 if previous is None or lead <= previous[0] else (lead - previous[0]) / timedelta(hours=1)

        out = {}
        if "PREC_TYPE" in self.variables:
            out["PREC_TYPE"] = _precipitation_type(
                amount / hours, snowlmt, meltlayer_top, t_2m, ground_temperature=t_2m, orography=self._orography
            )
        if "TOT_SNOW" in self.variables:
            fraction = _snow_fraction(snowlmt, meltlayer_top, self._orography, self.snow_fraction)
            self._snow_total = self._snow_total + np.nan_to_num(fraction * amount)
            out["TOT_SNOW"] = self._snow_total.copy()
        return out


@contextmanager
def _timer(timings: dict[str, float], label: str) -> Iterator[None]:
    """Add the wall time spent in the block to ``timings[label]`` (seconds)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[label] = timings.get(label, 0.0) + time.perf_counter() - start


def _log_timings(timings: dict[str, float], field: ekd.Field | None) -> None:
    """Log the time spent on each diagnostic for one output step."""
    if not timings:
        return
    lead = _leadtime(field) if field is not None else None
    LOG.info(
        "surface-diagnostics: computed step %s in %.2fs (%s)",
        "?" if lead is None else f"{lead / timedelta(hours=1):g}h",
        sum(timings.values()),
        ", ".join(f"{label} {seconds:.2f}s" for label, seconds in timings.items()),
    )


def _load_orography(path: Path, variable: str) -> np.ndarray:
    """Load the model orography (m), flattened in native cell order."""
    with xr.open_dataset(path) as ds:
        orography = ds[variable].values.astype(np.float64).ravel()
    LOG.info("surface-diagnostics: orography (%d points) loaded from %s", orography.size, path)
    return orography


def _single(x: ekd.Field | ekd.FieldList | None) -> ekd.Field | None:
    """With vertical grouping, a surface field carrying a level arrives as a
    one-element FieldList; unwrap it."""
    if x is None or isinstance(x, ekd.Field):
        return x
    if len(x) != 1:
        raise ValueError(f"Expected a single surface field, got {len(x)} fields.")
    return x[0]


def _stack_levels(inputs: dict[str, ekd.FieldList]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stack pressure-level inputs bottom-up (decreasing pressure).

    Returns the level pressures (Pa) and, per input, an array of shape
    (levels, points). All inputs must share the same levels.
    """
    by_level = {name: {float(f.metadata("levelist")): f for f in fields} for name, fields in inputs.items()}
    levels = set(next(iter(by_level.values())))
    for name, fields in by_level.items():
        if set(fields) != levels:
            raise ValueError(
                f"Pressure levels differ between inputs: {name} has {sorted(fields)}, expected {sorted(levels)}."
            )
    ordered = sorted(levels, reverse=True)
    profiles = {
        name: np.stack([fields[lev].to_numpy(flatten=True) for lev in ordered]) for name, fields in by_level.items()
    }
    return np.array(ordered) * 100.0, profiles


def _column(
    surface_value: np.ndarray,
    surface_height: np.ndarray,
    level_values: np.ndarray,
    level_heights: np.ndarray,
    above_ground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-up profile: the surface point followed by the pressure levels.

    Levels below ground are replaced by the surface point; the resulting
    zero-length segments can't contain a crossing, so the surface point
    connects directly to the lowest level above ground.
    """
    values = np.where(above_ground, level_values, surface_value)
    heights = np.where(above_ground, level_heights, surface_height)
    return np.vstack([surface_value, values]), np.vstack([surface_height, heights])


def _uppermost_crossing(values: np.ndarray, heights: np.ndarray, threshold: float) -> np.ndarray:
    """Height of the uppermost level pair with ``values`` >= ``threshold``
    below and < ``threshold`` above, linearly interpolated in height.

    ``values`` and ``heights`` have shape (levels, points), ordered
    bottom-up. Returns NaN where there is no crossing.
    """
    lower, upper = values[:-1], values[1:]
    crossing = (lower >= threshold) & (upper < threshold)
    found = crossing.any(axis=0)
    k = (crossing.shape[0] - 1 - np.argmax(crossing[::-1], axis=0))[None, :]

    v0 = np.take_along_axis(lower, k, axis=0)[0]
    v1 = np.take_along_axis(upper, k, axis=0)[0]
    h0 = np.take_along_axis(heights[:-1], k, axis=0)[0]
    h1 = np.take_along_axis(heights[1:], k, axis=0)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        height = h0 + (threshold - v0) / (v1 - v0) * (h1 - h0)
    return np.where(found, height, np.nan)


def _wet_bulb_levels(profiles: dict[str, np.ndarray], pressure: np.ndarray) -> np.ndarray:
    """Wet-bulb temperature (K) on every pressure level, shape (levels, points)."""
    return np.stack(
        [
            wet_bulb_temperature_from_specific_humidity(t, q, np.full_like(t, p))
            for t, q, p in zip(profiles["t"], profiles["q"], pressure)
        ]
    )


def _height_levels(
    variables: list[str],
    *,
    t_2m: np.ndarray,
    tw_2m: np.ndarray | None,
    orography: np.ndarray,
    t_levels: np.ndarray,
    tw_levels: np.ndarray | None,
    level_heights: np.ndarray,
    above_ground: np.ndarray,
) -> dict[str, np.ndarray]:
    """HZEROCL (Z0), SNOWLMT (ZS) and MELTLAYER_TOP (ZM) as INCA's ``heightlevel``.

    The crossing is the uppermost one in the column, with no search
    ceiling. Columns without a crossing are extrapolated from the surface
    point with a fixed lapse rate (which may place the level below ground)
    and clamped at ``MIN_HEIGHT``; MELTLAYER_TOP is kept at least
    ``MIN_MELTLAYER_DEPTH`` above SNOWLMT.
    """
    surface_height = orography + 2.0
    out = {}

    if "HZEROCL" in variables:
        values, heights = _column(t_2m, surface_height, t_levels, level_heights, above_ground)
        z0 = _uppermost_crossing(values, heights, HZEROCL_THRESHOLD)
        extrapolated = orography - np.abs(t_2m - HZEROCL_THRESHOLD) / T_LAPSE_RATE
        out["HZEROCL"] = np.maximum(np.where(np.isnan(z0), extrapolated, z0), MIN_HEIGHT)

    if tw_levels is None:
        return out

    values, heights = _column(tw_2m, surface_height, tw_levels, level_heights, above_ground)
    zs = _uppermost_crossing(values, heights, SNOWLMT_THRESHOLD)
    zm = _uppermost_crossing(values, heights, MELTLAYER_TOP_THRESHOLD)
    no_zs, no_zm = np.isnan(zs), np.isnan(zm)

    zs = np.where(no_zs, orography - (SNOWLMT_THRESHOLD - tw_2m) / TW_LAPSE_RATE, zs)
    zs = np.maximum(zs, MIN_HEIGHT)
    if "SNOWLMT" in variables:
        out["SNOWLMT"] = zs

    if "MELTLAYER_TOP" in variables:
        zm = np.where(no_zs & no_zm, orography - (MELTLAYER_TOP_THRESHOLD - tw_2m) / TW_LAPSE_RATE, zm)
        zm = np.where(~no_zs & no_zm, orography + MELTLAYER_TOP_OFFSET, zm)
        zm = np.maximum(zm, MIN_HEIGHT)
        out["MELTLAYER_TOP"] = np.where(zm - zs < MIN_MELTLAYER_DEPTH, zs + MIN_MELTLAYER_DEPTH, zm)

    return out


def _environment_column(
    *,
    t_2m: np.ndarray,
    td_2m: np.ndarray,
    sp: np.ndarray,
    orography: np.ndarray,
    pressure: np.ndarray,
    t_levels: np.ndarray,
    q_levels: np.ndarray,
    level_heights: np.ndarray,
    above_ground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pressure, height, temperature and specific humidity of the bottom-up
    column used for CAPE/CIN: the 2 m point followed by the pressure levels
    (below-ground levels collapsed onto the 2 m point, as in ``_column``)."""
    surface_height = orography + 2.0
    level_pressure = np.broadcast_to(pressure[:, None], t_levels.shape)
    p, height = _column(sp, surface_height, level_pressure, level_heights, above_ground)
    t, _ = _column(t_2m, surface_height, t_levels, level_heights, above_ground)
    q, _ = _column(specific_humidity_from_dewpoint(td_2m, sp), surface_height, q_levels, level_heights, above_ground)
    return p, height, t, q


def _mixed_layer_parcel(
    p: np.ndarray, height: np.ndarray, t: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pressure, temperature and specific humidity of the mixed-layer parcel:
    the pressure-weighted mean potential temperature and specific humidity of
    the lowest ``ML_DEPTH``, at the surface pressure."""
    sp = p[0]
    p_top = sp - ML_DEPTH
    p_layer, theta, q_layer = _insert_pressure(p_top, p, potential_temperature(t, p), q)
    dp = -np.diff(p_layer, axis=0)
    inside = p_layer[1:] >= p_top

    def layer_mean(x: np.ndarray) -> np.ndarray:
        return np.where(inside, 0.5 * (x[:-1] + x[1:]) * dp, 0.0).sum(axis=0) / ML_DEPTH

    return sp, temperature_from_potential_temperature(layer_mean(theta), sp), layer_mean(q_layer)


def _most_unstable_parcel(
    p: np.ndarray, height: np.ndarray, t: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pressure, temperature and specific humidity of the column point within
    ``MU_DEPTH`` of the surface with the highest equivalent potential
    temperature."""
    ept = np.where(p >= p[0] - MU_DEPTH, ept_from_specific_humidity(t, q, p, method=EPT_METHOD), -np.inf)
    k = np.argmax(ept, axis=0)[None]
    return tuple(np.take_along_axis(x, k, axis=0)[0] for x in (p, t, q))


def _parcel_cape_cin(
    p: np.ndarray,
    height: np.ndarray,
    t_env: np.ndarray,
    q_env: np.ndarray,
    p0: np.ndarray,
    t0: np.ndarray,
    q0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CAPE and CIN (J/kg) of a parcel starting at (``p0``, ``t0``, ``q0``),
    a point of the environment column (``p``, ``height``, ``t_env``,
    ``q_env``).

    The parcel rises along the dry adiabat up to its LCL, then along the
    pseudo-adiabat of its equivalent potential temperature. The LCL is
    inserted into the column so the kink of the parcel curve is resolved;
    buoyancy uses virtual temperature for parcel and environment and is
    integrated in geometric height by ``_cape_cin``.
    """
    td0 = np.minimum(dewpoint_from_specific_humidity(q0, p0), t0)
    _, p_lcl = lcl(t0, td0, p0)
    p_lcl = np.minimum(p_lcl, p0)
    p, height, t_env, q_env = _insert_pressure(p_lcl, p, height, t_env, q_env)

    saturated = p < p_lcl
    t_parcel = temperature_on_dry_adiabat(p, t0, p0)
    # earthkit's pseudo-adiabat solver takes 1-D arrays; only saturated points need it.
    ept = np.broadcast_to(ept_from_dewpoint(t0, td0, p0, method=EPT_METHOD), p.shape)
    t_parcel[saturated] = temperature_on_moist_adiabat(ept[saturated], p[saturated], ept_method=EPT_METHOD)
    q_parcel = np.where(saturated, saturation_specific_humidity(t_parcel, p), q0)

    tv_env = virtual_temperature(t_env, q_env)
    buoyancy = constants.g * (virtual_temperature(t_parcel, q_parcel) - tv_env) / tv_env
    return _cape_cin(buoyancy, height, p, p_lcl, p0)


def _insert_pressure(p_new: np.ndarray, p: np.ndarray, *profiles: np.ndarray) -> list[np.ndarray]:
    """Insert a point at pressure ``p_new`` into each bottom-up column.

    ``p`` and ``profiles`` have shape (levels, points), with ``p`` not
    increasing upwards. The profiles are interpolated linearly in log
    pressure. Returns ``p`` and the profiles with one more level.
    """
    log_p, log_new = np.log(p), np.log(p_new)[None]
    upper = np.clip((p >= p_new).sum(axis=0), 1, p.shape[0] - 1)[None]  # first point above p_new
    lower = upper - 1
    lp0, lp1 = np.take_along_axis(log_p, lower, axis=0), np.take_along_axis(log_p, upper, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.clip(np.nan_to_num((lp0 - log_new) / (lp0 - lp1)), 0.0, 1.0)
    rows = [p_new[None]]
    for x in profiles:
        x0, x1 = np.take_along_axis(x, lower, axis=0), np.take_along_axis(x, upper, axis=0)
        rows.append(x0 + weight * (x1 - x0))
    order = np.argsort(-np.vstack([p, rows[0]]), axis=0, kind="stable")
    return [np.take_along_axis(np.vstack([x, row]), order, axis=0) for x, row in zip((p, *profiles), rows)]


def _cape_cin(
    buoyancy: np.ndarray, height: np.ndarray, pressure: np.ndarray, p_lcl: np.ndarray, p_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CAPE and CIN (J/kg) from parcel buoyancy (m/s2) on a bottom-up column.

    Buoyancy is linear between the column points, so each segment's area is
    split exactly at its zero crossing. The LFC is the lowest point at or
    above the LCL (which must be a column point) where the parcel becomes
    buoyant, the EL the highest crossing back to non-positive buoyancy above
    it (the column top if the parcel is still buoyant there). CAPE is the area
    from the LFC to the EL, including any negative layers in between, and CIN
    the negative area below the LFC; positive buoyancy below the LCL (e.g. a
    superadiabatic surface layer) counts for neither. Segments below the
    parcel's starting pressure ``p_start`` (a column point) are ignored.
    Without an LFC, CAPE is 0 and CIN missing.
    """
    b0, b1 = buoyancy[:-1], buoyancy[1:]
    dz = np.diff(height, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zero = np.clip(b0 / (b0 - b1), 0.0, 1.0)  # zero crossing as a fraction of the segment
    p0 = pressure[:-1]
    dz = np.where(p0 <= p_start, dz, 0.0)  # nothing below the parcel's start counts
    area = 0.5 * (b0 + b1) * dz
    positive = np.where(
        (b0 >= 0) & (b1 >= 0),
        area,
        np.where((b0 <= 0) & (b1 <= 0), 0.0, np.where(b0 > 0, 0.5 * b0 * zero * dz, 0.5 * b1 * (1.0 - zero) * dz)),
    )
    negative = area - positive

    n = b0.shape[0]
    segment = np.arange(n)[:, None]
    # A segment starting at the LCL with positive buoyancy has its LFC at the LCL.
    lfc_segment = (p0 <= p_lcl) & (b1 > 0) & ((b0 <= 0) | (p0 == p_lcl))
    has_lfc = lfc_segment.any(axis=0)
    k_lfc = np.argmax(lfc_segment, axis=0)
    el_segment = (b0 > 0) & (b1 <= 0) & (segment > k_lfc)
    k_el = np.where(el_segment.any(axis=0), n - 1 - np.argmax(el_segment[::-1], axis=0), n)

    cape_area = np.where(
        (segment == k_lfc) | (segment == k_el),
        positive,
        np.where((segment > k_lfc) & (segment < k_el), area, 0.0),
    )
    cape = np.where(has_lfc, np.maximum(cape_area.sum(axis=0), 0.0), 0.0)
    cin = np.where(has_lfc, np.where(segment <= k_lfc, negative, 0.0).sum(axis=0), np.nan)
    return cape, cin


def _meltlayer_half_depth(snowlmt: np.ndarray, meltlayer_top: np.ndarray) -> np.ndarray:
    """MELTLAYER_TOP - SNOWLMT clamped to [150, 375] m, as INCA's ``dzml``."""
    return np.clip(meltlayer_top - snowlmt, MELTLAYER_HALF_DEPTH_MIN, MELTLAYER_HALF_DEPTH_MAX)


def _precipitation_type(
    rate: np.ndarray,
    snowlmt: np.ndarray,
    meltlayer_top: np.ndarray,
    t_2m: np.ndarray,
    *,
    ground_temperature: np.ndarray,
    orography: np.ndarray,
) -> np.ndarray:
    """INCA's ``precipitation_type`` without hail: 0 none, 1 rain, 2 mixed,
    3 snow, 4 freezing rain; missing where ``rate`` (mm/h) is.

    The ground's position relative to the melting layer around SNOWLMT gives
    snow, mixed or rain; rain on ground below 0.1 degC becomes freezing rain,
    unless the 2 m air is above 0.5 degC.
    """
    dzml = _meltlayer_half_depth(snowlmt, meltlayer_top)
    wet = rate > PREC_RATE_MIN
    above_snowlmt = orography - snowlmt
    prec_type = np.select(
        [
            ~wet,
            above_snowlmt >= dzml,
            above_snowlmt > -dzml,
        ],
        [PREC_TYPE_NONE, PREC_TYPE_SNOW, PREC_TYPE_MIXED],
        PREC_TYPE_RAIN,
    ).astype(np.float64)
    freezing = (prec_type == PREC_TYPE_RAIN) & (ground_temperature < FREEZING_GROUND_T) & (t_2m <= FREEZING_MAX_AIR_T)
    prec_type[freezing] = PREC_TYPE_FREEZING_RAIN
    prec_type[np.isnan(rate)] = np.nan
    return prec_type


def _snow_fraction(snowlmt: np.ndarray, meltlayer_top: np.ndarray, orography: np.ndarray, method: str) -> np.ndarray:
    """Fraction of precipitation falling as snow, as INCA's
    ``precip_snow_water_equivalent``: 0 at or below SNOWLMT, 1 at or above
    SNOWLMT + the melting layer depth, linear or logistic in between."""
    x = np.clip((orography - snowlmt) / _meltlayer_half_depth(snowlmt, meltlayer_top), 0.0, 1.0)
    if method == "linear":
        return x
    return 1.0 / (1.0 + np.exp(-(x - 0.5) * 10.0))
