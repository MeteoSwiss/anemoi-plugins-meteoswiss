"""Diagnostics derived from the Varda inference output.

- SP_10M, DD_10M: 10 m wind speed and direction.
- RELHUM_2M: 2 m relative humidity (%).
- TW_2M: 2 m wet-bulb temperature.
- HZEROCL, SNOWLMT: freezing level and snowfall limit, as in INCA.
- CAPE_ML, CIN_ML, CAPE_MU, CIN_MU: mixed-layer and most-unstable CAPE/CIN (J/kg).
- PTYPE: precipitation type, WMO code table 4.201 (as ICON).
- TOT_SNOW: snowfall accumulated from the start of the forecast (kg m-2).

Usage::

    typed_variables:
      SP_10M:
        mars: {param: SP_10M, levtype: sfc}
      ...  # one entry per output

    output:
      tee:
      - grib:
          post_processors:
          - forward_transform_filter:
              derive-diagnostic:
                orography_file: /path/to/external_parameter_icon_grid.nc
                variables: [SP_10M, DD_10M, RELHUM_2M, TW_2M, HZEROCL, SNOWLMT,
                            CAPE_ML, CIN_ML, CAPE_MU, CIN_MU, PTYPE, TOT_SNOW]

The inputs are read from the state by their GRIB name. Only
the inputs needed by the requested ``variables`` are used.

Notes
-----
- Each diagnostic needs a ``typed_variables`` entry and a GRIB template, since the
  checkpoint never produces it.
- Must be the last transform filter of its ``post_processors`` list: a later
  filter re-wraps the state against the checkpoint's variable catalog, which
  doesn't contain the fields added here.
- The orography file must be on the grid and cell order of the state, so the
  filter must run before any regridding.
- PTYPE and TOT_SNOW use the precipitation since the previous step, so the
  filter keeps state between steps and needs them in order.
- With pressure levels near the ground only, the height levels are
  interpolated over coarse layers and CAPE/CIN are rough estimates.
- ``n_threads`` (default: all CPUs available to the process, e.g. the Slurm
  allocation) splits the grid points of the expensive diagnostics (TW_2M,
  wet-bulb profile, CAPE/CIN) into chunks computed in parallel threads.
"""

import logging
import os
import time
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any
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

ZERO_CELSIUS = 273.15  # K
HEIGHT_2M_FIELDS = 2.0  # m above the orography
PA_PER_HPA = 100.0

# INCA height levels: thresholds, and fallbacks for columns without a crossing.
HZEROCL_THRESHOLD = ZERO_CELSIUS  # K, temperature at the freezing level
SNOWLMT_THRESHOLD = ZERO_CELSIUS + 1.3  # K, wet-bulb temperature at the snowfall limit
MELTING_LAYER_TOP_THRESHOLD = ZERO_CELSIUS + 0.2  # K, wet-bulb temperature where falling snow starts to melt
T_LAPSE_RATE = 0.65 / 100.0  # K/m, extrapolates HZEROCL
TW_LAPSE_RATE = 0.5 / 100.0  # K/m, extrapolates SNOWLMT and MELTLAYER_TOP
MIN_HEIGHT = -100.0  # m, lower bound of HZEROCL, SNOWLMT and MELTLAYER_TOP, which can lie below ground
MELTING_LAYER_TOP_OFFSET = 150.0  # m above ground, MELTLAYER_TOP when only SNOWLMT has a crossing
MIN_MELTING_LAYER_DEPTH = 75.0  # m, minimum MELTLAYER_TOP - SNOWLMT

# CAPE/CIN. Bolton (1980) eq. 39 is the equivalent potential temperature of
# the pseudo-adiabat.
EPT_METHOD = "bolton39"
ML_DEPTH = 5000.0  # Pa, depth of the mixed layer averaged for CAPE_ML/CIN_ML
MU_DEPTH = 30000.0  # Pa, depth searched for the most unstable parcel (CAPE_MU/CIN_MU)

# INCA precipitation type and snow fraction.
PREC_RATE_MIN = 0.1  # mm/h, minimum rate to assign a precipitation type
MELTING_LAYER_DEPTH_MIN = 150.0  # m, MELTLAYER_TOP - SNOWLMT is clamped to [min, max]
MELTING_LAYER_DEPTH_MAX = 2.5 * MELTING_LAYER_DEPTH_MIN
FREEZING_GROUND_T = ZERO_CELSIUS + 0.1  # K, rain on colder ground is freezing rain...
FREEZING_MAX_AIR_T = ZERO_CELSIUS + 0.5  # K, ...unless the air is warmer than this
SNOW_FRACTION_LOGISTIC_MIDPOINT = 0.5
SNOW_FRACTION_LOGISTIC_STEEPNESS = 10.0
# PTYPE codes, WMO GRIB2 code table 4.201
PTYPE_NONE, PTYPE_RAIN, PTYPE_MIXED, PTYPE_SNOW, PTYPE_FREEZING_RAIN = 0, 1, 7, 5, 3

_WIND = ("u_10m", "v_10m")
_HUMIDITY = ("temperature_2m", "dewpoint_2m")
_MOIST_PROFILE = (
    *_HUMIDITY,
    "surface_pressure",
    "temperature_levels",
    "specific_humidity_levels",
    "geopotential_levels",
)
# Inputs needed by each output.
_SUPPORTED_VARIABLES: dict[str, tuple[str, ...]] = {
    "SP_10M": _WIND,
    "DD_10M": _WIND,
    "RELHUM_2M": _HUMIDITY,
    "TW_2M": (*_HUMIDITY, "surface_pressure"),
    "HZEROCL": ("temperature_2m", "surface_pressure", "temperature_levels", "geopotential_levels"),
    "SNOWLMT": _MOIST_PROFILE,
    "CAPE_ML": _MOIST_PROFILE,
    "CIN_ML": _MOIST_PROFILE,
    "CAPE_MU": _MOIST_PROFILE,
    "CIN_MU": _MOIST_PROFILE,
    "PTYPE": (*_MOIST_PROFILE, "accumulated_precipitation"),
    "TOT_SNOW": (*_MOIST_PROFILE, "accumulated_precipitation"),
}
# Outputs encoded with the wind template; all others use the T_2M one.
_WIND_VARIABLES = {"SP_10M", "DD_10M"}
# Outputs computed from the pressure-level profiles and the orography.
_VERTICAL_VARIABLES = {
    "HZEROCL",
    "SNOWLMT",
    "CAPE_ML",
    "CIN_ML",
    "CAPE_MU",
    "CIN_MU",
    "PTYPE",
    "TOT_SNOW",
}
# Outputs needing SNOWLMT and MELTLAYER_TOP, whether or not SNOWLMT is requested.
_MELTING_LAYER_VARIABLES = {"PTYPE", "TOT_SNOW"}
_ALL_INPUTS = (
    "u_10m",
    "v_10m",
    "temperature_2m",
    "dewpoint_2m",
    "surface_pressure",
    "temperature_levels",
    "specific_humidity_levels",
    "geopotential_levels",
    "accumulated_precipitation",
)
# GRIB names of the inputs in the Varda output state; precipitation is
# accumulated from the start of the forecast, in kg m-2.
_DEFAULT_INPUTS = {
    "u_10m": "U_10M",
    "v_10m": "V_10M",
    "temperature_2m": "T_2M",
    "dewpoint_2m": "TD_2M",
    "surface_pressure": "PS",
    "temperature_levels": "T",
    "specific_humidity_levels": "QV",
    "geopotential_levels": "FI",
    "accumulated_precipitation": "TOT_PREC",
}


class DeriveDiagnostic(MatchingFieldsFilter):
    """Derive diagnostics starting from the Varda output.

    How it is called, once per output step:

    1. anemoi-inference's ``forward_transform_filter`` post-processor passes
       the state as a FieldList to ``forward``.
    2. ``forward`` handles the initial state, make sure all fields needed are present,
       then calls ``MatchingFieldsFilter.forward``.
    3. ``MatchingFieldsFilter.forward`` picks the inputs listed in
       ``MATCHING.forward`` from the state by ``param`` (all pressure levels
       of a param together when ``MATCHING.vertical``) and calls
       ``forward_transform`` with them as keyword arguments.
    4. ``forward_transform`` computes the requested diagnostics and yields
       them as new fields. The base class returns them together with the
       inputs and the fields it didn't match; anemoi-inference then writes
       them to GRIB using ``typed_variables`` and the GRIB templates.

    ``MATCHING`` below lists every possible input; ``__init__`` narrows it
    to the inputs the requested ``variables`` need.
    """

    MATCHING = MatchingSpec(
        select="param",
        forward=_ALL_INPUTS,
        return_inputs="all",
    )

    def __init__(
        self,
        *,
        variables: str | list[str],
        u_10m: str | None = _DEFAULT_INPUTS["u_10m"],
        v_10m: str | None = _DEFAULT_INPUTS["v_10m"],
        temperature_2m: str | None = _DEFAULT_INPUTS["temperature_2m"],
        dewpoint_2m: str | None = _DEFAULT_INPUTS["dewpoint_2m"],
        surface_pressure: str | None = _DEFAULT_INPUTS["surface_pressure"],
        temperature_levels: str | None = _DEFAULT_INPUTS["temperature_levels"],
        specific_humidity_levels: str | None = _DEFAULT_INPUTS["specific_humidity_levels"],
        geopotential_levels: str | None = _DEFAULT_INPUTS["geopotential_levels"],
        orography_file: str | None = None,
        orography_variable: str = "topography_c",
        accumulated_precipitation: str | None = _DEFAULT_INPUTS["accumulated_precipitation"],
        snow_fraction_profile: Literal["linear", "logistic"] = "linear",
        n_threads: int | None = None,
    ) -> None:
        """``variables`` selects the diagnostics to derive, any of ``_SUPPORTED_VARIABLES``.
        The inputs (``u_10m`` ... ``accumulated_precipitation``) are the
        ``param`` names of the fields in the state, by default their GRIB
        names. ``orography_file`` (NetCDF, variable ``orography_variable`` in m) is
        required for the outputs computed from the pressure levels.
        ``accumulated_precipitation`` is accumulated from the start of the
        forecast (kg m-2). ``snow_fraction_profile`` is the snow fraction
        across the melting layer used for ``TOT_SNOW``. ``n_threads`` is the
        number of threads the grid points are split over (default: all CPUs
        available to the process; 1 computes them all in the calling thread).
        """
        if isinstance(variables, str):
            variables = [variables]
        variables = list(variables)
        if not variables:
            raise ValueError(f"`variables` must list at least one of: {sorted(_SUPPORTED_VARIABLES)}.")
        unknown = set(variables) - set(_SUPPORTED_VARIABLES)
        if unknown:
            raise ValueError(f"Unsupported variable(s) {sorted(unknown)}; supported: {sorted(_SUPPORTED_VARIABLES)}.")
        if snow_fraction_profile not in ("linear", "logistic"):
            raise ValueError(f"`snow_fraction_profile` must be 'linear' or 'logistic', got {snow_fraction_profile!r}.")
        if n_threads is None:
            n_threads = _available_cpus()
        if not isinstance(n_threads, int) or n_threads < 1:
            raise ValueError(f"`n_threads` must be a positive integer, got {n_threads!r}.")

        self.u_10m = u_10m
        self.v_10m = v_10m
        self.temperature_2m = temperature_2m
        self.dewpoint_2m = dewpoint_2m
        self.surface_pressure = surface_pressure
        self.temperature_levels = temperature_levels
        self.specific_humidity_levels = specific_humidity_levels
        self.geopotential_levels = geopotential_levels
        self.accumulated_precipitation = accumulated_precipitation
        self.snow_fraction_profile = snow_fraction_profile
        self.n_threads = n_threads
        self.variables = variables

        requested = set(variables)
        self._needs_melting_layer = bool(_MELTING_LAYER_VARIABLES & requested)
        self._needs_wet_bulb_profile = self._needs_melting_layer or "SNOWLMT" in requested
        self._needs_tw_2m = "TW_2M" in requested or self._needs_wet_bulb_profile
        self._parcels = [kind for kind in ("ML", "MU") if {f"CAPE_{kind}", f"CIN_{kind}"} & requested]

        # Precipitation at the previous step and the snow accumulated so far,
        # carried from one call (output step) to the next.
        self._previous_precipitation: tuple[timedelta, np.ndarray] | None = None
        self._snow_total: np.ndarray | None = None

        required = {name for v in variables for name in _SUPPORTED_VARIABLES[v]}
        missing = sorted(name for name in required if getattr(self, name) is None)
        if missing:
            raise ValueError(f"Variables {variables} require the input(s) {missing}.")

        vertical = bool(_VERTICAL_VARIABLES & requested)
        self._orography = None
        if vertical:
            if orography_file is None:
                raise ValueError(f"{sorted(_VERTICAL_VARIABLES & requested)} require `orography_file`.")
            self._orography = _load_orography(Path(orography_file), orography_variable)

        # Match only the inputs the requested variables need. Grouped by level,
        # a pressure-level input arrives as a FieldList, which the base class
        # can't return, so forward_transform yields the inputs itself.
        self.MATCHING = replace(
            self.MATCHING,
            forward=tuple(name for name in _ALL_INPUTS if name in required),
            return_inputs="none" if vertical else "all",
            vertical=vertical,
        )

        LOG.info(
            "derive-diagnostic: will add %s to the output, computed from %s with %d thread(s)",
            self.variables,
            {name: getattr(self, name) for name in self.MATCHING.forward},
            self.n_threads,
        )
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Run the filter on one output state.

        The initial state (lead time 0) has no precipitation field: nothing
        has accumulated yet, so the filter runs without it. At any other lead
        time missing precipitation is an error. The check should run on TOT_PREC
        only because it is an input. TOT_SNOW is an output and requires no check
        """
        if "accumulated_precipitation" not in self.MATCHING.forward or self.accumulated_precipitation in data.metadata(
            "param"
        ):
            return super().forward(data)
        leads = {_leadtime(f) for f in data} - {None}
        if leads != {timedelta(0)}:
            raise ValueError(
                f"derive-diagnostic: precipitation {self.accumulated_precipitation!r} missing at lead time(s) "
                f"{sorted(str(lead) for lead in leads)}."
            )
        matching = self.MATCHING
        self.MATCHING = replace(
            matching, forward=tuple(name for name in matching.forward if name != "accumulated_precipitation")
        )
        try:
            return super().forward(data)
        finally:
            self.MATCHING = matching

    def forward_transform(
        self,
        u_10m: ekd.Field | None = None,
        v_10m: ekd.Field | None = None,
        temperature_2m: ekd.Field | None = None,
        dewpoint_2m: ekd.Field | None = None,
        surface_pressure: ekd.Field | None = None,
        temperature_levels: ekd.FieldList | None = None,
        specific_humidity_levels: ekd.FieldList | None = None,
        geopotential_levels: ekd.FieldList | None = None,
        accumulated_precipitation: ekd.Field | None = None,
    ) -> Iterator[ekd.Field]:
        """Yield the requested diagnostics, preceded by the inputs when they
        are grouped by level. Inputs not needed by the request are None."""
        if self.MATCHING.vertical:
            inputs = (
                u_10m,
                v_10m,
                temperature_2m,
                dewpoint_2m,
                surface_pressure,
                temperature_levels,
                specific_humidity_levels,
                geopotential_levels,
                accumulated_precipitation,
            )
            for x in inputs:
                if isinstance(x, ekd.Field):
                    yield x
                elif x is not None:
                    yield from x
            u_10m, v_10m, temperature_2m, dewpoint_2m, surface_pressure, accumulated_precipitation = (
                _as_single_field(x)
                for x in (u_10m, v_10m, temperature_2m, dewpoint_2m, surface_pressure, accumulated_precipitation)
            )

        timings: dict[str, float] = {}
        t_2m = None if temperature_2m is None else temperature_2m.to_numpy(flatten=True)
        sp = None if surface_pressure is None else surface_pressure.to_numpy(flatten=True)
        # The model can predict TD_2M above T_2M (supersaturation). Cap it at
        # T_2M, so that RELHUM_2M <= 100 %.
        td_2m = None if dewpoint_2m is None else np.minimum(dewpoint_2m.to_numpy(flatten=True), t_2m)

        # May hold intermediate fields (e.g. MELTLAYER_TOP); only the requested ones are yielded.
        diagnostics: dict[str, np.ndarray] = {}
        if u_10m is not None:
            diagnostics.update(self._wind_diagnostics(u_10m.to_numpy(), v_10m.to_numpy(), timings))

        if "RELHUM_2M" in self.variables:
            with _timer(timings, "RELHUM_2M"):
                # min(): rounding can give 100.00x where TD_2M == T_2M
                diagnostics["RELHUM_2M"] = np.minimum(relative_humidity_from_dewpoint(t_2m, td_2m), 100.0)

        if self._needs_tw_2m:
            with _timer(timings, "TW_2M"):
                tw_2m = _map_over_points(
                    wet_bulb_temperature_from_dewpoint, self.n_threads, per_point={"t": t_2m, "td": td_2m, "p": sp}
                )
                # Normand's rule can overshoot T_2M by a few tenths of a K near saturation
                diagnostics["TW_2M"] = np.clip(tw_2m, td_2m, t_2m)

        if self.MATCHING.vertical:
            diagnostics.update(
                self._vertical_diagnostics(
                    t_2m=t_2m,
                    td_2m=td_2m,
                    tw_2m=diagnostics.get("TW_2M"),
                    sp=sp,
                    temperature_levels=temperature_levels,
                    specific_humidity_levels=specific_humidity_levels,
                    geopotential_levels=geopotential_levels,
                    accumulated_precipitation=accumulated_precipitation,
                    timings=timings,
                )
            )

        _log_timings(timings, temperature_2m if temperature_2m is not None else u_10m)
        for name in self.variables:
            template = u_10m if name in _WIND_VARIABLES else temperature_2m
            yield self.new_field_from_numpy(
                diagnostics[name].reshape(template.shape),
                template=template,
                name=name,
                param=name,
                shortName=name,
            )

    def _wind_diagnostics(self, u: np.ndarray, v: np.ndarray, timings: dict[str, float]) -> dict[str, np.ndarray]:
        """SP_10M and DD_10M, as requested."""
        diagnostics = {}
        if "SP_10M" in self.variables:
            with _timer(timings, "SP_10M"):
                diagnostics["SP_10M"] = np.sqrt(u**2 + v**2)
        if "DD_10M" in self.variables:
            with _timer(timings, "DD_10M"):
                # Meteorological convention: where the wind blows FROM, clockwise from North.
                diagnostics["DD_10M"] = np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)
        return diagnostics

    def _vertical_diagnostics(
        self,
        *,
        t_2m: np.ndarray,
        td_2m: np.ndarray | None,
        tw_2m: np.ndarray | None,
        sp: np.ndarray,
        temperature_levels: ekd.FieldList,
        specific_humidity_levels: ekd.FieldList | None,
        geopotential_levels: ekd.FieldList,
        accumulated_precipitation: ekd.Field | None,
        timings: dict[str, float],
    ) -> dict[str, np.ndarray]:
        """Height levels, PTYPE, TOT_SNOW and CAPE/CIN from the pressure-level profiles."""
        orography = self._orography
        if orography.size != t_2m.size:
            raise ValueError(
                f"Orography has {orography.size} points but the state has {t_2m.size}; "
                "derive-diagnostic must run on the native grid of the orography file."
            )

        level_fields = {"t": temperature_levels, "z": geopotential_levels}
        if self._needs_wet_bulb_profile or self._parcels:
            level_fields["q"] = specific_humidity_levels
        with _timer(timings, "pressure-level profiles"):
            pressure, profiles = _stack_pressure_levels(level_fields)
            level_heights = geometric_height_from_geopotential(profiles["z"])
            above_ground = pressure[:, None] < sp[None, :]

        diagnostics = {}
        if "HZEROCL" in self.variables:
            with _timer(timings, "HZEROCL"):
                diagnostics["HZEROCL"] = _freezing_level(
                    t_2m=t_2m,
                    t_levels=profiles["t"],
                    orography=orography,
                    level_heights=level_heights,
                    above_ground=above_ground,
                )

        if self._needs_wet_bulb_profile:
            with _timer(timings, "SNOWLMT/MELTLAYER_TOP"):
                diagnostics.update(
                    _snow_and_melting_levels(
                        tw_2m=tw_2m,
                        tw_levels=_map_over_points(
                            _wet_bulb_levels,
                            self.n_threads,
                            per_point={"t": profiles["t"], "q": profiles["q"], "above_ground": above_ground},
                            shared={"pressure": pressure},
                        ),
                        orography=orography,
                        level_heights=level_heights,
                        above_ground=above_ground,
                    )
                )

        if self._needs_melting_layer:
            with _timer(timings, "PTYPE/TOT_SNOW"):
                diagnostics.update(
                    self._precipitation_type_and_snow(
                        accumulated_precipitation,
                        t_2m=t_2m,
                        snowlmt=diagnostics["SNOWLMT"],
                        melting_layer_top=diagnostics["MELTLAYER_TOP"],
                    )
                )

        if self._parcels:
            with _timer(timings, "CAPE/CIN column"):
                p, height, t, q = _environment_column(
                    t_2m=t_2m,
                    td_2m=td_2m,
                    sp=sp,
                    orography=orography,
                    pressure=pressure,
                    t_levels=profiles["t"],
                    q_levels=profiles["q"],
                    level_heights=level_heights,
                    above_ground=above_ground,
                )
            for kind in self._parcels:
                with _timer(timings, f"CAPE_{kind}/CIN_{kind}"):
                    diagnostics[f"CAPE_{kind}"], diagnostics[f"CIN_{kind}"] = _map_over_points(
                        _cape_cin,
                        self.n_threads,
                        per_point={"p": p, "height": height, "t": t, "q": q},
                        shared={"kind": kind},
                    )

        return diagnostics

    def _precipitation_type_and_snow(
        self,
        accumulated_precipitation: ekd.Field | None,
        *,
        t_2m: np.ndarray,
        snowlmt: np.ndarray,
        melting_layer_top: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """PTYPE and TOT_SNOW from the precipitation since the previous step.

        At the first step of a forecast (no previous step, or a lead time not
        after the previous one) PTYPE is missing and TOT_SNOW restarts from 0.
        ``accumulated_precipitation`` is None at the initial state (lead time
        0), where nothing has accumulated yet.
        """
        if accumulated_precipitation is None:
            lead, total = timedelta(0), np.zeros_like(t_2m)
        else:
            lead = _leadtime(accumulated_precipitation)
            if lead is None:
                raise ValueError("derive-diagnostic: can't determine the lead time of the precipitation field.")
            total = accumulated_precipitation.to_numpy(flatten=True)

        previous = self._previous_precipitation
        if previous is None or lead <= previous[0]:
            amount = np.full_like(total, np.nan)
            hours = 1.0
            self._snow_total = np.zeros_like(total)
        else:
            previous_lead, previous_total = previous
            # Accumulations should only grow; clip the drops where the
            # forecaster's precipitation is swapped in at the overlap steps.
            amount = np.maximum(total - previous_total, 0.0)
            hours = (lead - previous_lead) / timedelta(hours=1)
        self._previous_precipitation = (lead, total)

        diagnostics = {}
        if "PTYPE" in self.variables:
            diagnostics["PTYPE"] = _precipitation_type(
                amount / hours,
                snowlmt,
                melting_layer_top,
                t_2m,
                ground_temperature=t_2m,  # Varda has no ground temperature
                orography=self._orography,
            )
        if "TOT_SNOW" in self.variables:
            fraction = _snow_fraction(snowlmt, melting_layer_top, self._orography, self.snow_fraction_profile)
            self._snow_total = self._snow_total + np.nan_to_num(fraction * amount)
            diagnostics["TOT_SNOW"] = self._snow_total.copy()
        return diagnostics


@contextmanager
def _timer(timings: dict[str, float], label: str) -> Generator[None, None, None]:
    """Add the wall time spent in the block to ``timings[label]`` (seconds)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[label] = timings.get(label, 0.0) + time.perf_counter() - start


def _log_timings(timings: dict[str, float], field: ekd.Field | None) -> None:
    """Log the time spent on each diagnostic for one output step (i.e. one lead
    time of the forecast)."""
    if not timings:
        return
    lead = _leadtime(field) if field is not None else None
    LOG.info(
        "derive-diagnostic: computed step %s in %.2fs (%s)",
        "?" if lead is None else f"{lead / timedelta(hours=1):g}h",
        sum(timings.values()),
        ", ".join(f"{label} {seconds:.2f}s" for label, seconds in timings.items()),
    )


def _available_cpus() -> int:
    """CPUs this process may run on (e.g. the Slurm allocation), not all of the node's."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


def _map_over_points(
    func: Callable[..., Any],
    n_threads: int,
    *,
    per_point: dict[str, np.ndarray],
    shared: dict[str, Any] | None = None,
) -> Any:
    """``func(**per_point, **shared)``, with the points (last axis of the
    ``per_point`` arrays) split into ``n_threads`` chunks computed in parallel.

    ``func`` must compute each point independently of the others and return
    an array, or a tuple of arrays, with the points on the last axis; the
    chunks are concatenated back in order. numpy releases the GIL in its array
    operations, so the threads run concurrently. With ``n_threads=1``,
    ``func`` runs once on all points in the calling thread.
    """
    shared = shared or {}
    n_points = next(iter(per_point.values())).shape[-1]
    n_chunks = min(n_threads, n_points)
    if n_chunks <= 1:
        return func(**per_point, **shared)
    bounds = np.linspace(0, n_points, n_chunks + 1).astype(int)

    def run_chunk(i: int) -> Any:
        points = slice(bounds[i], bounds[i + 1])
        return func(**{name: x[..., points] for name, x in per_point.items()}, **shared)

    with ThreadPoolExecutor(n_chunks) as executor:
        chunks = list(executor.map(run_chunk, range(n_chunks)))
    if isinstance(chunks[0], tuple):
        return tuple(np.concatenate(parts, axis=-1) for parts in zip(*chunks))
    return np.concatenate(chunks, axis=-1)


def _load_orography(path: Path, variable: str) -> np.ndarray:
    """Load the model orography (m), flattened in native cell order."""
    with xr.open_dataset(path) as ds:
        orography = ds[variable].values.astype(np.float64).ravel()
    LOG.info("derive-diagnostic: orography (%d points) loaded from %s", orography.size, path)
    return orography


def _as_single_field(x: ekd.Field | ekd.FieldList | None) -> ekd.Field | None:
    """Grouped by level, a surface field carrying a level arrives as a
    one-element FieldList; unwrap it."""
    if x is None or isinstance(x, ekd.Field):
        return x
    if len(x) != 1:
        raise ValueError(f"Expected a single surface field, got {len(x)} fields.")
    return x[0]


def _values_at_level(array: np.ndarray, level_index: np.ndarray) -> np.ndarray:
    """Values of ``array`` (levels, points) at one level per point."""
    return np.take_along_axis(array, level_index[None, :], axis=0)[0]


def _last_true_index(mask: np.ndarray) -> np.ndarray:
    """Index of the last True along the first axis of ``mask`` (levels,
    points); 0 for points without any."""
    return mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0)


def _stack_pressure_levels(inputs: dict[str, ekd.FieldList]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    return np.array(ordered) * PA_PER_HPA, profiles


def _profile_with_surface_point(
    surface_value: np.ndarray,
    surface_height: np.ndarray,
    level_values: np.ndarray,
    level_heights: np.ndarray,
    above_ground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bottom-up values and heights: the surface point followed by the
    pressure levels.

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
    below, above = values[:-1], values[1:]
    crossing = (below >= threshold) & (above < threshold)
    index = _last_true_index(crossing)

    v_below, v_above = _values_at_level(below, index), _values_at_level(above, index)
    h_below, h_above = _values_at_level(heights[:-1], index), _values_at_level(heights[1:], index)
    with np.errstate(divide="ignore", invalid="ignore"):
        height = h_below + (threshold - v_below) / (v_above - v_below) * (h_above - h_below)
    return np.where(crossing.any(axis=0), height, np.nan)


def _extrapolated_height(orography: np.ndarray, threshold_minus_ground: np.ndarray, lapse_rate: float) -> np.ndarray:
    """Height of a level without a crossing, extrapolated from the ground with
    ``lapse_rate``; below ground where ``threshold_minus_ground`` > 0."""
    return orography - threshold_minus_ground / lapse_rate


def _wet_bulb_levels(t: np.ndarray, q: np.ndarray, above_ground: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Wet-bulb temperature (K) on the pressure levels, shape (levels, points).

    Computed above ground only; below ground it is NaN, as the surface point
    replaces those levels in the profile (``_profile_with_surface_point``).
    """
    tw = np.full_like(t, np.nan)
    for level, p in enumerate(pressure):
        points = above_ground[level]
        if points.any():
            t_level = t[level, points]
            tw[level, points] = wet_bulb_temperature_from_specific_humidity(
                t_level, q[level, points], np.full_like(t_level, p)
            )
    return tw


def _freezing_level(
    *,
    t_2m: np.ndarray,
    t_levels: np.ndarray,
    orography: np.ndarray,
    level_heights: np.ndarray,
    above_ground: np.ndarray,
) -> np.ndarray:
    """HZEROCL: the uppermost crossing of the temperature profile through 0 degC.

    Without a crossing it is extrapolated below ground with ``T_LAPSE_RATE``,
    down to ``MIN_HEIGHT``.
    """
    values, heights = _profile_with_surface_point(
        t_2m, orography + HEIGHT_2M_FIELDS, t_levels, level_heights, above_ground
    )
    freezing_level = _uppermost_crossing(values, heights, HZEROCL_THRESHOLD)
    extrapolated = _extrapolated_height(orography, np.abs(t_2m - HZEROCL_THRESHOLD), T_LAPSE_RATE)
    return np.maximum(np.where(np.isnan(freezing_level), extrapolated, freezing_level), MIN_HEIGHT)


def _snow_and_melting_levels(
    *,
    tw_2m: np.ndarray,
    tw_levels: np.ndarray,
    orography: np.ndarray,
    level_heights: np.ndarray,
    above_ground: np.ndarray,
) -> dict[str, np.ndarray]:
    """SNOWLMT and MELTLAYER_TOP: the uppermost crossings of the wet-bulb
    profile through ``SNOWLMT_THRESHOLD`` and ``MELTING_LAYER_TOP_THRESHOLD``.

    Without a crossing, SNOWLMT is extrapolated from the ground with
    ``TW_LAPSE_RATE``. MELTLAYER_TOP is extrapolated the same way if SNOWLMT
    has no crossing either, else set ``MELTING_LAYER_TOP_OFFSET`` above ground.
    Both are clamped at ``MIN_HEIGHT``, and MELTLAYER_TOP is kept at least
    ``MIN_MELTING_LAYER_DEPTH`` above SNOWLMT.
    """
    values, heights = _profile_with_surface_point(
        tw_2m, orography + HEIGHT_2M_FIELDS, tw_levels, level_heights, above_ground
    )
    snowlmt = _uppermost_crossing(values, heights, SNOWLMT_THRESHOLD)
    melting_layer_top = _uppermost_crossing(values, heights, MELTING_LAYER_TOP_THRESHOLD)
    snowlmt_missing, melting_layer_top_missing = np.isnan(snowlmt), np.isnan(melting_layer_top)

    snowlmt = np.where(
        snowlmt_missing, _extrapolated_height(orography, SNOWLMT_THRESHOLD - tw_2m, TW_LAPSE_RATE), snowlmt
    )
    snowlmt = np.maximum(snowlmt, MIN_HEIGHT)

    melting_layer_top = np.where(
        snowlmt_missing & melting_layer_top_missing,
        _extrapolated_height(orography, MELTING_LAYER_TOP_THRESHOLD - tw_2m, TW_LAPSE_RATE),
        melting_layer_top,
    )
    melting_layer_top = np.where(
        ~snowlmt_missing & melting_layer_top_missing, orography + MELTING_LAYER_TOP_OFFSET, melting_layer_top
    )
    melting_layer_top = np.maximum(melting_layer_top, MIN_HEIGHT)
    melting_layer_top = np.where(
        melting_layer_top - snowlmt < MIN_MELTING_LAYER_DEPTH, snowlmt + MIN_MELTING_LAYER_DEPTH, melting_layer_top
    )
    return {"SNOWLMT": snowlmt, "MELTLAYER_TOP": melting_layer_top}


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
    (below-ground levels collapsed onto the 2 m point)."""
    surface_height = orography + HEIGHT_2M_FIELDS
    level_pressure = np.broadcast_to(pressure[:, None], t_levels.shape)
    q_2m = specific_humidity_from_dewpoint(td_2m, sp)
    p, height = _profile_with_surface_point(sp, surface_height, level_pressure, level_heights, above_ground)
    t, _ = _profile_with_surface_point(t_2m, surface_height, t_levels, level_heights, above_ground)
    q, _ = _profile_with_surface_point(q_2m, surface_height, q_levels, level_heights, above_ground)
    return p, height, t, q


def _mixed_layer_parcel(
    p: np.ndarray, height: np.ndarray, t: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pressure, temperature and specific humidity of the mixed-layer parcel:
    the pressure-weighted mean potential temperature and specific humidity of
    the lowest ``ML_DEPTH``, at the surface pressure."""
    sp = p[0]
    p_top = sp - ML_DEPTH
    p_layer, theta, q_layer = _insert_pressure_level(p_top, p, potential_temperature(t, p), q)
    dp = -np.diff(p_layer, axis=0)
    in_layer = p_layer[1:] >= p_top

    def layer_mean(x: np.ndarray) -> np.ndarray:
        return np.where(in_layer, 0.5 * (x[:-1] + x[1:]) * dp, 0.0).sum(axis=0) / ML_DEPTH

    return sp, temperature_from_potential_temperature(layer_mean(theta), sp), layer_mean(q_layer)


def _most_unstable_parcel(
    p: np.ndarray, height: np.ndarray, t: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pressure, temperature and specific humidity of the column point within
    ``MU_DEPTH`` of the surface with the highest equivalent potential
    temperature."""
    ept = np.where(p >= p[0] - MU_DEPTH, ept_from_specific_humidity(t, q, p, method=EPT_METHOD), -np.inf)
    index = np.argmax(ept, axis=0)
    return tuple(_values_at_level(x, index) for x in (p, t, q))


def _cape_cin(
    p: np.ndarray, height: np.ndarray, t: np.ndarray, q: np.ndarray, *, kind: Literal["ML", "MU"]
) -> tuple[np.ndarray, np.ndarray]:
    """CAPE and CIN (J/kg) of the mixed-layer (``ML``) or most-unstable
    (``MU``) parcel of the environment column."""
    start = _mixed_layer_parcel(p, height, t, q) if kind == "ML" else _most_unstable_parcel(p, height, t, q)
    return _cape_cin_of_parcel(p, height, t, q, *start)


def _cape_cin_of_parcel(
    p: np.ndarray,
    height: np.ndarray,
    t_env: np.ndarray,
    q_env: np.ndarray,
    p_start: np.ndarray,
    t_start: np.ndarray,
    q_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CAPE and CIN (J/kg) of a parcel starting at (``p_start``, ``t_start``,
    ``q_start``), a point of the environment column (``p``, ``height``,
    ``t_env``, ``q_env``).

    The parcel rises along the dry adiabat up to its LCL, then along the
    pseudo-adiabat of its equivalent potential temperature. The LCL is
    inserted into the column so the kink of the parcel curve is resolved.
    Buoyancy uses virtual temperature for parcel and environment.
    """
    td_start = np.minimum(dewpoint_from_specific_humidity(q_start, p_start), t_start)
    _, p_lcl = lcl(t_start, td_start, p_start)
    p_lcl = np.minimum(p_lcl, p_start)
    p, height, t_env, q_env = _insert_pressure_level(p_lcl, p, height, t_env, q_env)

    saturated = p < p_lcl
    t_parcel = temperature_on_dry_adiabat(p, t_start, p_start)
    # earthkit's pseudo-adiabat solver takes 1-D arrays; only saturated points need it.
    ept = np.broadcast_to(ept_from_dewpoint(t_start, td_start, p_start, method=EPT_METHOD), p.shape)
    t_parcel[saturated] = temperature_on_moist_adiabat(ept[saturated], p[saturated], ept_method=EPT_METHOD)
    q_parcel = np.where(saturated, saturation_specific_humidity(t_parcel, p), q_start)

    tv_env = virtual_temperature(t_env, q_env)
    buoyancy = constants.g * (virtual_temperature(t_parcel, q_parcel) - tv_env) / tv_env
    return _integrate_cape_cin(buoyancy, height, p, p_lcl, p_start)


def _insert_pressure_level(p_new: np.ndarray, p: np.ndarray, *profiles: np.ndarray) -> list[np.ndarray]:
    """Insert a point at pressure ``p_new`` into each bottom-up column.

    ``p`` and ``profiles`` have shape (levels, points), with ``p`` not
    increasing upwards. The profiles are interpolated linearly in log
    pressure. Returns ``p`` and the profiles with one more level.
    """
    log_p, log_p_new = np.log(p), np.log(p_new)
    above_index = np.clip((p >= p_new).sum(axis=0), 1, p.shape[0] - 1)  # first point above p_new
    below_index = above_index - 1
    log_p_below, log_p_above = _values_at_level(log_p, below_index), _values_at_level(log_p, above_index)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.clip(np.nan_to_num((log_p_below - log_p_new) / (log_p_below - log_p_above)), 0.0, 1.0)

    new_rows = [p_new]
    for x in profiles:
        x_below, x_above = _values_at_level(x, below_index), _values_at_level(x, above_index)
        new_rows.append(x_below + weight * (x_above - x_below))
    order = np.argsort(-np.vstack([p, p_new]), axis=0, kind="stable")
    return [np.take_along_axis(np.vstack([x, row]), order, axis=0) for x, row in zip((p, *profiles), new_rows)]


def _positive_area(b_bottom: np.ndarray, b_top: np.ndarray, dz: np.ndarray, zero_fraction: np.ndarray) -> np.ndarray:
    """Positive part of the area of each segment with buoyancy linear from
    ``b_bottom`` to ``b_top``, its zero crossing at ``zero_fraction`` of ``dz``."""
    return np.select(
        [(b_bottom >= 0) & (b_top >= 0), (b_bottom <= 0) & (b_top <= 0), b_bottom > 0],
        [0.5 * (b_bottom + b_top) * dz, 0.0, 0.5 * b_bottom * zero_fraction * dz],
        0.5 * b_top * (1.0 - zero_fraction) * dz,
    )


def _integrate_cape_cin(
    buoyancy: np.ndarray, height: np.ndarray, pressure: np.ndarray, p_lcl: np.ndarray, p_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CAPE and CIN (J/kg) from parcel buoyancy (m/s2) on a bottom-up column.

    Buoyancy is linear between the column points, so each segment's area is
    split exactly at its zero crossing. The LFC is the lowest point at or
    above the LCL (which must be a column point) where the parcel becomes
    buoyant, the EL the highest crossing back to non-positive buoyancy above
    it (the column top if the parcel is still buoyant there).

    CAPE is the area from the LFC to the EL, including any negative layers in
    between; CIN is the magnitude of the negative area below the LFC (>= 0).
    Positive buoyancy below the LCL counts for neither, and segments below
    the parcel's starting pressure ``p_start`` (a column point) are ignored.
    Without an LFC, CAPE is 0 and CIN missing.
    """
    b_bottom, b_top = buoyancy[:-1], buoyancy[1:]
    p_bottom = pressure[:-1]
    dz = np.where(p_bottom <= p_start, np.diff(height, axis=0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zero_fraction = np.clip(b_bottom / (b_bottom - b_top), 0.0, 1.0)
    area = 0.5 * (b_bottom + b_top) * dz
    positive = _positive_area(b_bottom, b_top, dz, zero_fraction)
    negative = area - positive

    n_segments = b_bottom.shape[0]
    segment = np.arange(n_segments)[:, None]
    # A segment starting at the LCL with positive buoyancy has its LFC at the LCL.
    lfc_segment = (p_bottom <= p_lcl) & (b_top > 0) & ((b_bottom <= 0) | (p_bottom == p_lcl))
    has_lfc = lfc_segment.any(axis=0)
    lfc_index = np.argmax(lfc_segment, axis=0)
    el_segment = (b_bottom > 0) & (b_top <= 0) & (segment > lfc_index)
    el_index = np.where(el_segment.any(axis=0), _last_true_index(el_segment), n_segments)

    cape_area = np.where(
        (segment == lfc_index) | (segment == el_index),
        positive,
        np.where((segment > lfc_index) & (segment < el_index), area, 0.0),
    )
    cape = np.where(has_lfc, np.maximum(cape_area.sum(axis=0), 0.0), 0.0)
    cin = np.where(has_lfc, -np.where(segment <= lfc_index, negative, 0.0).sum(axis=0), np.nan)
    return cape, cin


def _melting_layer_depth(snowlmt: np.ndarray, melting_layer_top: np.ndarray) -> np.ndarray:
    """MELTLAYER_TOP - SNOWLMT clamped to [``MELTING_LAYER_DEPTH_MIN``,
    ``MELTING_LAYER_DEPTH_MAX``]."""
    return np.clip(melting_layer_top - snowlmt, MELTING_LAYER_DEPTH_MIN, MELTING_LAYER_DEPTH_MAX)


def _precipitation_type(
    rate: np.ndarray,
    snowlmt: np.ndarray,
    melting_layer_top: np.ndarray,
    t_2m: np.ndarray,
    *,
    ground_temperature: np.ndarray,
    orography: np.ndarray,
) -> np.ndarray:
    """Precipitation type as INCA, without hail, in WMO code table 4.201:
    0 none, 1 rain, 3 freezing rain, 5 snow, 7 mixed; missing where ``rate``
    (mm/h) is.

    Ground at least the melting-layer depth above SNOWLMT gets snow, at least
    that depth below it rain, and mixed in between. Rain on ground below
    0.1 degC becomes freezing rain, unless the 2 m air is above 0.5 degC.
    """
    melting_layer_depth = _melting_layer_depth(snowlmt, melting_layer_top)
    wet = rate > PREC_RATE_MIN
    ground_above_snowlmt = orography - snowlmt
    prec_type = np.select(
        [
            ~wet,
            ground_above_snowlmt >= melting_layer_depth,
            ground_above_snowlmt > -melting_layer_depth,
        ],
        [PTYPE_NONE, PTYPE_SNOW, PTYPE_MIXED],
        PTYPE_RAIN,
    ).astype(np.float64)
    freezing = (prec_type == PTYPE_RAIN) & (ground_temperature < FREEZING_GROUND_T) & (t_2m <= FREEZING_MAX_AIR_T)
    prec_type[freezing] = PTYPE_FREEZING_RAIN
    prec_type[np.isnan(rate)] = np.nan
    return prec_type


def _snow_fraction(
    snowlmt: np.ndarray, melting_layer_top: np.ndarray, orography: np.ndarray, profile: str
) -> np.ndarray:
    """Fraction of precipitation falling as snow, as INCA: 0 at or below
    SNOWLMT, 1 at or above SNOWLMT + the melting-layer depth, ``linear`` or
    ``logistic`` in between."""
    x = np.clip((orography - snowlmt) / _melting_layer_depth(snowlmt, melting_layer_top), 0.0, 1.0)
    if profile == "linear":
        return x
    return 1.0 / (1.0 + np.exp(-(x - SNOW_FRACTION_LOGISTIC_MIDPOINT) * SNOW_FRACTION_LOGISTIC_STEEPNESS))
