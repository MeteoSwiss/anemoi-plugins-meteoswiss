"""Computes SP_10M (wind speed), DD_10M (wind direction), RELHUM_2M from
U/V and temperature/dewpoint, in a single pass. Fields are added to Varda
inference output.

Usage::

    typed_variables:
      SP_10M:
        mars: {param: SP_10M, levtype: sfc}
      DD_10M:
        mars: {param: DD_10M, levtype: sfc}
      RELHUM_2M:
        mars: {param: RELHUM_2M, levtype: sfc}

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
                variables: [SP_10M, DD_10M, RELHUM_2M]

Notes
-----
- Each requested output needs a ``typed_variables`` + GRIB template entry
  of its own, since the checkpoint never produced it.
- Wind direction: meteorological convention (FROM, clockwise from North).
- Temperature/dewpoint in Kelvin; RH returned in percent.
- Must be the last transform filter in its post_processors list: chaining
  separate filters fails because ``TransformFilter.process()`` re-wraps
  the whole state each time, and a field invented by an earlier filter
  isn't in the checkpoint's variable catalog, so the next wrap raises
  KeyError. One filter computing everything avoids that.
"""

import logging
from collections.abc import Iterator

import earthkit.data as ekd
import numpy as np
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec
from earthkit.meteo.thermo import relative_humidity_from_dewpoint

LOG = logging.getLogger(__name__)

_WIND = ("u_component", "v_component")
_HUMIDITY = ("temperature", "dewpoint")
_SUPPORTED_VARIABLES: dict[str, tuple[str, ...]] = {
    "SP_10M": _WIND,
    "DD_10M": _WIND,
    "RELHUM_2M": _HUMIDITY,
}


class SurfaceDiagnostics(MatchingFieldsFilter):
    """Compute SP_10M/DD_10M/RELHUM_2M from U/V and temperature/dewpoint."""

    MATCHING = MatchingSpec(
        select="param",
        forward=("u_component", "v_component", "temperature", "dewpoint"),
        return_inputs="all",
    )

    def __init__(
        self,
        *,
        u_component: str,
        v_component: str,
        temperature: str,
        dewpoint: str,
        variables: str | list[str],
    ) -> None:
        """Field names are internal anemoi names. ``variables`` selects one
        or more of ``SP_10M``, ``DD_10M``, ``RELHUM_2M`` to compute."""
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
        self.variables = variables
        LOG.info(
            "surface-diagnostics: will add %s to the output, "
            "computed from u_component=%r, v_component=%r, temperature=%r, dewpoint=%r",
            self.variables,
            u_component,
            v_component,
            temperature,
            dewpoint,
        )
        super().__init__()

    def forward_transform(
        self,
        u_component: ekd.Field,
        v_component: ekd.Field,
        temperature: ekd.Field,
        dewpoint: ekd.Field,
    ) -> Iterator[ekd.Field]:
        if "SP_10M" in self.variables or "DD_10M" in self.variables:
            u = u_component.to_numpy()
            v = v_component.to_numpy()

        if "SP_10M" in self.variables:
            speed = np.sqrt(u**2 + v**2)
            yield self.new_field_from_numpy(
                speed,
                template=u_component,
                name="SP_10M",
                param="SP_10M",
                shortName="SP_10M",
            )

        if "DD_10M" in self.variables:
            direction = np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)
            yield self.new_field_from_numpy(
                direction,
                template=u_component,
                name="DD_10M",
                param="DD_10M",
                shortName="DD_10M",
            )

        if "RELHUM_2M" in self.variables:
            rh = relative_humidity_from_dewpoint(temperature.to_numpy(), dewpoint.to_numpy())
            yield self.new_field_from_numpy(
                rh,
                template=temperature,
                name="RELHUM_2M",
                param="RELHUM_2M",
                shortName="RELHUM_2M",
            )
