"""anemoi-transform filter that computes horizontal wind speed from its U and
V components, keeping the components in the output (unlike anemoi-transform's
built-in ``uv_to_ddff``, which replaces them with speed and direction).

Usage
-----
Register as an anemoi-inference output post-processor, once per output
stream that should contain wind speed::

    typed_variables:
      10si:
        mars: {param: SP_10M, levtype: sfc}

    output:
      tee:
      - grib:
          post_processors:
          - forward_transform_filter:
              wind-speed-from-components:
                u_component: 10u
                v_component: 10v
                wind_speed: 10si

Notes
-----
- ``u_component``/``v_component`` must be the *internal* anemoi variable
  names present in the state at the point this filter runs (e.g. ``10u``/
  ``10v``), not the final GRIB shortNames of the encoded file.
- ``wind_speed`` is the new field's internal state key. The model's
  checkpoint never produced this field, so anemoi-inference has no other way
  to know about it: it must also be declared in the run config's top-level
  ``typed_variables`` (with a ``mars`` block giving its GRIB identity), and
  that identity needs a matching entry in the output's GRIB template index
  (and, for streams that rewrite variable identities via ``modifiers``, a
  matching patch entry too) — otherwise GRIB encoding will fail to find a
  template/param for it.
"""

from collections.abc import Iterator

import earthkit.data as ekd
import numpy as np
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec


class WindSpeedFromComponents(MatchingFieldsFilter):
    """Compute horizontal wind speed from its U and V components."""

    MATCHING = MatchingSpec(
        select="param",
        forward=("u_component", "v_component"),
        return_inputs="all",
    )

    def __init__(
        self,
        *,
        u_component: str,
        v_component: str,
        wind_speed: str,
    ) -> None:
        """Initialise the filter.

        Parameters
        ----------
        u_component:
            Name of the U wind component field to match (internal/anemoi name).
        v_component:
            Name of the V wind component field to match (internal/anemoi name).
        wind_speed:
            Name to give the new wind speed field (both its internal state
            key and its GRIB param/shortName, unless overridden downstream
            e.g. via a ``modifiers`` patch).
        """
        self.u_component = u_component
        self.v_component = v_component
        self.wind_speed = wind_speed
        super().__init__()

    def forward_transform(
        self, u_component: ekd.Field, v_component: ekd.Field
    ) -> Iterator[ekd.Field]:
        speed = np.sqrt(u_component.to_numpy() ** 2 + v_component.to_numpy() ** 2)
        yield self.new_field_from_numpy(
            speed,
            template=u_component,
            name=self.wind_speed,
            param=self.wind_speed,
            shortName=self.wind_speed,
        )
