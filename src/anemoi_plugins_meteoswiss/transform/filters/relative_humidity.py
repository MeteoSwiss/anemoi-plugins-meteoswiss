"""anemoi-transform filter that computes 2-metre relative humidity from
temperature and dewpoint, keeping the inputs in the output. Uses
earthkit-meteo's ``relative_humidity_from_dewpoint`` for the underlying
thermodynamics (saturation vapour pressure over water).

Usage
-----
Register as an anemoi-inference output post-processor, once per output
stream that should contain relative humidity::

    typed_variables:
      relhum_2m:
        mars: {param: RELHUM_2M, levtype: sfc}

    output:
      tee:
      - grib:
          post_processors:
          - forward_transform_filter:
              relative-humidity-from-dewpoint:
                temperature: 2t
                dewpoint: 2d
                relative_humidity: relhum_2m

Notes
-----
- ``temperature``/``dewpoint`` must be the *internal* anemoi variable names
  present in the state at the point this filter runs (e.g. ``2t``/``2d``),
  not the final GRIB shortNames of the encoded file.
- ``relative_humidity`` is the new field's internal state key. The model's
  checkpoint never produced this field, so anemoi-inference has no other way
  to know about it: it must also be declared in the run config's top-level
  ``typed_variables`` (with a ``mars`` block giving its GRIB identity), and
  that identity needs a matching entry in the output's GRIB template index
  (and, for streams that rewrite variable identities via ``modifiers``, a
  matching patch entry too) — otherwise GRIB encoding will fail to find a
  template/param for it.
- Both inputs must be in Kelvin (anemoi's internal convention for
  temperature/dewpoint fields). The result is in percent.
"""

from collections.abc import Iterator

import earthkit.data as ekd
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec
from earthkit.meteo.thermo import relative_humidity_from_dewpoint


class RelativeHumidityFromDewpoint(MatchingFieldsFilter):
    """Compute relative humidity from temperature and dewpoint."""

    MATCHING = MatchingSpec(
        select="param",
        forward=("temperature", "dewpoint"),
        return_inputs="all",
    )

    def __init__(
        self,
        *,
        temperature: str,
        dewpoint: str,
        relative_humidity: str,
    ) -> None:
        """Initialise the filter.

        Parameters
        ----------
        temperature:
            Name of the temperature field to match (internal/anemoi name).
        dewpoint:
            Name of the dewpoint field to match (internal/anemoi name).
        relative_humidity:
            Name to give the new relative humidity field (both its internal
            state key and its GRIB param/shortName, unless overridden
            downstream e.g. via a ``modifiers`` patch).
        """
        self.temperature = temperature
        self.dewpoint = dewpoint
        self.relative_humidity = relative_humidity
        super().__init__()

    def forward_transform(
        self, temperature: ekd.Field, dewpoint: ekd.Field
    ) -> Iterator[ekd.Field]:
        rh = relative_humidity_from_dewpoint(
            temperature.to_numpy(), dewpoint.to_numpy()
        )
        yield self.new_field_from_numpy(
            rh,
            template=temperature,
            name=self.relative_humidity,
            param=self.relative_humidity,
            shortName=self.relative_humidity,
        )
