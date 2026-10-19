# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import timedelta
from typing import Any

import earthkit.data as ekd
import numpy as np
from anemoi.transform.filter import SingleFieldFilter

STEP_UNITS = {0: "m", 1: "h"}  # GRIB Codes table 4.4
STEP_SUFFIX_SECONDS = {"s": 1, "m": 60, "h": 3600}


def _aggregation_period_seconds(step: int, unit: int) -> float:
    """Convert a GRIB step and unit to a timedelta in seconds."""
    if unit not in STEP_UNITS:
        raise ValueError(f"Unsupported step unit: {unit}")
    if STEP_UNITS[unit] == "m":
        return timedelta(minutes=step).total_seconds()
    elif STEP_UNITS[unit] == "h":
        return timedelta(hours=step).total_seconds()
    else:
        raise ValueError(f"Unsupported step unit: {unit}")


def _time_key(field: ekd.Field, key: str) -> Any:
    """Return a raw GRIB time key (e.g. ``stepType``) of a field.

    Falls back to the ``metadata.time`` collection, which keeps the raw keys when a
    filter, flavour or grid has masked them.
    """
    value = field.get(f"metadata.{key}", default=None)
    if value is None:
        value = (field.get(collections="metadata.time") or {}).get(key)
    if value is None:
        raise KeyError(f"Cannot find GRIB key '{key}' in field {field}")
    return value


def _step_seconds(step: int | str, unit: int) -> float:
    """Convert a GRIB step to seconds.

    ecCodes returns steps in units other than hours with a suffix (e.g. "710m").
    """
    if isinstance(step, str) and step[-1:] in STEP_SUFFIX_SECONDS:
        return int(step[:-1]) * STEP_SUFFIX_SECONDS[step[-1]]
    return _aggregation_period_seconds(int(step), unit)


def _period_seconds(field: ekd.Field) -> float:
    """Length of the statistical processing period of a field, in seconds."""
    unit = _time_key(field, "stepUnits")
    return _step_seconds(_time_key(field, "endStep"), unit) - _step_seconds(_time_key(field, "startStep"), unit)


def _new_field(field: ekd.Field, values: np.ndarray, step_type: str) -> ekd.Field:
    """Create a new field with the given values and GRIB ``stepType``.

    ``stepType`` is a raw GRIB key: setting it re-encodes the GRIB header (and with it the
    parameter, e.g. ASOD_S -> DSWRF), while the values are kept as given. earthkit-data can
    silently drop the key, e.g. on a field without GRIB metadata or whose parameter was
    changed before (the parameter is re-encoded on top), hence the check.
    """
    new_field = field.set({"values": values, "metadata.stepType": step_type})
    if _time_key(new_field, "stepType") != step_type:
        raise ValueError(f"Cannot set GRIB key stepType={step_type} on field {field}")
    return new_field


class AverageFluxToCumulativeQuantity(SingleFieldFilter):
    """A filter to multiply a time-averaged flux by its aggregation period.

    This filter converts a time-averaged flux (e.g., W m^-2) into a cumulative quantity
    (e.g., J m^-2) by multiplying it by the aggregation period in seconds.

    Examples
    --------

    .. code-block:: yaml

      input:
        pipe:
            - source: # loads some time-averaged flux variables
            - average-flux-to-cumulative-quantity: {}
    """

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        """Multiply a time-averaged flux by its aggregation period."""
        if not (steptype := _time_key(field, "stepType")) == "avg":
            raise ValueError(f"Expected a time-averaged field, got stepType={steptype}")
        return _new_field(field, field.values * _period_seconds(field), "accum")

    def backward_transform(self, field: ekd.Field) -> ekd.Field:
        """Divide a cumulative quantity by its aggregation period."""
        if not (steptype := _time_key(field, "stepType")) == "accum":
            raise ValueError(f"Expected a cumulative field, got stepType={steptype}")
        return _new_field(field, field.values / _period_seconds(field), "avg")


CumulativeQuantityToAverageFlux = AverageFluxToCumulativeQuantity.reversed
