# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import timedelta
from collections.abc import Callable

from earthkit.data.core.fieldlist import Field

from anemoi.transform.filter import SingleFieldFilter

STEP_UNITS = {0: "m", 1: "h"} # GRIB Codes table 4.4

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

    def forward_transform(self, field: Field) -> Field:
        """Multiply a time-averaged flux by its aggregation period."""
        if not (steptype := field.metadata("stepType")) == "avg":
            raise ValueError(f"Expected a time-averaged field, got stepType={steptype}")
        start, end, unit = field.metadata("startStep", "endStep", "stepUnits")
        seconds = _aggregation_period_seconds(end - start, unit)
        return field.clone(values=field.values * seconds, stepType="accum")
    

    def backward_transform(self, field: Field) -> Field:
        """Divide a cumulative quantity by its aggregation period."""
        if not (steptype := field.metadata("stepType")) == "accum":
            raise ValueError(f"Expected a cumulative field, got stepType={steptype}")
        start, end, unit = field.metadata("startStep", "endStep", "stepUnits")
        seconds = _aggregation_period_seconds(end - start, unit)
        return field.clone(values=field.values / seconds, stepType="avg")


CumulativeQuantityToAverageFlux = AverageFluxToCumulativeQuantity.reversed