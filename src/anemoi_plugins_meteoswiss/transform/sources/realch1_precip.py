# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from datetime import timedelta
from typing import Any

import earthkit.data as ekd
from anemoi.datasets.create.source import Source
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.grids import grid_registry

FDB_REQUEST = {
    "param": 500041,  # TOT_PREC
    "levtype": "sfc",
    "stream": "reanl",
    "class": "rd",
    "expver": "r001",
    "model": "icon-rea-l-ch1",
    "type": "cf",
}

ICON_GRID_PATH = "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/icon_grid_0001_R19B08_mch.nc"


class ReaLCh1Precip(Source):
    """REA-L-CH1 data source plugin."""

    emoji = "💽"

    def __init__(
        self,
        context,
        accumulation_period_hours: int = 1,
    ):
        """Initialise the FDB input.

        Parameters
        ----------
        context : dict
            The context.
        accumulation_period_hours : int, optional
            The accumulation period in hours (default: 1).
        """
        super().__init__(context)

        if accumulation_period_hours < 1 or accumulation_period_hours > 24:
            raise ValueError("accumulation_period_hours must be between 1 and 24.")

        self.accumulation_period_hours = accumulation_period_hours
        self.request = FDB_REQUEST.copy()
        self.grid = grid_registry.from_config({"icon": {"path": ICON_GRID_PATH}})

    def execute(self, dates: list[datetime]) -> ekd.FieldList:
        dates = _prepare_dates(dates, self.accumulation_period_hours)
        fl = _get_data_from_fdb(self.request, dates)
        fl = _accumulation_logic(fl, self.accumulation_period_hours)
        fl = new_fieldlist_from_list([new_field_from_grid(f, self.grid) for f in fl])
        return fl


def _as_fct_time_request(dt: datetime) -> str:
    """Defines the time-related keys for the FDB request."""
    out = {}

    # the accumulated precipitation for 00UTC of a given day, is actually
    # the difference between the "24UTC" (+24h step) and the 23UTC (+23h step)
    # of the previous day
    if dt.hour == 0 and dt.minute == 0:
        out["date"] = (dt - timedelta(days=1)).strftime("%Y%m%d")
        out["time"] = "0000"
        out["step"] = 24
        return out

    out["date"] = dt.strftime("%Y%m%d")
    out["time"] = "0000"
    out["step"] = int((dt - dt.replace(hour=0, minute=0)).total_seconds() // 3600)
    return out


def _prepare_dates(dates: list[datetime], accumulation_period_hours: int) -> list[datetime]:
    """Ensure unique and sorted dates."""
    if len(set(dates)) != len((dates := sorted(dates))):
        raise ValueError("dates must be unique and sorted.")

    # add previous datetimes for accumulation
    accum_steps = []
    for step in range(1, accumulation_period_hours + 1):
        accum_steps.append(dates[0] - timedelta(hours=step))
    dates = sorted(set(dates) | set(accum_steps))
    return dates


def _get_data_from_fdb(request: dict[str, Any], dates: list[datetime]):
    """Get data from FDB for the given request and dates."""

    # build requests
    requests = []
    for i, date in enumerate(dates):
        time_request = _as_fct_time_request(date)
        requests.append(request | time_request)

    # load data from FDB
    fields = []
    for request in requests:
        fields.extend(ekd.from_source("fdb", request).to_fieldlist(read_all=True))

    return new_fieldlist_from_list(fields)


def _accumulation_logic(fl: ekd.FieldList, accumulation_period_hours: int) -> ekd.FieldList:
    """Compute accumulation for the requested dates."""

    fields = list(fl)
    previous_day_accum = {f.get("time.base_datetime"): f for f in fields if f.get("time.step") == timedelta(hours=24)}
    accum_fl = []
    for i, field in enumerate(fields[accumulation_period_hours:]):
        previous_field = fields[i]

        # if previous field is from previous day
        # then the value is the sum of the current value and the difference between
        # the previous day 24h accumulation and the previous field value
        if previous_field.get("time.base_datetime") < field.get("time.base_datetime"):
            if (previous_accum := previous_day_accum.get(previous_field.get("time.base_datetime"))) is None:
                raise ValueError("Cannot compute accumulation, missing previous day 24h accumulation.")
            accum_field = field.values + (previous_accum.values - previous_field.values)
            start_step = _hours(previous_field.get("time.step")) - 24
            accum_fl.append(_new_accumulated_field(field, accum_field, start_step))
            continue

        # in the regular case, the accumulation is simply the difference
        # between the current and the previous field
        accum_field = field.values - previous_field.values
        accum_fl.append(_new_accumulated_field(field, accum_field, _hours(previous_field.get("time.step"))))
    return new_fieldlist_from_list(accum_fl)


def _hours(step: timedelta) -> int:
    """Convert a step to whole hours."""
    return int(step.total_seconds() // 3600)


def _new_accumulated_field(field: ekd.Field, values, start_step: int) -> ekd.Field:
    """Create a new field with the given values, accumulated from ``start_step`` (in hours) to the field's step.

    ``startStep`` is a raw GRIB key: setting it re-encodes the GRIB header (``stepRange``,
    ``lengthOfTimeRange``), while ``endStep``, the valid time and the values are kept.
    """
    return field.set({"values": values, "metadata.startStep": start_step})
