# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from datetime import datetime
from datetime import timedelta
from typing import Any

import earthkit.data as ekd
from anemoi.datasets.create.source import Source
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.grids import grid_registry

LOG = logging.getLogger(__name__)

FDB_REQUEST = {
    "param": 500164,  # VMAX_10M
    "levtype": "sfc",
    "stream": "reanl",
    "class": "rd",
    "expver": "r001",
    "model": "icon-rea-l-ch1",
    "type": "cf",
}

DT = 10

ICON_GRID_PATH = "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/icon_grid_0001_R19B08_mch.nc"


class ReaLCh1WindGust(Source):
    """REA-L-CH1 wind gust data source plugin."""

    def __init__(
        self,
        context,
        aggregation_period_minutes: int = 60,
    ):
        """Initialise the FDB input.

        Parameters
        ----------
        context : dict
            The context.
        aggregation_period_hours : int, optional
            The aggregation period in hours (default: 1).
        """
        super().__init__(context)

        if aggregation_period_minutes < 10 or aggregation_period_minutes > 360:
            raise ValueError("aggregation_period_minutes must be between 10 and 360.")

        self.aggregation_period_minutes = aggregation_period_minutes
        self.request = FDB_REQUEST.copy()
        self.grid = grid_registry.from_config({"icon": {"path": ICON_GRID_PATH}})

    def execute(self, dates: list[datetime]) -> ekd.FieldList:
        dates = _prepare_dates(dates, self.aggregation_period_minutes)
        if len(dates) != self.aggregation_period_minutes // DT:
            raise ValueError(
                f"Expected {self.aggregation_period_minutes // DT} dates needed for aggregation, got {len(dates)}."
            )

        fieldlist = _get_data_from_fdb(self.request, dates)
        field = _aggregate(fieldlist)
        fieldlist = new_fieldlist_from_list([new_field_from_grid(field, self.grid)])
        return fieldlist


def _as_fct_time_request(dt: datetime) -> str:
    """Defines the time-related keys for the FDB request."""
    out = {}

    # the derived wind gust for 00UTC of a given day, is actually
    # the value from the previous day's initialization +24h lead time
    if dt.hour == 0 and dt.minute == 0:
        out["date"] = (dt - timedelta(days=1)).strftime("%Y%m%d")
        out["time"] = "0000"
        out["step"] = 24
        return out

    out["date"] = dt.strftime("%Y%m%d")
    out["time"] = "0000"
    out["step"] = int((dt - dt.replace(hour=0, minute=0)).total_seconds() // 3600)
    return out


def _prepare_dates(
    dates: list[datetime], aggregation_period_minutes: int
) -> list[datetime]:
    """Ensure unique and sorted dates."""
    if len(set(dates)) != len((dates := sorted(dates))):
        raise ValueError("dates must be unique and sorted.")

    # add previous datetimes for aggregation
    required_dates = []
    for step in range(DT, aggregation_period_minutes, DT):
        required_dates.append(dates[0] - timedelta(minutes=step))
    dates = sorted(set(dates) | set(required_dates))
    return dates


def _get_data_from_fdb(request: dict[str, Any], dates: list[datetime]):
    """Get data from FDB for the given request and dates."""

    # build requests
    requests = []
    for i, date in enumerate(dates):
        time_request = _as_fct_time_request(date)
        requests.append(request | time_request)

    # load data from FDB
    fl = ekd.from_source("empty")
    for request in requests:
        fl += ekd.from_source("fdb", request, read_all=True)

    return fl


def _aggregate(fl: ekd.FieldList) -> ekd.FieldList:
    """Compute aggregation for the requested dates."""
    values = fl.values.max(axis=0)
    return fl[-1].clone(values=values, startStep=fl[0].metadata("startStep"))
