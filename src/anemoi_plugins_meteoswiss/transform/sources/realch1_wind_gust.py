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
import numpy as np
from anemoi.datasets.create.source import Source
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_field_with_metadata
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
        out = []
        for date in dates:
            aggregation_dates = _prepare_dates(date, self.aggregation_period_minutes)
            if len(aggregation_dates) != self.aggregation_period_minutes // DT:
                raise ValueError(
                    f"Expected {self.aggregation_period_minutes // DT} dates needed for aggregation, got {len(dates)}."
                )

            fieldlist = _get_data_from_fdb(self.request, aggregation_dates)
            field = _aggregate(fieldlist)
            out.append(new_field_from_grid(field, self.grid))
        return new_fieldlist_from_list(out)


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

    # TODO: the step is floored to whole hours, so the 10-min dates of an aggregation window all
    # map to the 10-min maximum ending at the full hour. A 60 min window at HH:00 is then the max of
    # the values ending at (HH-1):00 and HH:00 (period 70 min; at 01 UTC only 10 min, since step 0
    # has no data). Kept as is for consistency with the released datasets; fixing it changes the data.
    out["date"] = dt.strftime("%Y%m%d")
    out["time"] = "0000"
    out["step"] = int((dt - dt.replace(hour=0, minute=0)).total_seconds() // 3600)
    return out


def _prepare_dates(date: datetime, aggregation_period_minutes: int) -> list[datetime]:
    """Ensure unique and sorted dates."""
    # add previous datetimes for aggregation
    required_dates = []
    for step in range(DT, aggregation_period_minutes, DT):
        required_dates.append(date - timedelta(minutes=step))
    dates = sorted(set([date] + required_dates))
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


def _aggregate(fl: ekd.FieldList) -> ekd.Field:
    """Compute aggregation for the requested dates."""
    values = np.max([f.values for f in fl], axis=0)
    # startStep is a raw GRIB key: setting it re-encodes the GRIB header (stepRange,
    # lengthOfTimeRange), while endStep, the valid time and the values are kept.
    field = fl[-1].set({"values": values, "metadata.startStep": fl[0].metadata("startStep")})
    # remove the level information (VMAX_10M is on heightAboveGround 10 m), so the variable is named VMAX_10M
    return new_field_with_metadata(field, levelist=None)
