# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Statistics of REA-L-CH1 surface fields over a period (e.g. the 6 h maximum wind gust).

The reanalysis archives its surface fields every 10 minutes: instantaneous fields (T_2M,
U_10M, CAPE_ML, ...) and statistics over the previous 10 minutes (VMAX_10M, LPI_MAX,
DHAIL_MX: maximum; U_10M_AV, V_10M_AV: mean). For every dataset date this source reads
the 10-minute fields in the period ending at that date and reduces them with the requested
statistic, e.g. the 6 h maximum gust is the maximum of the 36 VMAX_10M fields valid in
(date - 6 h, date].

Fields are read from the 00 UTC run of their day; the fields valid at 00 UTC come from
step 24 of the previous day's run (there is no step 0). The output field keeps the name of
the input parameter, rename it in the recipe (e.g. VMAX_10M -> VMAX_10M_6H).
"""

import logging
from collections import defaultdict
from datetime import date as date_type
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable

import earthkit.data as ekd
import numpy as np
from anemoi.datasets.create.source import Source
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.grids import grid_registry

LOG = logging.getLogger(__name__)

FDB_REQUEST = {
    "levtype": "sfc",
    "stream": "reanl",
    "class": "rd",
    "expver": "r001",
    "model": "icon-rea-l-ch1",
    "type": "cf",
}

# minutes between two archived fields
DT = 10

ICON_GRID_PATH = "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/icon_grid_0001_R19B08_mch.nc"

# surface parameters archived every 10 minutes that can be reduced over a period
# (instantaneous fields and statistics over the previous 10 minutes; accumulations and
# averages since the start of the run, such as TOT_PREC or ASWDIR_S, are not in the list)
PARAM_IDS = {
    "PS": 500000,
    "PMSL": 500002,
    "T_G": 500010,
    "T_2M": 500011,
    "TD_2M": 500017,
    "U_10M": 500027,
    "V_10M": 500029,
    "TQI": 500040,
    "H_SNOW": 500045,
    "CLCT": 500046,
    "TQC": 500051,
    "ALB_RAD": 500056,
    "TQR": 500104,
    "TQS": 500105,
    "TQG": 500107,
    "TWATER": 500108,
    "HZEROCL": 500127,
    "SNOWLMT": 500128,
    "CAPE_MU": 500151,
    "CIN_MU": 500152,
    "CAPE_ML": 500153,
    "CIN_ML": 500154,
    "VMAX_10M": 500164,
    "DBZ_CMAX": 500175,
    "HPBL": 502757,
    "LPI": 503142,
    "LFC_ML": 503187,
    "SLI": 503204,
    "TOT_PR": 503303,
    "LPI_MAX": 503348,
    "DHAIL_MX": 503557,
    "U_10M_AV": 503743,
    "V_10M_AV": 503744,
}

STATISTICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "max": lambda values: np.max(values, axis=0),
    "min": lambda values: np.min(values, axis=0),
    "mean": lambda values: np.mean(values, axis=0),
}


class ReaLCh1TimeStatistic(Source):
    """Statistic of REA-L-CH1 surface fields over the period ending at each date."""

    def __init__(
        self,
        context,
        param: str | int | list[str | int],
        statistic: str = "max",
        period_hours: int = 1,
    ):
        """Initialise the source.

        Parameters
        ----------
        context : dict
            The context.
        param : str | int | list[str | int]
            Parameter(s) to reduce, as short names (see PARAM_IDS) or paramIds.
        statistic : str, optional
            One of "max", "min", "mean" (default: "max").
        period_hours : int, optional
            Length of the period ending at each date, in hours (default: 1).
        """
        super().__init__(context)

        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {sorted(STATISTICS)}, got {statistic!r}.")
        if not 1 <= period_hours <= 24:
            raise ValueError(f"period_hours must be between 1 and 24, got {period_hours}.")

        self.statistic = statistic
        self.period = timedelta(hours=period_hours)
        self.param_ids = [_param_id(p) for p in (param if isinstance(param, list) else [param])]
        self.grid = grid_registry.from_config({"icon": {"path": ICON_GRID_PATH}})

    def execute(self, dates: list[datetime]) -> ekd.FieldList:
        dates = [_naive(d) for d in dates]
        windows = {d: window(d, self.period) for d in dates}
        times = sorted({t for w in windows.values() for t in w})
        fields = _retrieve(self.param_ids, times)

        out = []
        for d in dates:
            for param_id in self.param_ids:
                try:
                    window_fields = [fields[(param_id, t)] for t in windows[d]]
                except KeyError as e:
                    raise ValueError(f"Missing field for param {param_id} valid at {e.args[0][1]}") from None
                values = STATISTICS[self.statistic](np.stack([f.values for f in window_fields]))
                # the last field of the window is valid at the dataset date
                field = window_fields[-1].clone(values=values, level=None, levelist=None)
                out.append(new_field_from_grid(field, self.grid))
        return new_fieldlist_from_list(out)


class ReaLCh1WindGust(ReaLCh1TimeStatistic):
    """Maximum 10 m wind gust (VMAX_10M) over the period ending at each date."""

    def __init__(self, context, aggregation_period_minutes: int = 60):
        """Initialise the source.

        Parameters
        ----------
        context : dict
            The context.
        aggregation_period_minutes : int, optional
            The aggregation period in minutes, a multiple of 60 (default: 60).
        """
        if aggregation_period_minutes % 60:
            raise ValueError(f"aggregation_period_minutes must be a multiple of 60, got {aggregation_period_minutes}.")
        super().__init__(context, param="VMAX_10M", statistic="max", period_hours=aggregation_period_minutes // 60)


def _param_id(param: str | int) -> int:
    if isinstance(param, int):
        return param
    try:
        return PARAM_IDS[param]
    except KeyError:
        raise ValueError(f"Unknown parameter {param!r}, use a paramId or one of {sorted(PARAM_IDS)}.") from None


def _naive(d: datetime) -> datetime:
    return d.replace(tzinfo=None) if d.tzinfo is not None else d


def window(date: datetime, period: timedelta) -> list[datetime]:
    """Valid times of the archived fields in (date - period, date]."""
    minutes = int(period.total_seconds() // 60)
    if minutes <= 0 or minutes % DT:
        raise ValueError(f"The period must be a positive multiple of {DT} minutes, got {period}.")
    return [date - timedelta(minutes=m) for m in range(minutes - DT, -1, -DT)]


def run_and_step(valid: datetime) -> tuple[date_type, int]:
    """Run date (00 UTC) and step in minutes of the field valid at `valid`.

    Midnight belongs to the previous day's run (step 24 h), there is no step 0.
    """
    run_date = (valid - timedelta(minutes=1)).date()
    step = int((valid - datetime.combine(run_date, datetime.min.time())).total_seconds() // 60)
    return run_date, step


def format_step(minutes: int) -> str:
    """Format a step in minutes as the FDB step value (60 -> '1', 70 -> '1h10m', 10 -> '10m')."""
    hours, minutes = divmod(minutes, 60)
    if not minutes:
        return str(hours)
    return f"{hours}h{minutes}m" if hours else f"{minutes}m"


def requests(param_ids: list[int], times: list[datetime]) -> list[dict[str, Any]]:
    """One FDB request per run date covering all the fields valid at `times`."""
    steps_by_run = defaultdict(set)
    for t in times:
        run_date, step = run_and_step(t)
        steps_by_run[run_date].add(step)
    return [
        FDB_REQUEST
        | {
            "date": run_date.strftime("%Y%m%d"),
            "time": "0000",
            "param": list(param_ids),
            "step": [format_step(s) for s in sorted(steps)],
        }
        for run_date, steps in sorted(steps_by_run.items())
    ]


def _retrieve(param_ids: list[int], times: list[datetime]) -> dict[tuple[int, datetime], ekd.Field]:
    """Fields keyed by (paramId, valid time)."""
    fields = {}
    for request in requests(param_ids, times):
        LOG.debug("FDB request %s", request)
        for f in ekd.from_source("fdb", request, read_all=True):
            fields[(f.metadata("paramId"), f.datetime()["valid_time"])] = f
    return fields
