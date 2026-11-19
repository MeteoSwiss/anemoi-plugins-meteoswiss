# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

import earthkit.data as ekd
import numpy as np
from anemoi.datasets.create.source import Source
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.sources.array_list import ArrayField

from .thx_lightning_core import AGGREGATIONS
from .thx_lightning_core import DEFAULT_NETWORK_BBOX
from .thx_lightning_core import IconGrid
from .thx_lightning_core import IncaGrid
from .thx_lightning_core import StrikeArchive
from .thx_lightning_core import window_counts

if TYPE_CHECKING:
    from anemoi.datasets.create.arguments import ValidDates

LOG = logging.getLogger(__name__)

DEFAULT_ARCHIVE = "/store_new/mch/msclim/hamann/database/lightning/THX"

# GRIB code table 4.10 and GRIB2 stepType, as read by anemoi-datasets to fill
# the ``process`` / ``period`` entries of ``variables_metadata``.
_STATISTICS = {
    "accumulation": dict(typeOfStatisticalProcessing=1, stepType="accum"),
    "maximum": dict(typeOfStatisticalProcessing=2, stepType="max"),
    "average": dict(typeOfStatisticalProcessing=0, stepType="avg"),
}
_NAME_SUFFIX = {"maximum": "MAX", "average": "AVG"}


class _LightningField(ArrayField):
    """Array field that also answers the ``mars`` metadata namespace.

    anemoi-datasets stores ``metadata(namespace="mars")`` as the variable's
    ``mars`` entry; plain array fields return an empty dict there.
    """

    def __init__(
        self, array: np.ndarray, metadata: dict[str, Any], mars: dict[str, Any]
    ):
        super().__init__(array, metadata)
        self._mars = mars

    def metadata(self, *args, **kwargs):
        if not args and kwargs.get("namespace") == "mars":
            return dict(self._mars)
        return super().metadata(*args, **kwargs)

    # earthkit's version always raises NotImplementedError for array fields
    # (typed NoReturn), which anemoi-datasets logs as an error when it builds
    # ``data_request``; returning None is the intended replacement.
    @property
    def mars_grid(self) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """No MARS grid specification for unstructured or projected grids."""
        return None


def variable_name(
    radius_km: float, period: timedelta, aggregation: str, sub_period: timedelta | None
) -> str:
    """``LC_<radius>KM_<period>[_<MAX|AVG><sub_period>]``, e.g. ``LC_4KM_1H_MAX10M``."""
    name = f"LC_{radius_km:g}KM_{frequency_to_string(period).upper()}"
    if aggregation == "accumulation":
        return name
    if sub_period is None:
        raise ValueError(f"aggregation {aggregation!r} requires sub_period")
    return (
        f"{name}_{_NAME_SUFFIX[aggregation]}{frequency_to_string(sub_period).upper()}"
    )


class ThxLightning(Source):
    """Lightning strike counts from the raw MeteoSwiss THX archive.

    Reads the daily ``THX{YY}{JJJ}0000.prd[.zip]`` files and counts, for every
    grid cell, the strikes within ``radius_km`` of the cell centre during the
    window ending at each dataset date. No intermediate files are written.

    Parameters
    ----------
    context :
        Anemoi build context (injected by the framework).
    grid_definition : dict
        ``{"icon": {"path": <ICON grid NetCDF>}}`` (as in ``assign-grid``) or
        ``{"inca": {}}`` for the INCA/RADAR 710 x 640 LV95 grid.
    period : str
        Length of the window ending at each date, e.g. ``"10m"``, ``"1h"``.
    radius_km : float or list of float
        Neighbourhood radii in km; one variable per radius. ``0`` counts the
        nearest cell only.
    aggregation : str
        ``"accumulation"`` (default): strikes in the window.
        ``"maximum"`` / ``"average"``: max / mean of the ``sub_period``
        accumulations tiling the window.
    sub_period : str, optional
        Required for ``maximum`` / ``average``; must divide ``period``.
    discharge_type : list of str
        ``["CG", "IC"]`` (default), ``["CG"]`` or ``["IC"]``.
    archive_path : str
        Directory of the THX archive.
    network_bbox : list of float
        ``[lat_min, lat_max, lon_min, lon_max]`` of the THX network. Cells
        outside are set to NaN (no detection is not the same as no lightning).

    Notes
    -----
    A missing or unreadable archive day raises an error; list such days under
    ``dates: missing:`` in the recipe.
    """

    emoji = "⚡"

    def __init__(
        self,
        context,
        grid_definition: dict,
        period: str,
        radius_km: float | Sequence[float],
        aggregation: str = "accumulation",
        sub_period: str | None = None,
        discharge_type: Sequence[str] = ("CG", "IC"),
        archive_path: str = DEFAULT_ARCHIVE,
        network_bbox: Sequence[float] = DEFAULT_NETWORK_BBOX,
    ):
        super().__init__(context)

        self.period = frequency_to_timedelta(period)
        radii = radius_km if isinstance(radius_km, Sequence) else [radius_km]
        self.radii_km = [float(r) for r in radii]
        if len(set(self.radii_km)) != len(self.radii_km) or any(
            r < 0 for r in self.radii_km
        ):
            raise ValueError(
                f"radius_km must be distinct non-negative values, got {radius_km}"
            )

        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
            )
        self.aggregation = aggregation
        if aggregation == "accumulation":
            if sub_period is not None:
                raise ValueError(
                    "sub_period is only used with aggregation 'maximum' or 'average'"
                )
            self.sub_period = None
        else:
            if sub_period is None:
                raise ValueError(f"aggregation {aggregation!r} requires sub_period")
            self.sub_period = frequency_to_timedelta(sub_period)
            if self.period % self.sub_period or self.sub_period >= self.period:
                raise ValueError(
                    f"sub_period ({sub_period}) must divide period ({period}) and be shorter"
                )

        if not isinstance(grid_definition, dict) or len(grid_definition) != 1:
            raise ValueError(
                f"grid_definition must be {{'icon': {{'path': ...}}}} or {{'inca': {{}}}}, got {grid_definition}"
            )
        ((grid_type, grid_args),) = grid_definition.items()
        if grid_type == "icon":
            self.grid = IconGrid.from_file(grid_args["path"])
        elif grid_type == "inca":
            self.grid = IncaGrid()
        else:
            raise ValueError(f"Unsupported grid {grid_type!r}; use 'icon' or 'inca'")

        self.archive = StrikeArchive(archive_path, discharge_type, self.grid)
        self.outside_network = ~self.grid.network_mask(network_bbox)
        self.names = [
            variable_name(r, self.period, self.aggregation, self.sub_period)
            for r in self.radii_km
        ]

    def execute_valid_dates(self, argument: ValidDates) -> ekd.FieldList:
        """Fields for each valid date; called by ``Source.execute`` (anemoi-datasets >= 0.5.38)."""
        statistics = _STATISTICS[self.aggregation]
        step = frequency_to_string(self.period)
        out = []
        valid: datetime
        for valid in argument.dates:
            self.context.trace(self.emoji, f"{valid} {self.names}")
            counts = window_counts(
                self.archive,
                valid,
                self.period,
                self.radii_km,
                self.aggregation,
                self.sub_period,
            ).astype(np.float32)
            counts[:, self.outside_network] = np.nan

            start = valid - self.period
            for name, values in zip(self.names, counts):
                mars = dict(
                    date=int(start.strftime("%Y%m%d")),
                    time=int(start.strftime("%H%M")),
                    step=step,
                    levtype="sfc",
                    param=name,
                )
                metadata = dict(
                    mars,
                    latitudes=self.grid.latitudes,
                    longitudes=self.grid.longitudes,
                    valid_datetime=valid.isoformat(),
                    startStep=timedelta(0),
                    endStep=self.period,
                    units="1",
                    **statistics,
                )
                out.append(_LightningField(values, metadata, mars))
        return new_fieldlist_from_list(out)
