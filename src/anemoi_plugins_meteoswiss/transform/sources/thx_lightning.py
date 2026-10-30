# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import earthkit.data as ekd
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources.xarray_support import XarrayFieldList
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.grids import grid_registry

LOG = logging.getLogger(__name__)


class ThxLightning(Source):
    """Source plugin that reads preprocessed THX lightning NetCDF files.

    The NetCDF files are produced by the ``lightningpre_anemoi_dataset``
    package (``lightning-grid`` CLI), which grids raw THX lightning strike
    data onto a target grid at configurable time intervals.  One file per
    calendar day is expected at::

        {data_dir}/lightning_YYYYMMDD.nc

    Two target grids are supported:

    **ICON-CH1 native (unstructured)**
        Each file has dimensions ``(time, ncells)`` with auxiliary coordinate
        variables ``clat(ncells)`` / ``clon(ncells)`` (CF standard names
        ``latitude`` / ``longitude``).  Set ``icon_grid_path`` to assign the
        official ICON grid object (required for anemoi to identify the grid).

    **INCA/RADAR Swiss Grid LV95 (regular)**
        Each file has dimensions ``(time, radius, p_i, p_j)`` with 2-D coordinate
        arrays ``lat(p_i, p_j)`` / ``lon(p_i, p_j)`` in WGS84.  Leave
        ``icon_grid_path`` unset — the coordinates embedded in the NetCDF
        are used directly.

    Parameters
    ----------
    context :
        Anemoi build context (injected by the framework).
    data_dir : str
        Shared directory produced by ``lightning-grid``
        (``/store_new/mch/msclim/lightning_gridded``).
    grid_type : str
        ``"icon"`` or ``"inca"``. Determines the filename pattern:
        ``lightning_icon_YYYYMMDD.nc`` or ``lightning_inca_YYYYMMDD.nc``.
    radius_km : float
        Accumulation radius in km to select from the ``radius`` dimension of
        the preprocessed NetCDF.  Must match one of the values used during
        preprocessing (e.g. 4, 8, 16).
    icon_grid_path : str, optional
        Path to the ICON-CH1 native-grid NetCDF used by
        ``anemoi.transform.grids.grid_registry``.  Required for the ICON
        grid variant; omit for the INCA/RADAR variant.
    """

    emoji = "⚡"

    def __init__(
        self,
        context,
        data_dir: str,
        grid_type: str,
        radius_km: float,
        icon_grid_path: str | None = None,
    ):
        super().__init__(context)
        self.data_dir = Path(data_dir)
        if grid_type not in ("icon", "inca"):
            raise ValueError(f"grid_type must be 'icon' or 'inca', got {grid_type!r}")
        self.grid_type = grid_type
        self.radius_km = float(radius_km)
        self.grid = (
            grid_registry.from_config({"icon": {"path": icon_grid_path}})
            if icon_grid_path is not None
            else None
        )

    def execute(self, dates: list[datetime]) -> ekd.FieldList:
        import numpy as np
        import xarray as xr
        from collections import defaultdict

        out = []

        # Group requested dates by calendar day to open each file at most once.
        by_day: dict = defaultdict(list)
        for date in dates:
            by_day[date.date()].append(date)

        for day in sorted(by_day):
            nc_path = self.data_dir / f"lightning_{self.grid_type}_{day.strftime('%Y%m%d')}.nc"
            self.context.trace(self.emoji, str(nc_path))

            ds = xr.open_dataset(nc_path)

            # Select the requested radius from the radius dimension.
            ds = ds.sel(radius=self.radius_km)

            # If lat/lon were not embedded in the daily file (embed_latlon=False),
            # load them from the static grid file written alongside the daily files.
            coord_vars = ("clat", "clon") if self.grid_type == "icon" else ("lat", "lon")
            missing = [v for v in coord_vars if v not in ds and v not in ds.coords]
            if missing:
                grid_path = self.data_dir / f"{self.grid_type}_grid.nc"
                grid_ds = xr.open_dataset(grid_path)
                ds = ds.assign({v: grid_ds[v] for v in missing})
                grid_ds.close()

            # Promote auxiliary coordinate variables if CF decoding has not
            # done so already (behaviour varies across xarray versions).
            aux_coords = [v for v in ("clat", "clon", "lat", "lon")
                          if v in ds.data_vars and v not in ds.coords]
            if aux_coords:
                ds = ds.set_coords(aux_coords)

            # Replace fill value (−1) with NaN so anemoi treats masked cells
            # as missing data rather than negative strike counts.
            ds["lightning_count"] = (
                ds["lightning_count"]
                .astype(np.float32)
                .where(ds["lightning_count"] >= 0)
            )

            fs = XarrayFieldList.from_xarray(ds)

            for date in by_day[day]:
                selected = fs.sel(valid_datetime=date.isoformat())
                for f in selected:
                    out.append(
                        new_field_from_grid(f, self.grid)
                        if self.grid is not None
                        else f
                    )

            ds.close()

        return new_fieldlist_from_list(out)
