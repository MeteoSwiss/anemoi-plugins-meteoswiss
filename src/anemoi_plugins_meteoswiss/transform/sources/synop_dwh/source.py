"""Anemoi-datasets source plugin: synop measurement stations via DWH."""

from __future__ import annotations

import functools
import logging
from datetime import datetime
from typing import Any

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr

from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources.xarray_support import XarrayFieldList

from . import jretrieve
from .stations import StationCatalog

LOG = logging.getLogger(__name__)


class SynopDwhSource(Source):
    """Retrieves Swiss synop station observations from DWH and exposes them as a
    gridded earthkit FieldList (one cell per station)."""

    emoji = "📡"

    def __init__(
        self,
        context: Any,
        *,
        param: list[str],
        stations: dict[str, Any],
        stage: str = "prod",
        seq_type: str = "surface",
        increment_minutes: int = 10,
        timeout: int = 600,
    ):
        super().__init__(context)
        if not param:
            raise ValueError("param must be a non-empty list of DWH short names.")
        self.param: list[str] = list(param)
        self.stations: dict[str, Any] = stations
        self.stage: str = stage
        self.seq_type: str = seq_type
        self.increment_minutes: int = int(increment_minutes)
        self.timeout: int = int(timeout)

    @functools.cached_property
    def catalog(self) -> StationCatalog:
        """Canonical station catalog — fetched once per process, deterministic
        across parallel workers because the meta-info call uses a fixed wide
        time range."""
        meta = jretrieve.fetch_meta(
            stations=self.stations,
            params=self.param,
            seq_type=self.seq_type,
            stage=self.stage,
            timeout_s=min(self.timeout, 300),
        )
        cat = StationCatalog.from_meta(meta)
        LOG.info("synop-dwh: resolved %d stations: %s",
                 cat.n, ", ".join(cat.nat_abbr[:10].tolist()) + ("..." if cat.n > 10 else ""))
        return cat

    def execute(self, dates: Any) -> ekd.FieldList:
        # Accept GroupOfDates or plain list[datetime].
        date_list: list[datetime] = list(getattr(dates, "dates", dates))
        if not date_list:
            return ekd.from_source("empty")

        date_list = sorted(set(date_list))
        catalog = self.catalog

        df = jretrieve.fetch_data(
            stations=self.stations,
            params=self.param,
            start=date_list[0],
            end=date_list[-1],
            increment_minutes=self.increment_minutes,
            seq_type=self.seq_type,
            stage=self.stage,
            timeout_s=self.timeout,
        )

        ds = self._df_to_xarray(df, date_list, catalog)
        return XarrayFieldList.from_xarray(ds)

    def _df_to_xarray(
        self,
        df: pd.DataFrame,
        dates: list[datetime],
        catalog: StationCatalog,
    ) -> xr.Dataset:
        """Pivot a long-form jretrieve dataframe into a `(time, station)` cube
        aligned to the canonical station order, with NaN for missing cells."""
        time_index = pd.DatetimeIndex(dates)
        n_t = len(time_index)
        n_s = catalog.n

        coords = {
            "time": ("time", time_index.values.astype("datetime64[ns]")),
            "station": ("station", catalog.nat_abbr),
            "latitude": ("station", catalog.latitude),
            "longitude": ("station", catalog.longitude),
        }
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

        if df.empty:
            for p in self.param:
                data_vars[p] = (("time", "station"), np.full((n_t, n_s), np.nan, dtype=np.float32))
        else:
            df = df.copy()
            df["time"] = pd.to_datetime(df["termin"], format="%Y%m%d%H%M%S", utc=False)
            station_to_idx = {sid: i for i, sid in enumerate(catalog.station_id)}
            time_to_idx = {t: i for i, t in enumerate(time_index)}

            df["_si"] = df["station"].map(station_to_idx)
            df["_ti"] = df["time"].map(time_to_idx)
            df = df.dropna(subset=["_si", "_ti"])
            df["_si"] = df["_si"].astype(int)
            df["_ti"] = df["_ti"].astype(int)

            for p in self.param:
                arr = np.full((n_t, n_s), np.nan, dtype=np.float32)
                if p in df.columns:
                    arr[df["_ti"].to_numpy(), df["_si"].to_numpy()] = (
                        df[p].to_numpy(dtype=np.float32)
                    )
                data_vars[p] = (("time", "station"), arr)

        # Attach CF-ish attributes so the xarray flavour guesser picks the dims.
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds["latitude"].attrs.update(units="degrees_north", standard_name="latitude")
        ds["longitude"].attrs.update(units="degrees_east", standard_name="longitude")
        ds["station"].attrs.update(standard_name="station")
        ds["time"].attrs.update(standard_name="time")
        return ds


