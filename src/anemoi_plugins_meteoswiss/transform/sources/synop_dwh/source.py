"""Anemoi-datasets source plugin: synop measurement stations via DWH."""

from __future__ import annotations

import functools
import logging
import re
from datetime import datetime
from typing import Any, Sequence

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr

from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources.xarray_support import XarrayFieldList

from . import jretrieve
from .stations import StationCatalog

LOG = logging.getLogger(__name__)

# SwissMetNet stations carry an official WMO-synop WIGOS id (Swiss block 06,
# e.g. 0-20000-0-06710); agrometeo / precip-only gauges have only a national
# 0-756-... id or none. This is the dynamic way to isolate SMN from the
# inventory, which has no station-group filter.
DEFAULT_SMN_WIGOS_REGEX = r"^0-20000-0-06\d{3}$"
# Drop co-located networks that also carry WMO ids (NABEL air quality).
DEFAULT_SMN_EXCLUDE_PREFIXES: tuple[str, ...] = ("NAB",)


def _filter_inventory_stations(
    df: pd.DataFrame,
    wigos_regex: str = DEFAULT_SMN_WIGOS_REGEX,
    exclude_nat_abbr_prefixes: Sequence[str] = DEFAULT_SMN_EXCLUDE_PREFIXES,
) -> list[str]:
    """Select stations from an inventory response by WIGOS id, returning sorted
    unique nat_abbr. See the module constants for why the WIGOS pattern isolates
    SwissMetNet."""
    if df.empty or "location" not in df.columns or "wigosId" not in df.columns:
        return []
    rx = re.compile(wigos_regex)
    keep: set[str] = set()
    for nat_abbr, wigos in zip(df["location"], df["wigosId"]):
        if not isinstance(nat_abbr, str) or not isinstance(wigos, str):
            continue
        if not rx.match(wigos):
            continue
        if any(nat_abbr.startswith(p) for p in exclude_nat_abbr_prefixes):
            continue
        keep.add(nat_abbr)
    return sorted(keep)


class SynopDwhSource(Source):
    """Retrieves Swiss synop station observations from DWH and exposes them as a
    gridded earthkit FieldList (one cell per station)."""

    emoji = "🗂️"

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
        if stage != "prod":
            raise ValueError(f"Only 'prod' stage is supported, got {stage!r}.")
        recognized = ("group", "locations", "bbox", "inventory")
        present = [k for k in recognized if stations.get(k) is not None]
        if len(present) != 1:
            raise ValueError(
                f"stations must specify exactly one of {list(recognized)}, got {present}"
            )
        self.param: list[str] = list(param)
        self.stations: dict[str, Any] = stations
        self.stage: str = stage
        self.seq_type: str = seq_type
        self.increment_minutes: int = int(increment_minutes)
        self.timeout: int = int(timeout)

    @staticmethod
    def _as_datetime(v: Any) -> datetime | None:
        """Coerce a recipe date to a naive datetime. It arrives as a `datetime`
        in the `init`/`create` path but as an ISO string in the `load` path
        (the recipe is reloaded from the zarr metadata there)."""
        if v is None or isinstance(v, datetime):
            return v
        ts = pd.to_datetime(str(v))
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()

    def _recipe_date_range(self) -> tuple[datetime | None, datetime | None]:
        """The dataset's full [start, end] from the recipe, used to scope the
        station catalog to stations operating in the period. Same for every
        parallel worker (they share the recipe), so the cell axis stays stable.
        Falls back to (None, None) — the wide default — if unavailable."""
        try:
            dates = self.context.recipe.dates
            return self._as_datetime(dates.start), self._as_datetime(dates.end)
        except AttributeError:
            return None, None

    @functools.cached_property
    def resolved_stations(self) -> dict[str, Any]:
        """The station selection actually sent to DWH.

        `inventory:` mode resolves — once, via the inventory endpoint — to an
        explicit `locations` nat_abbr list, and is deterministic across workers
        (same params, period, and filter). Other modes pass through unchanged.
        The inventory endpoint has no station-group filter, so SwissMetNet is
        isolated in Python by WIGOS id (see `_filter_inventory_stations`)."""
        inv = self.stations.get("inventory")
        if inv is None:
            return self.stations
        opts = inv if isinstance(inv, dict) else {}
        start, end = self._recipe_date_range()
        if start is None or end is None:
            raise ValueError("inventory station selection requires recipe dates.")
        df = jretrieve.fetch_inventory(
            params=self.param, start=start, end=end, timeout_s=min(self.timeout, 180)
        )
        nat_abbr = _filter_inventory_stations(
            df,
            wigos_regex=opts.get("wigos_regex", DEFAULT_SMN_WIGOS_REGEX),
            exclude_nat_abbr_prefixes=opts.get(
                "exclude_nat_abbr_prefixes", DEFAULT_SMN_EXCLUDE_PREFIXES
            ),
        )
        if not nat_abbr:
            raise jretrieve.JretrieveError(
                "inventory station selection matched no stations — "
                "check params/dates/filter."
            )
        LOG.info("synop-dwh: inventory selected %d stations", len(nat_abbr))
        return {"locations": nat_abbr}

    @functools.cached_property
    def catalog(self) -> StationCatalog:
        """Canonical station catalog — fetched once per process, deterministic
        across parallel workers because it is scoped to the recipe's fixed date
        range and station selection.

        The catalog is built from a `--meta-info` call over `resolved_stations`,
        whose `-i nat_abbr,...` selector filters the meta response to exactly
        those stations. (`group:` is *not* honoured by `--meta-info`; prefer
        `inventory:` or an explicit `locations:` list.)"""
        # Fail fast on a missing binary / conf / credentials before we start a
        # potentially long build, rather than hours in.
        jretrieve.check_prerequisites(self.stage)
        start, end = self._recipe_date_range()
        meta = jretrieve.fetch_meta(
            stations=self.resolved_stations,
            params=self.param,
            start=start,
            end=end,
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
            stations=self.resolved_stations,
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


