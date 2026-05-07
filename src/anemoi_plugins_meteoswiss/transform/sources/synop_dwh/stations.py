"""Canonical station-catalog construction from a jretrieve meta-info response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationCatalog:
    """Stable, ordered station catalog used as the dataset's cell axis.

    All arrays have length N == number of stations, sorted by `nat_abbr`
    ascending. The `station_id` column is the numeric jretrieve stationId,
    which is what every data row keys on.
    """
    nat_abbr: np.ndarray  # (N,) str
    station_id: np.ndarray  # (N,) int
    latitude: np.ndarray  # (N,) float64
    longitude: np.ndarray  # (N,) float64
    elevation: np.ndarray  # (N,) float64
    name: np.ndarray  # (N,) str

    @property
    def n(self) -> int:
        return len(self.nat_abbr)

    @classmethod
    def from_meta(cls, meta: pd.DataFrame) -> "StationCatalog":
        # The meta-info response has one row per (station × parameter × operating
        # period). Collapse to one row per station deterministically: sort by
        # (nat_abbr, parameter, op_since) then take the first per station.
        per_station = (
            meta.sort_values(["nat_abbr", "parameter", "op_since"], kind="stable")
            .drop_duplicates(subset=["station"], keep="first")
            .sort_values("nat_abbr", kind="stable")
            .reset_index(drop=True)
        )
        return cls(
            nat_abbr=per_station["nat_abbr"].to_numpy(dtype=object),
            station_id=per_station["station"].to_numpy(dtype=np.int64),
            latitude=per_station["latitude"].to_numpy(dtype=np.float64),
            longitude=per_station["longitude"].to_numpy(dtype=np.float64),
            elevation=per_station["elev"].to_numpy(dtype=np.float64),
            name=per_station["stn_name"].to_numpy(dtype=object),
        )
