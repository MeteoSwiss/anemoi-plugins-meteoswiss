"""Canonical station-catalog construction from a jretrieve meta-info response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Priority for choosing which parameter's metadata row defines a station's
# single coordinate when several parameters are current at once. Parameters not
# listed sort last.
_META_PARAM_PRIORITY: tuple[str, ...] = (
    "tre200s0",  # T_2M
    "tde200s0",  # TD_2M
    "pp0qffs0",  # PMSL
    "prestas0",  # PS
    "fkl010z0",  # 10m wind speed
    "dkl010z0",  # 10m wind direction
    "rre150h0",  # 1h precip
    "rre006i0",  # 6h precip
)


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
        # A station has one metadata row per parameter and operational period.
        # Collapse to one row per station by preferring, in order:
        #   1. the *current* location (empty/absent op_till),
        #   2. a fixed parameter priority,
        #   3. the most recent operational period (largest op_since).
        # Stations with no current row fall back to their latest period. This
        # matters for relocated stations, whose retired rows carry stale coords.
        #
        # We select one metadata entry per station even though coordinates can
        # in principle vary across parameters; handling per-(station, parameter)
        # metadata could be a future improvement.
        m = meta.copy()
        op_till = m["op_till"]
        m["_current"] = op_till.isna() | (op_till.astype(str).str.strip() == "")
        priority = {p: i for i, p in enumerate(_META_PARAM_PRIORITY)}
        m["_prio"] = m["parameter"].map(priority).fillna(len(priority)).astype(int)
        per_station = (
            m.sort_values(
                ["nat_abbr", "_current", "_prio", "op_since"],
                ascending=[True, False, True, False],
                kind="stable",
            )
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
