# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Reading and gridding of raw THX lightning strikes.

This module has no anemoi dependency. It reproduces, bit for bit, the
neighbourhood counts of the former ``lightning-grid`` preprocessing
(lightning-anemoi-dataset ``v1-netcdf``), without its per-strike Python loop
and without intermediate NetCDF files.

Conventions
-----------
- A window ``[start, end)`` contains strikes with ``start <= t < end`` (UTC).
- Strikes are only taken from the archive file of their own calendar day.
- A strike adds 1 to every cell whose centre is within the radius (chord
  distance on the unit sphere for ICON, LV95 metres for INCA). Radius ``0``
  means nearest cell only.
- Strikes outside the grid tessellation are dropped before counting.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from collections.abc import Iterable
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

LOG = logging.getLogger(__name__)

# Pipe-separated columns of the THX .prd format (the trailing "|" yields 17 fields).
THX_COLUMNS = [
    "datetime",
    "lon",
    "lat",
    "current",
    "nS",
    "mode",
    "intra",
    "Ax",
    "Ki2",
    "Exc",
    "Incl",
    "Arc",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
]
DISCHARGE_INTRA = {"CG": 0, "IC": 1}

DEFAULT_NETWORK_BBOX = (42.5, 50.2, 0.0, 17.0)  # lat_min, lat_max, lon_min, lon_max

# The original code used scipy's cKDTree, whose default leaf size is 16. The
# leaf size shapes the tree and with it how points exactly on the radius are
# decided, so it is fixed here to reproduce the original counts.
KDTREE_LEAFSIZE = 16

_NS_PER_DAY = 86_400 * 10**9
_NAT = np.iinfo(np.int64).min  # integer value of NaT
_EPOCH = datetime(1970, 1, 1)
_EPOCH_ORDINAL = _EPOCH.toordinal()


class ThxArchiveError(RuntimeError):
    """A THX archive file is missing or cannot be read."""


class Strikes(NamedTuple):
    """Strikes of one day as parallel arrays."""

    time_ns: np.ndarray  # int64, nanoseconds since 1970-01-01 UTC
    lon: np.ndarray  # float64, degrees
    lat: np.ndarray  # float64, degrees
    current: np.ndarray  # float64, kA
    intra: np.ndarray  # int64, 0 = cloud-to-ground, 1 = intra-cloud


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def thx_file_stem(day: date) -> str:
    """Archive file stem for a UTC day, e.g. ``THX252130000`` for 2025-08-01."""
    return f"THX{day.strftime('%y%j')}0000"


def _read_raw(day: date, archive_path: Path) -> str:
    stem = thx_file_stem(day)
    prd = archive_path / f"{stem}.prd"
    prd_zip = archive_path / f"{stem}.prd.zip"
    try:
        if prd.exists():
            return prd.read_text(errors="replace")
        if prd_zip.exists():
            with zipfile.ZipFile(prd_zip) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".prd"))
                return zf.read(name).decode(errors="replace")
    except (OSError, zipfile.BadZipFile, StopIteration) as e:
        raise ThxArchiveError(
            f"Cannot read THX file for {day} ({prd_zip if prd_zip.exists() else prd}): {e!r}. "
            "If this day has no usable data, list it under `dates: missing:` in the recipe."
        ) from e
    raise ThxArchiveError(
        f"No THX file for {day}: neither {prd} nor {prd_zip} exists. "
        "If this day has no data, list it under `dates: missing:` in the recipe."
    )


def _drop_malformed_lines(raw: str, day: date) -> str:
    """Remove lines with more fields than the THX format has.

    Such lines come from records that were glued together in the archive and
    make the pandas C parser fail. They are dropped and logged.
    """
    n_fields = len(THX_COLUMNS)
    lines = raw.splitlines()
    bad = [i for i, line in enumerate(lines) if line.count("|") + 1 > n_fields]
    if not bad:
        return raw
    LOG.warning(
        "THX %s: dropping %d malformed line(s) with more than %d fields (line numbers %s)",
        day,
        len(bad),
        n_fields,
        [i + 1 for i in bad[:10]],
    )
    bad_set = set(bad)
    return "\n".join(line for i, line in enumerate(lines) if i not in bad_set)


def _parse_times_ns(values: np.ndarray) -> np.ndarray:
    """Parse THX timestamps (``dd.mm.YYYY HH:MM:SS.f [UTC]``) to int64 ns since the epoch.

    The fixed format is tried first (fast); rows it cannot parse fall back to
    ``format="mixed", dayfirst=True``, the parser of the original code.
    Unparseable rows become NaT (``_NAT``).
    """
    s = (
        pd.Series(values)
        .astype(str)
        .str.strip()
        .str.replace(r"\s*UTC$", "", regex=True)
    )
    t = pd.to_datetime(s, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce", utc=True)
    fallback = t.isna()
    if fallback.any():
        t[fallback] = pd.to_datetime(
            s[fallback], format="mixed", dayfirst=True, utc=True, errors="coerce"
        )
    return pd.DatetimeIndex(t).as_unit("ns").asi8


def _to_float(values: np.ndarray) -> np.ndarray:
    """Numeric conversion as in the original code (``pd.to_numeric(errors="coerce")``)."""
    return np.asarray(
        pd.to_numeric(pd.Series(values), errors="coerce"), dtype=np.float64
    )


def read_thx_strikes(
    day: date, archive_path: str | Path, discharge_type: Iterable[str]
) -> Strikes:
    """Return the strikes of one UTC day.

    Rows with unparseable time, position, current or discharge type are
    dropped, as in the original code. Strikes whose timestamp is not on
    ``day`` are dropped as well.

    Raises
    ------
    ThxArchiveError
        If the file is missing or unreadable.
    """
    discharge_type = list(discharge_type)
    unknown = set(discharge_type) - set(DISCHARGE_INTRA)
    if unknown:
        raise ValueError(
            f"Unknown discharge_type {sorted(unknown)}; use any of {sorted(DISCHARGE_INTRA)}"
        )

    raw = _drop_malformed_lines(_read_raw(day, Path(archive_path)), day)
    df = pd.read_csv(
        io.StringIO(raw), sep="|", header=None, names=THX_COLUMNS, parse_dates=False
    )

    time_ns = _parse_times_ns(df["datetime"].to_numpy())
    lon, lat, current, intra = (
        _to_float(df[c].to_numpy()) for c in ("lon", "lat", "current", "intra")
    )

    bad_time = time_ns == _NAT
    if bad_time.any():
        LOG.warning(
            "THX %s: dropping %d row(s) with an unparseable timestamp",
            day,
            int(bad_time.sum()),
        )
    keep = (
        ~bad_time
        & ~np.isnan(lon)
        & ~np.isnan(lat)
        & ~np.isnan(current)
        & ~np.isnan(intra)
    )
    # Garbled rows can carry impossible coordinates (e.g. latitude 408.09). On the
    # unit sphere angles wrap around, so such a row could land inside the ICON grid
    # (408.09 -> 48.09 N); drop them explicitly.
    bad_position = keep & ((np.abs(lat) > 90) | (np.abs(lon) > 180))
    if bad_position.any():
        LOG.warning(
            "THX %s: dropping %d row(s) with latitude/longitude out of range",
            day,
            int(bad_position.sum()),
        )
    keep &= ~bad_position
    strikes = Strikes(
        time_ns[keep], lon[keep], lat[keep], current[keep], intra[keep].astype(np.int64)
    )

    wanted = np.isin(strikes.intra, [DISCHARGE_INTRA[d] for d in discharge_type])
    start = (day.toordinal() - _EPOCH_ORDINAL) * _NS_PER_DAY
    on_day = (strikes.time_ns >= start) & (strikes.time_ns < start + _NS_PER_DAY)
    if not on_day[wanted].all():
        LOG.warning(
            "THX %s: dropping %d strike(s) timestamped outside the file's day",
            day,
            int((wanted & ~on_day).sum()),
        )
    return Strikes(*(a[wanted & on_day] for a in strikes))


def read_thx_day(
    day: date, archive_path: str | Path, discharge_type: Iterable[str]
) -> pd.DataFrame:
    """:func:`read_thx_strikes` as a DataFrame indexed by UTC timestamp (``lon``, ``lat``, ``current``, ``intra``)."""
    s = read_thx_strikes(day, archive_path, discharge_type)
    index = pd.DatetimeIndex(
        pd.to_datetime(s.time_ns, unit="ns", utc=True), name="datetime"
    )
    return pd.DataFrame(
        {"lon": s.lon, "lat": s.lat, "current": s.current, "intra": s.intra},
        index=index,
    )


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def _km_to_chord(radius_km: float) -> float:
    """Great-circle radius in km as a chord length on the unit sphere."""
    return 2.0 * np.sin(np.radians(radius_km / 111.195))


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects SLURM / taskset CPU limits
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


def _unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    return np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )


class KDTreeGrid:
    """Shared counting logic for grids searched with a k-d tree.

    Subclasses set the attributes below and implement ``_to_tree_space``
    (strike lon/lat -> tree coordinates plus in-tessellation mask) and
    ``_radius`` (km -> tree units).
    """

    latitudes: np.ndarray  # cell centres, degrees
    longitudes: np.ndarray
    ncells: int
    _tree: KDTree

    #: strikes per neighbour search; bounds memory per thread for busy windows
    batch_size = 256
    #: threads for the neighbour search (capped by the CPUs available to the process)
    max_threads = 16

    def _to_tree_space(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _radius(self, radius_km: float) -> float:
        raise NotImplementedError

    def network_mask(self, bbox: Sequence[float]) -> np.ndarray:
        """True for cells inside the THX network bbox ``[lat_min, lat_max, lon_min, lon_max]`` (inclusive)."""
        lat_min, lat_max, lon_min, lon_max = bbox
        lat, lon = self.latitudes, self.longitudes
        return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

    def prepare(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tree coordinates of the strikes and their in-tessellation mask."""
        return self._to_tree_space(
            np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        )

    def count(self, points: np.ndarray, radii_km: Sequence[float]) -> np.ndarray:
        """Neighbourhood counts, shape ``(len(radii_km), ncells)``, int64.

        ``points`` are in-tessellation strikes in tree coordinates (see
        :meth:`prepare`).

        For each batch of strikes, ``sparse_distance_matrix`` returns every
        (cell, strike) pair with distance <= r, computed exactly (``eps=0``);
        the cell indices are then counted. Batches run in parallel threads.
        This is the same criterion as the original per-strike
        ``query_ball_point`` loop, and gives identical counts on all verified
        days.
        """
        counts = np.zeros((len(radii_km), self.ncells), dtype=np.int64)
        if len(points) == 0:
            return counts
        threads = max(1, min(self.max_threads, _available_cpus()))
        with ThreadPoolExecutor(threads) as pool:
            for i, r_km in enumerate(radii_km):
                if r_km == 0:
                    _, nearest = self._tree.query(points, k=1, workers=threads)
                    counts[i] = np.bincount(nearest, minlength=self.ncells)
                    continue
                r = self._radius(r_km)
                batches = (
                    points[b : b + self.batch_size]
                    for b in range(0, len(points), self.batch_size)
                )
                for partial in pool.map(
                    lambda batch: self._count_batch(batch, r), batches
                ):
                    counts[i] += partial
        return counts

    def _count_batch(self, batch: np.ndarray, r: float) -> np.ndarray:
        """Cells within ``r`` of each strike in ``batch``, counted per cell."""
        strikes = KDTree(batch, leafsize=KDTREE_LEAFSIZE)
        pairs = self._tree.sparse_distance_matrix(strikes, r, output_type="ndarray")
        return np.bincount(pairs["i"], minlength=self.ncells)


class IconGrid(KDTreeGrid):
    """ICON native unstructured grid; distances are chords on the unit sphere.

    Parameters
    ----------
    clat_deg, clon_deg:
        Cell-centre latitudes/longitudes in degrees.
    """

    def __init__(self, clat_deg: np.ndarray, clon_deg: np.ndarray):
        self.latitudes = np.asarray(clat_deg, dtype=np.float64)
        self.longitudes = np.asarray(clon_deg, dtype=np.float64)
        self.ncells = len(self.latitudes)
        self._tree = KDTree(
            _unit_xyz(self.latitudes, self.longitudes), leafsize=KDTREE_LEAFSIZE
        )

        # Tessellation radius: half the largest nearest-neighbour chord among a
        # strided sample of cell centres. Strikes farther than this from their
        # nearest cell centre are outside the grid and are dropped.
        stride = max(1, self.ncells // 100_000)
        sample = self._tree.data[::stride]
        nn_dists, _ = self._tree.query(sample, k=2, workers=-1)
        self._domain_chord = float(np.max(nn_dists[:, 1])) / 2

    @classmethod
    def from_file(cls, path: str | Path) -> IconGrid:
        """Read ``clat``/``clon`` (radians) from an ICON grid NetCDF file."""
        import xarray as xr

        with xr.open_dataset(path, engine="netcdf4") as ds:
            return cls(
                np.degrees(ds["clat"].to_numpy()), np.degrees(ds["clon"].to_numpy())
            )

    def _to_tree_space(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz = _unit_xyz(lat, lon)
        if len(xyz) == 0:
            return xyz, np.zeros(0, dtype=bool)
        nn_dist, _ = self._tree.query(xyz, k=1, workers=-1)
        return xyz, nn_dist <= self._domain_chord

    def _radius(self, radius_km: float) -> float:
        return _km_to_chord(radius_km)


class IncaGrid(KDTreeGrid):
    """INCA/RADAR 710 x 640 regular 1 km grid in CH1903+/LV95 (EPSG:2056).

    Cells are ordered row-major over ``(y, x)``, with ``y`` increasing
    northwards from the first row.
    """

    NX = 710
    NY = 640
    X0 = 2_255_500.0  # easting of the first column centre [m]
    Y0 = 840_500.0  # northing of the first row centre [m]
    DX = 1_000.0
    DY = 1_000.0

    def __init__(self):
        from pyproj import Transformer

        self._to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
        to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

        xx, yy = np.meshgrid(
            self.X0 + np.arange(self.NX) * self.DX,
            self.Y0 + np.arange(self.NY) * self.DY,
        )
        lon, lat = to_wgs84.transform(xx, yy)
        self.latitudes = np.asarray(lat).ravel()
        self.longitudes = np.asarray(lon).ravel()
        self.ncells = self.NX * self.NY
        self._tree = KDTree(
            np.column_stack([xx.ravel(), yy.ravel()]), leafsize=KDTREE_LEAFSIZE
        )

    def _to_tree_space(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sx, sy = self._to_lv95.transform(lon, lat)
        sx, sy = np.asarray(sx), np.asarray(sy)
        # The tessellation of a regular grid extends half a cell beyond the outer centres.
        inside = (
            (sx >= self.X0 - self.DX / 2)
            & (sx <= self.X0 + (self.NX - 0.5) * self.DX)
            & (sy >= self.Y0 - self.DY / 2)
            & (sy <= self.Y0 + (self.NY - 0.5) * self.DY)
        )
        return np.column_stack([sx, sy]), inside

    def _radius(self, radius_km: float) -> float:
        return radius_km * 1_000.0


# ---------------------------------------------------------------------------
# Strikes and windows
# ---------------------------------------------------------------------------


def _to_ns(t: datetime) -> int:
    """Nanoseconds since the epoch; naive datetimes are taken as UTC."""
    if t.tzinfo is not None:
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    d = t - _EPOCH
    return (d.days * 86_400 + d.seconds) * 10**9 + d.microseconds * 1_000


def _ns_to_day(ns: int) -> date:
    return date.fromordinal(_EPOCH_ORDINAL + ns // _NS_PER_DAY)


class StrikeArchive:
    """Strikes of the THX archive, prepared for one grid.

    Days are read on demand and kept in a small cache, so consecutive anemoi
    groups (which share the day boundary) do not re-read files.
    """

    def __init__(
        self,
        archive_path: str | Path,
        discharge_type: Iterable[str],
        grid: KDTreeGrid,
        cache_days: int = 3,
    ):
        self.archive_path = Path(archive_path)
        self.discharge_type = list(discharge_type)
        self.grid = grid
        self.cache_days = cache_days
        self._cache: dict[date, tuple[np.ndarray, np.ndarray]] = {}

    def day(self, day: date) -> tuple[np.ndarray, np.ndarray]:
        """In-tessellation strikes of a day: (time in ns since epoch, tree coordinates)."""
        if day not in self._cache:
            s = read_thx_strikes(day, self.archive_path, self.discharge_type)
            points, inside = self.grid.prepare(s.lon, s.lat)
            self._cache[day] = (s.time_ns[inside], points[inside])
            while len(self._cache) > self.cache_days:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[day]

    def window(self, start: datetime, end: datetime) -> np.ndarray:
        """Tree coordinates of the strikes with ``start <= t < end``."""
        s, e = _to_ns(start), _to_ns(end)
        parts = []
        d, last = _ns_to_day(s), _ns_to_day(e - 1)
        while d <= last:
            times, points = self.day(d)
            parts.append(points[(times >= s) & (times < e)])
            d += timedelta(days=1)
        return np.concatenate(parts)


AGGREGATIONS = ("accumulation", "maximum", "average")


def window_counts(
    archive: StrikeArchive,
    valid: datetime,
    period: timedelta,
    radii_km: Sequence[float],
    aggregation: str = "accumulation",
    sub_period: timedelta | None = None,
) -> np.ndarray:
    """Lightning counts for the window ending at ``valid``, shape ``(n_radii, ncells)``.

    - ``accumulation``: strikes in ``[valid - period, valid)``.
    - ``maximum`` / ``average``: max / mean of the accumulations over the
      consecutive ``sub_period`` windows that tile ``[valid - period, valid)``.

    Returns float64 (exact integers for accumulation and maximum).
    """
    grid = archive.grid
    if aggregation == "accumulation":
        return grid.count(archive.window(valid - period, valid), radii_km).astype(
            np.float64
        )
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
        )
    if sub_period is None:
        raise ValueError(f"aggregation {aggregation!r} requires sub_period")

    n_sub = period // sub_period
    stack = np.empty((n_sub, len(radii_km), grid.ncells), dtype=np.int64)
    start = valid - period
    for k in range(n_sub):
        stack[k] = grid.count(
            archive.window(start + k * sub_period, start + (k + 1) * sub_period),
            radii_km,
        )
    if aggregation == "maximum":
        return stack.max(axis=0).astype(np.float64)
    return stack.mean(axis=0)
