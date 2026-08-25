"""
CleanObservation — real-time quality control of station observations before nudging.

Runs with only the two inputs available at forecast time: the raw station
observation Parquet (from RetrieveObservation) and the model's own T0 forecast
fields (already passed through this filter's ``data`` argument, alongside every
other GRIB field in the cutout). No historical climatology is used or required.

Two QC tiers, applied per variable/column, in order:

1.  Physical bounds — a fixed (min, max) sanity range per column, no reference
    data needed. Catches unit bugs, sentinel/fill values, and otherwise
    nonsensical readings regardless of season or synoptic situation. Loose by
    design: a backstop, not the primary detector.
2.  Background check (a.k.a. first-guess check — standard practice in
    operational NWP QC/data assimilation) — compares each station's
    observation against the model's own T0 forecast at its nearest ICON cell
    (temperature-like variables get the same lapse-rate reduction
    NudgeTowardObservation itself applies before differencing, so a real
    elevation-driven offset isn't mistaken for an error), then flags any
    station whose residual is a robust (median/MAD, Tukey-style) outlier
    relative to every other station reporting that variable at this ref_time.

    This is deliberately benchmarked against the model background rather than
    against a consensus of neighbouring stations (the way
    NudgeTowardObservation's own leave-one-out reliability check works):
    the background can never itself be corrupted by another station's bad
    value, whereas a leave-one-out neighbour check can — a single badly
    corrupted station can drag down the *neighbours'* reliability too, since
    they use it as one of their own leave-one-out inputs. Running this check
    here, before nudging, removes a bad observation outright rather than
    letting it degrade a whole neighbourhood's correction radius downstream.

Precipitation/gust (``tp``/``vmax``) are exempted from the background check
by default (see _DEFAULT_BACKGROUND_CHECK_STD): both are highly localised
(convective bursts, gust fronts), so a station's true value can legitimately
be a large network-wide statistical outlier without being wrong. Physical
bounds still apply to them.

Any station failing either check has that column's value set to NaN, and a
companion boolean column ``"{col}_qc_dropped"`` set to True for it (created
lazily — absent entirely if nothing was ever dropped for that column).
The nudging filter (IDW- or kriging-based) already treats a NaN observation
exactly like a station that simply didn't report that variable (``valid =
stations[col].notna()``) — no changes are required there for the nudging
correction itself; the flag column exists purely so its diagnostic plot can
distinguish "QC removed this" from "never reported this" and show the
former as a dedicated panel (see that filter's own diagnostic-plot method).

The column/shortName/offset mapping and lapse-rate defaults below mirror
nudging.py's own PARAM_MAP/_DEFAULT_LAPSE_RATE*/_DEFAULT_ICON_OROG_FILE.
They are intentionally duplicated rather than imported, to keep this filter
self-contained — if nudging.py's mapping or lapse-rate defaults change,
update these to match.
"""

import logging
from pathlib import Path
from typing import Optional

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
from anemoi.transform.filter import Filter
from pyproj import Transformer
from scipy.spatial import cKDTree

LOG = logging.getLogger(__name__)

# Mirrors nudging.py's PARAM_MAP: GRIB shortName -> (station Parquet column,
# unit offset applied to obs before differencing against the background).
_PARAM_MAP = {
    "T_2M":     ("2t",   0.0),
    "TD_2M":    ("2d",   0.0),
    "U_10M":    ("10u",  0.0),
    "V_10M":    ("10v",  0.0),
    "PMSL":     ("msl",  0.0),
    "TOT_PREC": ("tp",   0.0),
    "VMAX_10M": ("vmax", 0.0),
}

# Mirrors nudging.py's _DEFAULT_LAPSE_RATE / _DEFAULT_LAPSE_RATE_VARS.
_DEFAULT_LAPSE_RATE = 0.0065  # K/m
_DEFAULT_LAPSE_RATE_VARS = frozenset({"T_2M"})

# Mirrors nudging.py's _DEFAULT_ICON_OROG_FILE.
_DEFAULT_ICON_OROG_FILE = (
    "/scratch/mch/icontest/testing-input-data/c2sm/icon-1/"
    "external_parameter_icon_grid_0001_R19B08_mch.nc"
)

# Tier 0 — physical plausibility bounds (min, max), SI units, keyed by the
# station Parquet column name. Deliberately loose (global-extreme-scale, not
# Swiss-climatology-scale): a real, uncorrupted reading should never come
# close to these. Only meant to catch unit bugs / sentinel values (e.g. a
# literal -999) independent of season or weather situation.
_DEFAULT_PHYSICAL_BOUNDS = {
    "2t":   (183.15, 333.15),    # -90..60 degC
    "2d":   (183.15, 313.15),    # -90..40 degC
    "10u":  (-75.0, 75.0),       # m/s
    "10v":  (-75.0, 75.0),       # m/s
    "msl":  (85000.0, 108500.0), # Pa (850..1085 hPa)
    "tp":   (0.0, 300.0),        # mm per observation interval
    "vmax": (0.0, 120.0),        # m/s
}

# Tier 1 — background-check robust z-score threshold (median/MAD sigmas),
# keyed by column. A column with no entry here only gets the physical-bounds
# check above — see module docstring for why tp/vmax are omitted by default.
_DEFAULT_BACKGROUND_CHECK_STD = {
    "2t": 8.0,
    "2d": 8.0,
    "10u": 10.0,
    "10v": 10.0,
    "msl": 8.0,
}

# Below this many valid stations, a median/MAD estimate is unstable/meaningless;
# the background check is skipped for that column/ref_time (physical bounds
# still apply).
_MIN_STATIONS_FOR_BACKGROUND_CHECK = 10


class CleanObservation(Filter):
    """Quality-control station observations against the model's own T0 field.

    Reads a Parquet file produced by RetrieveObservation, applies the two-tier
    QC described in the module docstring (physical bounds, then a background
    check against the T0 forecast already present in ``data``), and writes the
    cleaned DataFrame to a new Parquet file for use by NudgeTowardObservation.
    The forecast fields themselves are passed through unchanged.

    Holdout protection
    ------------------
    NudgeTowardObservation's *exclude_stations*/*holdout_fraction* withholds
    specific stations from the nudging correction so their true observation
    can be used as an independent check afterwards (the diagnostic plot's
    "post-nudging RMSE (holdout)" panel: was the corrected field actually
    closer to a station the correction never saw?). QC runs upstream of that,
    on the full station set, with no inherent knowledge of which stations
    will later be held out — so without special handling, QC dropping a
    holdout station's value would silently destroy that independent check
    instead of merely excluding the station from nudging (holdout's actual
    intended effect).

    To prevent that, *exclude_stations*/*holdout_fraction*/*holdout_seed*
    below reproduce NudgeTowardObservation's own station selection (see
    ``_compute_protected_ids``), and neither QC tier is ever allowed to drop
    a protected station's value — a station that would otherwise fail QC is
    logged (so a bad holdout observation is still visible) but left
    untouched. These three parameters must be configured identically to
    NudgeTowardObservation's own for the two to agree on the same set; using
    a YAML anchor/alias so both filters read the same list (rather than two
    independently-maintained copies) is strongly recommended.

    Parameters
    ----------
    obs_path_in : str
        Path to the raw observation Parquet file written by RetrieveObservation.
    obs_path_out : str
        Path where the cleaned observation Parquet file will be written.
    icon_grid_dir : str
        Directory containing ``icon_grid_0001_R19B08_mch.nc`` (for nearest-cell
        snapping — same grid NudgeTowardObservation uses).
    icon_orog_file : str
        Path to the ICON extpar NetCDF providing the model's native orography
        (variable ``topography_c``), used for the same lapse-rate reduction
        NudgeTowardObservation applies to temperature-like variables.
    lapse_rate : float
        Standard-atmosphere lapse rate [K/m] used to reduce station
        observations to the model's elevation before differencing against the
        background, for variables in *lapse_rate_vars* (default: 0.0065).
    lapse_rate_vars : list of str, optional
        GRIB shortNames to which *lapse_rate* is applied. Defaults to
        ``["T_2M"]``, matching NudgeTowardObservation's default.
    run_mode : str
        ``'depl'``: ref_time = minimum valid_time across all fields.
        ``'devt'``: ref_time = valid_time of the first field.
    physical_bounds : dict, optional
        Per-column ``(min, max)`` overrides merged on top of
        *_DEFAULT_PHYSICAL_BOUNDS* (only the given keys are replaced).
    background_check_std : dict, optional
        Per-column robust z-score threshold overrides merged on top of
        *_DEFAULT_BACKGROUND_CHECK_STD*. A column set to ``None`` here
        disables the background check for that column (physical bounds still
        apply).
    reliability_eps : float
        Numerical floor on the robust (MAD-based) scale estimate, avoiding
        division by zero when stations agree with the background almost
        exactly (default: 1e-6).
    holdout_fraction : float, optional
        Must match NudgeTowardObservation's own *holdout_fraction* exactly
        (see "Holdout protection" above) — used only to identify which
        stations to protect from QC, never to actually remove any station
        here. Mutually exclusive with *exclude_stations*.
    holdout_seed : int
        Must match NudgeTowardObservation's own *holdout_seed* exactly, so
        the random holdout draw is reproduced identically (default: 42).
    exclude_stations : list of str, optional
        Must match NudgeTowardObservation's own *exclude_stations* exactly.
        Mutually exclusive with *holdout_fraction*.
    """

    def __init__(
        self,
        obs_path_in: str,
        obs_path_out: str,
        icon_grid_dir: str = "/scratch/mch/llanzila/sruc/aux_files",
        icon_orog_file: str = _DEFAULT_ICON_OROG_FILE,
        lapse_rate: float = _DEFAULT_LAPSE_RATE,
        lapse_rate_vars: Optional[list] = None,
        run_mode: str = "depl",
        physical_bounds: Optional[dict] = None,
        background_check_std: Optional[dict] = None,
        reliability_eps: float = 1e-6,
        holdout_fraction: Optional[float] = None,
        holdout_seed: int = 42,
        exclude_stations: Optional[list] = None,
    ):
        if run_mode not in ("devt", "depl"):
            raise ValueError(f"run_mode must be 'devt' or 'depl', got {run_mode!r}")
        if holdout_fraction is not None and exclude_stations is not None:
            raise ValueError(
                "holdout_fraction and exclude_stations are mutually exclusive."
            )
        if holdout_fraction is not None and not (0.0 <= holdout_fraction <= 1.0):
            raise ValueError(
                f"holdout_fraction must be in [0, 1], got {holdout_fraction!r}"
            )

        self.obs_path_in = Path(obs_path_in)
        self.obs_path_out = Path(obs_path_out)
        self.icon_grid_dir = Path(icon_grid_dir)
        self.icon_orog_file = Path(icon_orog_file)
        self.lapse_rate = lapse_rate
        self.lapse_rate_vars = (
            frozenset(lapse_rate_vars) if lapse_rate_vars is not None else _DEFAULT_LAPSE_RATE_VARS
        )
        self.run_mode = run_mode
        self.reliability_eps = reliability_eps
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed
        self.exclude_stations = list(exclude_stations) if exclude_stations is not None else None

        self.physical_bounds = dict(_DEFAULT_PHYSICAL_BOUNDS)
        if physical_bounds is not None:
            self.physical_bounds.update(physical_bounds)

        self.background_check_std = dict(_DEFAULT_BACKGROUND_CHECK_STD)
        if background_check_std is not None:
            self.background_check_std.update(background_check_std)

        self._load_icon_grid()

        LOG.info(
            "CleanObservation initialised: physical_bounds=%s, "
            "background_check_std=%s, lapse_rate=%.5f K/m (vars=%s)",
            self.physical_bounds, self.background_check_std,
            self.lapse_rate, sorted(self.lapse_rate_vars),
        )
        super().__init__()

    def _load_icon_grid(self) -> None:
        """Load the ICON grid (for nearest-cell snapping) and native orography
        (for the lapse-rate correction), once at construction time."""
        ds = xr.open_dataset(self.icon_grid_dir / "icon_grid_0001_R19B08_mch.nc")
        lat_icon = np.degrees(ds["clat"].values).ravel()
        lon_icon = np.degrees(ds["clon"].values).ravel()

        # always_xy=True: transform(lon, lat) -> (easting, northing).
        self._wgs84_to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
        grid_x, grid_y = self._wgs84_to_lv95.transform(lon_icon, lat_icon)
        self._grid_xy_km = np.c_[grid_x, grid_y] / 1000.0  # (n_cells, 2), km
        self._grid_tree = cKDTree(self._grid_xy_km)

        ds_orog = xr.open_dataset(self.icon_orog_file)
        self._icon_orog = ds_orog["topography_c"].values.astype(np.float32)

        LOG.info(
            "CleanObservation: ICON grid (%d cells) loaded from %s; orography "
            "loaded from %s",
            len(lat_icon), self.icon_grid_dir, self.icon_orog_file,
        )

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Read, QC, and write observations; return forecast fields unchanged.

        Parameters
        ----------
        data : ekd.FieldList
            Forecast fields (passed through unchanged) — also used as the T0
            background reference for the QC background check.

        Returns
        -------
        ekd.FieldList
            The input data, unchanged.
        """
        if not self.obs_path_in.exists():
            raise FileNotFoundError(f"Observation file not found: {self.obs_path_in}")

        df = pd.read_parquet(self.obs_path_in)
        LOG.info("Loaded %d stations from %s", len(df), self.obs_path_in)

        # Same ref_time convention as NudgeTowardObservation.forward(): the
        # time step the background check compares observations against.
        ref_time = (
            data[0].datetime()["valid_time"]
            if self.run_mode == "devt"
            else min(f.datetime()["valid_time"] for f in data)
        )

        protected_ids = self._compute_protected_ids(df)
        if protected_ids:
            LOG.info(
                "CleanObservation: %d station(s) protected from QC (holdout/"
                "excluded — see class docstring's 'Holdout protection'): %s",
                len(protected_ids), sorted(protected_ids),
            )

        df = self._clean(df, data, ref_time, protected_ids)

        self.obs_path_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.obs_path_out)
        LOG.info("Saved %d cleaned stations to %s", len(df), self.obs_path_out)

        return data

    def _compute_protected_ids(self, df: pd.DataFrame) -> frozenset:
        """Reproduce NudgeTowardObservation._apply_holdout's station selection,
        so QC can protect the same station IDs from ever being dropped (see
        class docstring, "Holdout protection"). Deliberately mirrors that
        method's branch structure exactly (mutual exclusivity of
        *exclude_stations*/*holdout_fraction* is already enforced in
        __init__, so only one of the two branches below is ever active) —
        including using the *same* np.random.default_rng(self.holdout_seed)
        draw over the *same* df.index, so the random holdout selection is
        bit-for-bit reproducible between this filter and
        NudgeTowardObservation, as long as both are configured with the same
        holdout_fraction/holdout_seed and see the same station row set (QC
        never adds/removes/reorders rows, only sets individual values to NaN,
        so the index NudgeTowardObservation._load_stations() later sees is
        identical to this one).
        """
        if self.exclude_stations is not None:
            return frozenset(self.exclude_stations) & set(df.index)
        if self.holdout_fraction is not None:
            if self.holdout_fraction == 0.0:
                return frozenset()
            if self.holdout_fraction == 1.0:
                return frozenset(df.index)
            n_holdout = round(len(df) * self.holdout_fraction)
            rng = np.random.default_rng(self.holdout_seed)
            held_out = rng.choice(df.index, size=n_holdout, replace=False)
            return frozenset(held_out)
        return frozenset()

    def _clean(
        self,
        df: pd.DataFrame,
        data: ekd.FieldList,
        ref_time,
        protected_ids: frozenset,
    ) -> pd.DataFrame:
        """Apply the two-tier QC (see module docstring) to every column present
        in *df* that has a matching T0 field in *data*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw station observations with columns for each variable plus
            ``latitude``/``longitude`` (and ideally ``elevation``), in SI units.
        data : ekd.FieldList
            Forecast fields, used to look up each variable's T0 background.
        ref_time
            The valid_time to compare observations against.
        protected_ids : frozenset
            Station IDs (see _compute_protected_ids) that must never be
            dropped by either QC tier, regardless of outcome.

        Returns
        -------
        pd.DataFrame
            Cleaned observations: values failing either QC tier are set to
            NaN, except for stations in *protected_ids*.
        """
        df = df.copy()

        if "elevation" not in df.columns:
            LOG.warning(
                "Station Parquet missing 'elevation' column; the background "
                "check's lapse-rate correction will be skipped (the check "
                "itself still runs, just without the elevation adjustment)."
            )

        n_dropped_by_col = {}
        for shortname, (col, offset) in _PARAM_MAP.items():
            if col not in df.columns:
                continue
            n_before = int(df[col].notna().sum())
            if n_before == 0:
                continue

            matches = [
                f for f in data.sel(shortName=shortname)
                if f.datetime()["valid_time"] == ref_time
            ]
            if not matches:
                LOG.warning(
                    "CleanObservation: no '%s' field at ref_time=%s; skipping "
                    "the background check for column '%s' (physical-bounds "
                    "check still applied).",
                    shortname, ref_time, col,
                )
                n_flagged = self._apply_physical_bounds(df, col, offset, protected_ids)
            else:
                if len(matches) > 1:
                    LOG.warning(
                        "CleanObservation: %d '%s' fields at ref_time=%s; using the first.",
                        len(matches), shortname, ref_time,
                    )
                B_flat = np.asarray(matches[0].values, dtype=float).ravel()
                n_flagged = self._qc_column(df, col, offset, shortname, B_flat, protected_ids)

            n_dropped_by_col[col] = n_flagged
            # The number of stations deleted by QC for this variable — always
            # logged, whether or not any were actually dropped, so a clean run
            # is as visible in the logs as a dirty one.
            LOG.info(
                "CleanObservation: '%s' (%s) — %d/%d station(s) dropped by QC",
                col, shortname, n_flagged, n_before,
            )

        LOG.info(
            "CleanObservation: QC summary for ref_time=%s — %d station(s) dropped "
            "in total across %d variable(s): %s",
            ref_time, sum(n_dropped_by_col.values()), len(n_dropped_by_col), n_dropped_by_col,
        )

        return df

    def _drop_flagged(
        self,
        df: pd.DataFrame,
        col: str,
        fail_idx: pd.Index,
        reason: str,
        protected_ids: frozenset,
    ) -> int:
        """Set df.loc[fail_idx, col] = NaN, excluding any station in
        *protected_ids* (see class docstring, "Holdout protection"): a
        holdout station's true observation must survive QC untouched, since
        NudgeTowardObservation's diagnostic plot computes pre/post-nudging
        RMSE against it as an independent check the correction itself never
        saw — dropping it here would silently corrupt that check instead of
        just excluding the station from nudging (holdout's actual intended
        effect). A protected station that fails QC is still logged (so a bad
        holdout observation remains visible), just not dropped. Returns the
        number of stations actually dropped.
        """
        protected = [s for s in fail_idx if s in protected_ids]
        if protected:
            LOG.warning(
                "CleanObservation: %d holdout/excluded station(s) failed QC "
                "for '%s' (%s) but were NOT dropped — a held-out station's "
                "true observation is always preserved for independent "
                "verification: %s",
                len(protected), col, reason, protected,
            )
        to_drop = fail_idx.difference(protected_ids)
        if len(to_drop):
            df.loc[to_drop, col] = np.nan
            # "{col}_qc_dropped" flag column, persisted into the cleaned Parquet:
            # lets NudgeTowardObservation's diagnostic plot show which stations
            # QC removed for this variable, distinct from a station that simply
            # never reported it. Created lazily so a column with nothing ever
            # dropped doesn't grow the Parquet with an all-False column.
            flag_col = f"{col}_qc_dropped"
            if flag_col not in df.columns:
                df[flag_col] = False
            df.loc[to_drop, flag_col] = True
        return int(len(to_drop))

    def _apply_physical_bounds(
        self, df: pd.DataFrame, col: str, offset: float, protected_ids: frozenset,
    ) -> int:
        """Tier 0: flag (set to NaN) any non-protected station outside
        *self.physical_bounds* for *col*. No reference data required.
        Returns the number actually dropped."""
        bounds = self.physical_bounds.get(col)
        if bounds is None:
            return 0
        lo, hi = bounds
        valid = df[col].notna()
        obs = df.loc[valid, col] + offset
        out_of_range = (obs < lo) | (obs > hi)
        n = int(out_of_range.sum())
        if n == 0:
            return 0
        fail_idx = df.index[valid][out_of_range.to_numpy()]
        LOG.warning(
            "CleanObservation: %d station(s) failed physical bounds for "
            "'%s' (%.2f..%.2f): %s",
            n, col, lo, hi, fail_idx.tolist(),
        )
        return self._drop_flagged(df, col, fail_idx, "physical bounds", protected_ids)

    def _qc_column(
        self,
        df: pd.DataFrame,
        col: str,
        offset: float,
        shortname: str,
        B_flat: np.ndarray,
        protected_ids: frozenset,
    ) -> int:
        """Apply both QC tiers to one column: physical bounds, then (if
        configured and enough stations remain) the background check against
        *B_flat* (this variable's T0 field). Returns the total number of
        stations actually dropped across both tiers (excluding any
        protected station — see _drop_flagged)."""
        n_bounds = self._apply_physical_bounds(df, col, offset, protected_ids)

        valid = df[col].notna()
        if valid.sum() == 0:
            return n_bounds

        threshold = self.background_check_std.get(col)
        if threshold is None:
            return n_bounds

        if int(valid.sum()) < _MIN_STATIONS_FOR_BACKGROUND_CHECK:
            LOG.warning(
                "CleanObservation: only %d station(s) report '%s' — too few "
                "for a robust background check (min %d); skipping it for "
                "this ref_time.",
                int(valid.sum()), col, _MIN_STATIONS_FOR_BACKGROUND_CHECK,
            )
            return n_bounds

        st_lat = df.loc[valid, "latitude"].to_numpy()
        st_lon = df.loc[valid, "longitude"].to_numpy()
        st_obs = df.loc[valid, col].to_numpy() + offset

        sta_x, sta_y = self._wgs84_to_lv95.transform(st_lon, st_lat)
        sta_xy = np.c_[sta_x, sta_y] / 1000.0
        _, gi = self._grid_tree.query(sta_xy, k=1)

        st_obs_lr = st_obs
        if shortname in self.lapse_rate_vars and "elevation" in df.columns:
            st_elev = df.loc[valid, "elevation"].to_numpy(dtype=float)
            elev_model_at_sta = self._icon_orog[gi]
            nan_mask = np.isnan(st_elev)
            if nan_mask.any():
                # No station elevation on record: fall back to the model's own
                # elevation at that cell, making the lapse-rate term a no-op
                # for just those stations rather than skipping the whole check.
                st_elev[nan_mask] = elev_model_at_sta[nan_mask]
            st_obs_lr = st_obs - self.lapse_rate * (elev_model_at_sta - st_elev)

        # background - obs, same convention as nudging.py's r_at_st.
        residual = B_flat[gi] - st_obs_lr

        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        scale = max(1.4826 * mad, self.reliability_eps)
        z = (residual - med) / scale
        flagged = np.abs(z) > threshold
        n_bg_flagged = int(flagged.sum())
        n_bg = 0
        if n_bg_flagged:
            fail_idx = df.index[valid][flagged]
            LOG.warning(
                "CleanObservation: %d station(s) failed the background check "
                "for '%s' (|z|>%.1f, network median residual=%.2f): %s",
                n_bg_flagged, col, threshold, med, fail_idx.tolist(),
            )
            n_bg = self._drop_flagged(df, col, fail_idx, "background check", protected_ids)

        return n_bounds + n_bg
