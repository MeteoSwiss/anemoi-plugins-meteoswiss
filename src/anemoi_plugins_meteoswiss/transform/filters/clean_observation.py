import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

# Make clean_observation_config and clean_observation_tests importable (bare imports inside them)
_FILTERS_DIR = str(Path(__file__).parent)
if _FILTERS_DIR not in sys.path:
    sys.path.insert(0, _FILTERS_DIR)

try:
    import clean_observation_config as _qc_config
    from clean_observation_tests import make_tests as _make_tests
    from clean_observation_tests import tests_summary as _tests_summary

    _QC_AVAILABLE = True
except ImportError as _e:
    LOG.warning("QC modules not available, cleaning will be a no-op: %s", _e)
    _QC_AVAILABLE = False

# Parquet column -> (QC parameter name, unit converter); K values pass through unchanged
_PARQUET_TO_QC = {
    "2t":   ("T_2M",    lambda x: x),
    "2d":   ("TD_2M",   lambda x: x),
    "vmax": ("VMAX10M", lambda x: x),
    "sp":   ("PS",      lambda x: x),  # Pa
    "msl":  ("PMSL",    lambda x: x),  # Pa
}
# FF_10M is derived from 10u/10v components — handled separately in _clean

# QC parameter -> parquet columns to mark as NaN when flagged
_QC_TO_PARQUET = {
    "T_2M":    ["2t"],
    "TD_2M":   ["2d"],
    "FF_10M":  ["10u", "10v"],
    "VMAX10M": ["vmax"],
    "PS":      ["sp"],
    "PMSL":    ["msl"],
}

# Tests that work without model/background data
_OBS_ONLY_TESTS = {"hard", "buddy_obs", "DWH_flag", "plateau_test"}


class CleanObservation(Filter):
    """Clean pre-fetched station observations and write the result to disk.

    Reads a Parquet file produced by RetrieveObservation, applies quality-
    control cleaning, and writes the cleaned DataFrame to a new Parquet file
    for use by NudgeTowardObservation.  The forecast fields are passed through
    unchanged.

    Parameters
    ----------
    obs_path_in : str
        Path to the raw observation Parquet file written by RetrieveObservation.
    obs_path_out : str
        Path where the cleaned observation Parquet file will be written.
    model_grib_path : str, optional
        Path to a model GRIB file (e.g. ``202501020600_0.grib``) whose fields
        are interpolated to station locations and used as the NWP background.
        When provided, ``buddy_diff`` and ``fgt`` QC tests are also run in
        addition to the obs-only tests.  HSURF is read from the same file if
        available (used by ``min_elev_diff`` interpolation).  If *None*
        (default) the NWP-dependent tests are skipped and the background frames
        are filled with NaN.
    model_interp : str, optional
        Grid-to-station interpolation strategy.  ``"nearest"`` (default) picks
        the closest grid point on the unit sphere.  ``"min_elev_diff"`` queries
        the 4 nearest grid points and picks the one whose HSURF elevation is
        closest to the station elevation (requires HSURF in the GRIB file).
    """

    def __init__(
        self,
        obs_path_in: str,
        obs_path_out: str,
        model_grib_path: str = None,
        model_interp: str = "nearest",
    ):
        if model_interp not in ("nearest", "min_elev_diff"):
            raise ValueError(
                f"model_interp must be 'nearest' or 'min_elev_diff', got {model_interp!r}"
            )
        self.obs_path_in = Path(obs_path_in)
        self.obs_path_out = Path(obs_path_out)
        self.model_grib_path = model_grib_path
        self.model_interp = model_interp
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Read, clean, and write observations; return forecast fields unchanged.

        Parameters
        ----------
        data : ekd.FieldList
            Forecast fields (passed through unchanged).

        Returns
        -------
        ekd.FieldList
            The input data, unchanged.
        """
        if not self.obs_path_in.exists():
            raise FileNotFoundError(f"Observation file not found: {self.obs_path_in}")

        df = pd.read_parquet(self.obs_path_in)
        LOG.info("Loaded %d stations from %s", len(df), self.obs_path_in)

        df = self._clean(df)

        self.obs_path_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.obs_path_out)
        LOG.info("Saved %d cleaned stations to %s", len(df), self.obs_path_out)

        flagged = getattr(self, "_flagged", [])
        json_path = self.obs_path_out.with_suffix("").with_name(
            self.obs_path_out.stem + "_flagged.json"
        )
        n_flagged_per_parameter: dict = {}
        for entry in flagged:
            para = entry.get("qc_parameter", "unknown")
            n_flagged_per_parameter[para] = n_flagged_per_parameter.get(para, 0) + 1
        output = {
            "duration_seconds": round(getattr(self, "_qc_duration", 0.0), 3),
            "n_flagged": len(flagged),
            "n_flagged_per_parameter": n_flagged_per_parameter,
            "flagged": flagged,
        }
        with open(json_path, "w") as fh:
            json.dump(output, fh, indent=2)
        LOG.info("Wrote %d flagged entries to %s", len(flagged), json_path)

        return data

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality-control cleaning to station observations.

        The method runs in two stages:

        1. **Automated QC tests** (``clean_observation_config.par2check``): for each parameter,
           the subset of active tests is run via ``make_tests``.  Without a
           model GRIB file only ``hard`` (range check) and ``buddy_obs``
           (spatial consistency) run.  When ``model_grib_path`` is set the
           model background is interpolated to station locations and
           ``buddy_diff`` and ``fgt`` are also activated.  The weighted score
           returned by ``tests_summary`` is compared against
           ``titan_ntests_threshold[para]['threshold_summary']``.  Stations
           whose score exceeds the threshold are blacklisted unless they appear
           in ``clean_observation_config.stations_excluded[para]``.

        2. **Hard blacklist** (``clean_observation_config.hard_blacklist``): a fixed list of
           station/parameter pairs that are always set to NaN regardless of
           QC scores or the exclusion list.

        Parameters
        ----------
        df : pd.DataFrame
            Raw station observations produced by ``RetrieveObservation``.
            Index is the station ``nat_abbr``; columns include the parquet
            variable columns (``2t``, ``2d``, ``10u``, ``10v``, ``vmax``, …)
            plus ``latitude``, ``longitude``, and ``altitude`` (m a.s.l.).
            Temperature values are in K, wind speeds in m/s.

        Returns
        -------
        pd.DataFrame
            Same DataFrame (same index and columns) with the values of
            suspected observations replaced by NaN.
        """
        # Diagnostics accumulated below; accessible as self._qc_diagnostics after the call.
        # Structure: {para: {"scores": {station: float}, "flagged": [str], "tests_run": [str],
        #                     "threshold": float, "blacklist": dict}}
        self._qc_diagnostics: dict = {}

        # Flagged observations: list of {"station", "column", "qc_parameter", "original_value"}
        self._flagged: list = []

        if not _QC_AVAILABLE:
            LOG.warning("QC modules unavailable, skipping cleaning")
            return df

        # --- Build df_qc: the view of df that make_tests expects ----------
        # Requires integer index, a 'sta_name' column, and QC parameter columns
        # (T_2M, TD_2M, FF_10M, VMAX10M) derived from the parquet columns.
        df_qc = pd.DataFrame()
        df_qc["sta_name"] = df.index.to_list()

        # Direct column mappings (T_2M, TD_2M, VMAX10M)
        for parquet_col, (qc_para, converter) in _PARQUET_TO_QC.items():
            if parquet_col in df.columns:
                df_qc[qc_para] = converter(df[parquet_col].to_numpy())

        # FF_10M is wind speed derived from the U/V components
        if "10u" in df.columns and "10v" in df.columns:
            df_qc["FF_10M"] = np.sqrt(df["10u"].to_numpy() ** 2 + df["10v"].to_numpy() ** 2)

        # Station coordinates and elevation from the parquet
        lats = df["latitude"].to_numpy(dtype=float)
        lons = df["longitude"].to_numpy(dtype=float)
        elevs = df["altitude"].to_numpy(dtype=float) if "altitude" in df.columns else np.zeros(len(df), dtype=float)
        stations = np.array(df.index.to_list())

        # Plausibility frame for DWH_flag test: index = station name, columns = *_pi
        pi_cols = [col for col in df.columns if col.endswith("_pi")]
        df_pi = df[pi_cols] if pi_cols else pd.DataFrame(index=df.index)

        # --- Model background: interpolate from GRIB when available ---------
        # NWP-dependent tests (buddy_diff, fgt) are only activated when a
        # model GRIB file is provided; otherwise only obs-only tests run.
        if self.model_grib_path is not None:
            try:
                df_mod, df_diff = self._load_model_at_stations(df_qc, lats, lons, elevs)
                active_tests = _OBS_ONLY_TESTS | {"buddy_diff", "fgt"}
                LOG.info("Model background loaded; extended test set active")
            except Exception as exc:
                LOG.warning(
                    "Failed to load model GRIB (%s); falling back to obs-only tests", exc
                )
                df_mod, df_diff = self._nan_frames(df_qc)
                active_tests = _OBS_ONLY_TESTS
        else:
            df_mod, df_diff = self._nan_frames(df_qc)
            active_tests = _OBS_ONLY_TESTS

        current_f = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        _t0 = time.monotonic()

        # --- Stage 1: automated QC tests per parameter --------------------
        for para in _qc_config.par2check:
            if para not in df_qc.columns:
                LOG.debug("Skipping %s: not available in parquet", para)
                continue

            all_tests = _qc_config.titan_ntests_threshold[para]["tests_QC"]
            all_weights = _qc_config.titan_ntests_threshold[para]["tests_QC_w"]
            tests_to_do = [t for t in all_tests if t in active_tests]
            if not tests_to_do:
                LOG.info("%-10s  no active tests (configured: %s)", para, all_tests)
                continue
            skipped = [t for t in all_tests if t not in active_tests]
            LOG.info("%-10s  running: %s", para, tests_to_do)
            if skipped:
                LOG.warning(
                    "%-10s  skipped %s — no model background (set --model-grib-path to enable)",
                    para, skipped,
                )
            weights = [all_weights[all_tests.index(t)] for t in tests_to_do]

            # Initialise the blacklist accumulator dict expected by make_tests
            my_dict = {"n": len(tests_to_do), "tests": tests_to_do}
            for test in tests_to_do:
                my_dict[test] = {"ID": [], "Station": [], "Time": [], "Parameter": []}

            blacklist, _ = _make_tests(
                current_f, df_qc, df_diff, df_mod, para,
                stations.copy(), lats.copy(), lons.copy(), elevs.copy(),
                0, 0, my_dict, tests_to_do,
                df_pi=df_pi,
                obs_path_in=self.obs_path_in,
            )

            # Compute a weighted score [0, 1] per station across all tests run
            qc_summary = _tests_summary(blacklist, weights, tests_to_do)
            threshold = _qc_config.titan_ntests_threshold[para]["threshold_summary"]

            flagged_stations = []

            # Stations in stations_excluded are never blacklisted for this parameter
            excluded = _qc_config.stations_excluded.get(para, {}).get("stations", [])
            for station in qc_summary.columns:
                if qc_summary[station].iloc[0] > threshold:
                    if station in excluded:
                        LOG.info("QC flagged %s for station %s but station is excluded — skipping",
                                 para, station)
                        continue
                    LOG.info("QC flagged %s for station %s (score=%.2f)",
                             para, station, qc_summary[station].iloc[0])
                    flagged_stations.append(station)
                    # For derived parameters (e.g. FF_10M from u/v) record the QC
                    # value that was actually tested, not the raw vector components.
                    sta_row = df_qc.index[df_qc["sta_name"] == station]
                    qc_val = (
                        float(df_qc.loc[sta_row[0], para])
                        if len(sta_row) and para in df_qc.columns
                        else None
                    )
                    # Which individual tests were positive for this station?
                    positive_tests = [
                        t for t in tests_to_do
                        if station in blacklist.get(t, {}).get("Station", [])
                    ]
                    for parquet_col in _QC_TO_PARQUET.get(para, []):
                        if parquet_col in df.columns and station in df.index:
                            self._flagged.append({
                                "station": station,
                                "column": parquet_col,
                                "qc_parameter": para,
                                "qc_value": qc_val,
                                "source": "qc_test",
                                "tests_positive": positive_tests,
                            })
                            df.loc[station, parquet_col] = np.nan

            self._qc_diagnostics[para] = {
                "scores": {col: float(qc_summary[col].iloc[0]) for col in qc_summary.columns},
                "flagged": flagged_stations,
                "tests_run": tests_to_do,
                "threshold": float(threshold),
                "blacklist": {
                    k: (v if not hasattr(v, "tolist") else v.tolist())
                    for k, v in blacklist.items()
                    if k not in ("n", "tests")
                },
            }

        # --- Stage 2: hard blacklist — always applied, ignores exclusions --
        for entry in _qc_config.hard_blacklist.values():
            station = entry["station"]
            if station not in df.index:
                continue
            for para in entry["paras"]:
                sta_row = df_qc.index[df_qc["sta_name"] == station]
                qc_val = (
                    float(df_qc.loc[sta_row[0], para])
                    if len(sta_row) and para in df_qc.columns
                    else None
                )
                for parquet_col in _QC_TO_PARQUET.get(para, []):
                    if parquet_col in df.columns:
                        self._flagged.append({
                            "station": station,
                            "column": parquet_col,
                            "qc_parameter": para,
                            "qc_value": qc_val,
                            "source": "hard_blacklist",
                        })
                        LOG.info("Hard blacklist: setting %s / %s to NaN", station, parquet_col)
                        df.loc[station, parquet_col] = np.nan

        self._qc_duration: float = time.monotonic() - _t0
        LOG.info("QC tests completed in %.2f s", self._qc_duration)
        return df

    @staticmethod
    def _nan_frames(df_qc: pd.DataFrame):
        """Return (df_mod, df_diff) filled with NaN for all non-meta columns."""
        df_mod = df_qc.copy()
        df_diff = df_qc.copy()
        for col in df_qc.columns:
            if col != "sta_name":
                df_mod[col] = np.nan
                df_diff[col] = np.nan
        return df_mod, df_diff

    def _load_model_at_stations(
        self,
        df_qc: pd.DataFrame,
        lats: np.ndarray,
        lons: np.ndarray,
        elevs: np.ndarray,
    ):
        """Interpolate model GRIB fields to station locations.

        Reads ``self.model_grib_path`` with earthkit.data, builds a spherical
        KD-tree over the model grid, then interpolates each QC parameter to
        station locations using the strategy chosen by ``self.model_interp``.
        HSURF is read from the same GRIB file when ``min_elev_diff`` is
        requested.

        Parameters
        ----------
        df_qc : pd.DataFrame
            QC observation frame (integer index, ``sta_name`` column, parameter
            columns).
        lats, lons, elevs : np.ndarray
            Station latitudes (°N), longitudes (°E), and elevations (m a.s.l.).

        Returns
        -------
        df_mod : pd.DataFrame
            Model values at station locations, same shape as *df_qc*.
        df_diff : pd.DataFrame
            Observation minus model background (``df_qc - df_mod``).
        """
        from scipy.spatial import cKDTree

        # QC parameter -> GRIB shortName candidates (first match wins)
        _QC_TO_GRIB = {
            "T_2M":    ["T_2M", "2t"],
            "TD_2M":   ["TD_2M", "2d"],
            "VMAX10M": ["VMAX_10M", "vmax"],
            # FF_10M derived from U/V — handled separately below
        }
        _U_SHORTNAMES    = ["U_10M", "10u"]
        _V_SHORTNAMES    = ["V_10M", "10v"]
        _HSURF_SHORTNAMES = ["HSURF", "z"]

        LOG.info("Reading model GRIB: %s", self.model_grib_path)
        fs = ekd.from_source("file", self.model_grib_path)

        # Build unit-sphere KD-tree from the model grid (first field sets the grid)
        ll = fs[0].to_latlon()
        grid_lats = np.asarray(ll["lat"]).flatten()
        grid_lons = np.asarray(ll["lon"]).flatten()

        def _xyz(lat_deg, lon_deg):
            lat_r = np.deg2rad(lat_deg)
            lon_r = np.deg2rad(lon_deg)
            return np.stack(
                [np.cos(lat_r) * np.cos(lon_r),
                 np.cos(lat_r) * np.sin(lon_r),
                 np.sin(lat_r)],
                axis=-1,
            )

        tree = cKDTree(_xyz(grid_lats, grid_lons))

        def _first_field_values(shortnames):
            """Return flattened values array for the first matching shortName."""
            for sn in shortnames:
                try:
                    sel = fs.sel(shortName=sn)
                    if len(sel) > 0:
                        return np.asarray(sel[0].values).flatten()
                except Exception:
                    pass
            return None

        # Determine the grid index for each station
        if self.model_interp == "min_elev_diff":
            _, indices = tree.query(_xyz(lats, lons), k=4)  # (n_sta, 4)

            # Try to read HSURF for elevation-guided selection
            grid_hsurf = _first_field_values(_HSURF_SHORTNAMES)
            if grid_hsurf is None:
                LOG.warning(
                    "HSURF not found in %s; falling back to nearest for min_elev_diff",
                    self.model_grib_path,
                )
                station_idx = indices[:, 0]
            else:
                LOG.debug("HSURF loaded (%d grid points)", len(grid_hsurf))
                station_idx = np.array([
                    nbrs[np.argmin(np.abs(grid_hsurf[nbrs] - sta_elev))]
                    for nbrs, sta_elev in zip(indices, elevs)
                ])
        else:  # "nearest"
            _, station_idx = tree.query(_xyz(lats, lons), k=1)  # (n_sta,)

        # Build df_mod — start with NaN and fill where GRIB fields exist
        df_mod, _ = self._nan_frames(df_qc)

        for para, shortnames in _QC_TO_GRIB.items():
            if para not in df_qc.columns:
                continue
            vals = _first_field_values(shortnames)
            if vals is not None:
                df_mod[para] = vals[station_idx]
                LOG.debug("Interpolated %s to %d stations", para, len(station_idx))

        # FF_10M from U/V wind components
        if "FF_10M" in df_qc.columns:
            u_vals = _first_field_values(_U_SHORTNAMES)
            v_vals = _first_field_values(_V_SHORTNAMES)
            if u_vals is not None and v_vals is not None:
                df_mod["FF_10M"] = np.sqrt(
                    u_vals[station_idx] ** 2 + v_vals[station_idx] ** 2
                )
                LOG.debug("Derived FF_10M (U/V) for %d stations", len(station_idx))

        # df_diff = obs - model background (innovation vector)
        df_diff = df_qc.copy()
        for col in df_qc.columns:
            if col != "sta_name":
                df_diff[col] = df_qc[col].to_numpy() - df_mod[col].to_numpy()

        return df_mod, df_diff
