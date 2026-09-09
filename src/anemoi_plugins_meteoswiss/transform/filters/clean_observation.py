import json
import logging
import re
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

# Make clean_observation_tests importable via bare import
_FILTERS_DIR = str(Path(__file__).parent)
if _FILTERS_DIR not in sys.path:
    sys.path.insert(0, _FILTERS_DIR)


def _load_qc_config() -> SimpleNamespace:
    """Load QC configuration from clean_observation_config.yaml."""
    cfg = yaml.safe_load(Path(__file__).with_name("clean_observation_config.yaml").read_text())

    def _identity(x):
        return x

    ns = SimpleNamespace(**cfg)
    ns.obs_only_tests = set(cfg["obs_only_tests"])
    ns.parquet_to_qc = {col: (qp, _identity) for col, qp in cfg["parquet_to_qc"].items()}
    return ns


try:
    _qc_config = _load_qc_config()
    from clean_observation_tests import (
        DWH_flag as _DWH_flag,
        buddy_check as _buddy_check,
        first_guess_test as _first_guess_test,
        hard_test as _hard_test,
        isolation_check as _isolation_check,
        plateau_test as _plateau_test,
        spacial_ct_dual as _spacial_ct_dual,
        spacial_ct_resistant as _spacial_ct_resistant,
        tests_summary as _tests_summary,
    )

    _QC_AVAILABLE = True
except (ImportError, FileNotFoundError) as _e:
    LOG.warning("QC modules not available, cleaning will be a no-op: %s", _e)
    _QC_AVAILABLE = False


class CleanObservation(Filter):
    """Apply titanlib QC to station observations and write the cleaned parquet to disk.

    Reads a parquet file produced by ``RetrieveObservation``, runs per-parameter
    QC tests configured in ``clean_observation_config``, sets suspected values to
    NaN, and saves the result.  Forecast fields passed via ``forward()`` are
    returned unchanged.

    After ``forward()`` or ``_clean()`` completes the following attributes are set:

    ``_flagged`` : list of dict
        One entry per (station, parquet column) pair whose value was set to NaN,
        with keys ``station``, ``column``, ``qc_parameter``, ``qc_value``,
        ``source`` (``"qc_test"`` or ``"hard_blacklist"``), and
        ``tests_positive`` (list of test names that voted to flag; qc_test only).
    ``_qc_diagnostics`` : dict
        Per-parameter diagnostic information: scores per station, list of flagged
        stations, tests run, threshold, and the raw per-test blacklist.
    ``_tests_done`` : dict
        Per-parameter list of test names that were actually executed.
    ``_qc_duration`` : float
        Wall-clock time in seconds for the QC stage.

    Parameters
    ----------
    obs_path_in : str
        Path to the raw observation parquet file written by ``RetrieveObservation``.
    obs_path_out : str
        Destination path for the cleaned observation parquet file.
    model_grib_path : str, optional
        Path to a model GRIB file whose fields are interpolated to station
        locations and used as the NWP background.  When provided, the
        model-dependent tests ``buddy_diff`` and ``fgt`` are activated in
        addition to the obs-only tests.  If *None* (default) those tests are
        skipped and background frames are filled with NaN.
    model_interp : str, optional
        Grid-to-station interpolation method: ``"nearest"`` (default) picks the
        closest grid point on the unit sphere; ``"min_elev_diff"`` queries the 4
        nearest points and selects the one whose HSURF elevation is closest to
        the station elevation (requires HSURF in the GRIB file).
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Run QC, write cleaned parquet + flagged JSON, optionally produce maps.

        Reads ``obs_path_in``, calls ``_clean()``, writes the result to
        ``obs_path_out`` (dropping internal ``*_pi`` columns), and saves a
        ``*_flagged.json`` summary alongside the parquet.  If
        ``clean_observation_config.plot_maps`` is *True*, per-parameter PNG
        maps are also written to the same directory.

        Parameters
        ----------
        data : ekd.FieldList
            Forecast fields — passed through unchanged.

        Returns
        -------
        ekd.FieldList
            The input *data*, unchanged.
        """
        if not self.obs_path_in.exists():
            raise FileNotFoundError(f"Observation file not found: {self.obs_path_in}")

        df = pd.read_parquet(self.obs_path_in)
        LOG.info("Loaded %d stations from %s", len(df), self.obs_path_in)

        df = self._clean(df)

        self.obs_path_out.parent.mkdir(parents=True, exist_ok=True)
        # Drop _pi columns (DWH plausibility values) from output — they are only
        # needed internally during QC and should not propagate downstream.
        pi_cols = [c for c in df.columns if c.endswith("_pi")]
        df.to_parquet(self.obs_path_out, columns=[c for c in df.columns if c not in pi_cols])
        LOG.info("Saved %d cleaned stations to %s", len(df), self.obs_path_out)

        ts_match = re.search(r'\d{12}', self.obs_path_in.stem)
        obs_timestamp = ts_match.group() if ts_match else None
        json_path = self.obs_path_out.with_name(self.obs_path_out.stem + "_flagged.json")
        self._write_flagged_json(
            json_path,
            getattr(self, "_flagged", []),
            getattr(self, "_tests_done", {}),
            getattr(self, "_qc_duration", 0.0),
            obs_timestamp,
        )

        if getattr(_qc_config, "plot_maps", False):
            try:
                self._plot_station_maps(df)
            except Exception:
                LOG.exception("QC station map generation failed")

        return data

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply QC tests and set suspected values to NaN.

        Two stages:

        1. **Automated QC** (``par2check`` parameters): for each parameter the
           configured tests are run via ``make_tests``.  Obs-only tests
           (``hard``, ``isolation_check``, ``buddy_obs``, ``DWH_flag``,
           ``plateau_test``) always run; model-dependent tests (``buddy_diff``,
           ``fgt``) are added only when ``model_grib_path`` is set.  Each test
           returns a blacklist; the weighted score from ``tests_summary`` is
           compared against ``titan_ntests_threshold[para]["threshold_summary"]``.
           Stations above the threshold are set to NaN unless listed in
           ``stations_excluded[para]``.

        2. **Hard blacklist** (``hard_blacklist``): station/parameter pairs
           permanently set to NaN regardless of QC scores or exclusions.

        Populates ``self._flagged``, ``self._qc_diagnostics``,
        ``self._tests_done``, and ``self._qc_duration``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw observations from ``RetrieveObservation``.  Index is station
            ``nat_abbr``; columns include parquet variable columns (``2t``,
            ``2d``, ``10u``, ``10v``, ``vmax``, …), ``latitude``,
            ``longitude``, and ``altitude`` (m a.s.l.).

        Returns
        -------
        pd.DataFrame
            Same DataFrame with suspected observation values replaced by NaN.
        """
        # Diagnostics accumulated below; accessible as self._qc_diagnostics after the call.
        # Structure: {para: {"scores": {station: float}, "flagged": [str], "tests_run": [str],
        #                     "threshold": float, "blacklist": dict}}
        self._qc_diagnostics: dict = {}
        self._tests_done: dict = {}

        # Flagged observations: list of {"station", "column", "qc_parameter", "original_value"}
        self._flagged: list = []

        if not _QC_AVAILABLE:
            LOG.warning("QC modules unavailable, skipping cleaning")
            return df

        df_qc, lats, lons, elevs, stations, df_pi = self._build_df_qc(df)
        df_mod, df_diff, active_tests = self._resolve_model_background(df_qc, lats, lons, elevs)

        current_f = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        _t0 = time.monotonic()

        # --- Stage 1: automated QC tests per parameter --------------------
        for para in _qc_config.par2check:
            if para not in df_qc.columns:
                LOG.warning("Skipping %s: not available in parquet", para)
                continue

            all_tests = _qc_config.titan_ntests_threshold[para]["tests_QC"]
            all_weights = _qc_config.titan_ntests_threshold[para]["tests_QC_w"]
            tests_to_do = [t for t in all_tests if t in active_tests]
            if not tests_to_do:
                LOG.info("%-10s  no active tests (configured: %s)", para, all_tests)
                continue
            # tests not in active_tests are model-dependent (buddy_diff, fgt) and
            # are skipped when no model background is available.
            skipped = [t for t in all_tests if t not in active_tests]
            LOG.info("%-10s  running: %s", para, tests_to_do)
            if skipped:
                LOG.warning(
                    "%-10s  skipped %s — no model background (set --model-grib-path to enable)",
                    para, skipped,
                )

            # Blacklist accumulator expected by make_tests:
            #   'n'    : number of planned tests
            #   'tests': list of planned test names
            #   <name> : per-test sub-dict with lists 'ID', 'Station', 'Time', 'Parameter'
            my_dict = {"n": len(tests_to_do), "tests": tests_to_do}
            for test in tests_to_do:
                my_dict[test] = {"ID": [], "Station": [], "Time": [], "Parameter": []}

            blacklist, _, executed_tests = self.make_tests(
                current_f, df_qc, df_diff, df_mod, para,
                stations.copy(), lats.copy(), lons.copy(), elevs.copy(),
                0, 0, my_dict, tests_to_do,
                df_pi=df_pi,
                obs_path_in=self.obs_path_in,
            )

            # executed_tests may be shorter than tests_to_do (e.g. plateau_test skipped)
            # or longer (isolation_check always runs but is not in tests_QC).
            # Score uses only the tests that are in tests_QC (have configured weights).
            # isolation_check stations are flagged directly after the score loop.
            score_tests = [t for t in executed_tests if t in all_tests]
            self._tests_done[para] = executed_tests
            blacklist['tests'] = score_tests
            blacklist['n'] = len(score_tests)
            weights = [all_weights[all_tests.index(t)] for t in score_tests]

            # Score = sum(weight_i for flagging tests) / n_score_tests.
            # A station is blacklisted when score > threshold_summary (default 0.2).
            qc_summary = _tests_summary(blacklist, weights, score_tests)
            threshold = _qc_config.titan_ntests_threshold[para]["threshold_summary"]

            flagged_stations = []

            # stations_excluded: trusted stations (e.g. reference stations) that are
            # never blacklisted by automated QC, regardless of their score.
            excluded = _qc_config.stations_excluded.get(para, {}).get("stations", [])
            for station in qc_summary.columns:
                score = qc_summary[station].iloc[0]
                if score <= threshold:
                    continue
                if station in excluded:
                    LOG.info("QC flagged %s for station %s but station is excluded — skipping",
                             para, station)
                    continue
                LOG.info("QC flagged %s for station %s (score=%.2f)", para, station, score)
                flagged_stations.append(station)
                # Which individual tests voted to flag this station?
                positive_tests = [
                    t for t in executed_tests
                    if station in blacklist.get(t, {}).get("Station", [])
                ]
                # A QC parameter may map to multiple parquet columns
                # (e.g. SP_10M → 10u and 10v); set all of them to NaN.
                self._flag_station_columns(df, df_qc, station, para, "qc_test",
                                           tests_positive=positive_tests)

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

        # --- Stage 2: hard blacklist — always applied, ignores exclusions -------
        # Stations here are permanently unreliable for specific parameters and are
        # always set to NaN regardless of QC score or stations_excluded membership.
        for entry in _qc_config.hard_blacklist.values():
            station = entry["station"]
            if station not in df.index:
                continue
            for para in entry["paras"]:
                LOG.info("Hard blacklist: setting %s / %s to NaN", station, para)
                self._flag_station_columns(df, df_qc, station, para, "hard_blacklist")

        self._qc_duration: float = time.monotonic() - _t0
        LOG.info("QC tests completed in %.2f s", self._qc_duration)
        return df

    def make_tests(
        self,
        current_f,
        df_obs,
        df_diff,
        df_mod,
        para,
        stations,
        lats,
        lons,
        elevs,
        ii,
        jj,
        my_dict,
        tests_to_do,
        df_pi=None,
        obs_path_in=None,
    ):
        """Run the requested QC tests for a single parameter and accumulate flagged stations.

        Args:
            current_f (str): Wall-clock timestamp ('%Y%m%d%H%M') used as the Time label in blacklist entries.
            df_obs (DataFrame): Current observation DataFrame (must contain 'sta_name' and parameter columns).
            df_diff (DataFrame): Obs-minus-model difference DataFrame (same shape as df_obs).
            df_mod (DataFrame): Model background DataFrame (same shape as df_obs).
            para (str): QC parameter name (e.g. 'T_2M', 'PS').
            stations (ndarray): Station name array (NaN rows pre-removed).
            lats (ndarray): Station latitudes.
            lons (ndarray): Station longitudes.
            elevs (ndarray): Station elevations (m).
            ii (int): Unused index (kept for interface compatibility).
            jj (int): Unused index (kept for interface compatibility).
            my_dict (dict): Accumulator dict with keys 'tests', 'n', and one sub-dict per test name,
                each containing lists 'ID', 'Station', 'Time', 'Parameter'.
            tests_to_do (list[str]): Subset of tests to run, e.g. ['hard', 'buddy_obs', 'plateau_test'].
                Model-dependent tests ('buddy_diff', 'fgt') are included only when a model is available.
            df_pi (DataFrame, optional): Plausibility frame (columns = '*_pi') used by DWH_flag.
            obs_path_in (str | Path, optional): Path to the current observation parquet file,
                used by plateau_test to locate historical files.

        Returns:
            tuple: (my_dict, freq, executed_tests) where my_dict is the updated accumulator,
                freq is the plausibility value-counts Series from DWH_flag (empty DataFrame
                otherwise), and executed_tests is the list of test names that were actually run.
        """
        values = df_obs[para].iloc[:].to_numpy()
        freq = pd.DataFrame()
        executed_tests = []

        ind = np.argwhere(np.isnan(values))
        values = np.delete(values, ind)
        stations = np.delete(stations, ind)
        lats = np.delete(lats, ind)
        lons = np.delete(lons, ind)
        elevs = np.delete(elevs, ind)
        mods = np.delete(df_mod[para].iloc[:].to_numpy(), ind)
        diffs = np.delete(df_diff[para].iloc[:].to_numpy(), ind)
        LOG.info('make_tests %s: %d stations (non-NaN)', para, len(stations))

        runners = self._build_test_runners(
            tests_to_do, df_obs, values, diffs, mods,
            stations, lats, lons, elevs,
            para, current_f, df_pi, obs_path_in,
        )

        for test in tests_to_do:
            if test not in runners:
                continue

            raw = runners[test]()

            if test == "DWH_flag":
                blacklist, freq = raw
            elif test == "plateau_test":
                if raw is None:
                    LOG.warning('plateau_test skipped — no historical files available')
                    continue
                blacklist = raw
            else:
                blacklist = raw

            self._merge_blacklist(my_dict, test, blacklist)
            n = len((blacklist or {}).get("Station", []))
            LOG.info('%-18s %s blacklisted stations: %d', test, para, n)
            executed_tests.append(test)

        return my_dict, freq, executed_tests

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_df_qc(self, df: pd.DataFrame):
        """Build the QC observation frame and supporting arrays from the raw parquet DataFrame.

        Returns
        -------
        df_qc : pd.DataFrame
            Integer-indexed frame with ``sta_name`` and QC parameter columns.
        lats, lons, elevs : np.ndarray
            Station latitudes (°N), longitudes (°E), and elevations (m a.s.l.).
        stations : np.ndarray
            Station name array aligned with df rows.
        df_pi : pd.DataFrame
            Plausibility frame (``*_pi`` columns only); empty if none present.
        """
        df_qc = pd.DataFrame()
        df_qc["sta_name"] = df.index.to_list()

        # Direct column mappings (T_2M, TD_2M, VMAX10M)
        for parquet_col, (qc_para, converter) in _qc_config.parquet_to_qc.items():
            if parquet_col in df.columns:
                df_qc[qc_para] = converter(df[parquet_col].to_numpy())

        # SP_10M is wind speed derived from the U/V components; the scalar speed
        # is what titanlib spatial tests operate on (not the vector components).
        if "10u" in df.columns and "10v" in df.columns:
            df_qc["SP_10M"] = np.sqrt(df["10u"].to_numpy() ** 2 + df["10v"].to_numpy() ** 2)

        lats = df["latitude"].to_numpy(dtype=float)
        lons = df["longitude"].to_numpy(dtype=float)
        # altitude may be missing for synthetic test data; default to 0 m a.s.l.
        elevs = (
            df["altitude"].to_numpy(dtype=float)
            if "altitude" in df.columns
            else np.zeros(len(df), dtype=float)
        )
        stations = np.array(df.index.to_list())

        # Plausibility frame for DWH_flag: index = station name, columns = *_pi.
        # Each *_pi column contains a DWH plausibility value (0–1) pre-fetched by
        # RetrieveObservation.  Stations with pi < dwh_plausibility_thr are flagged.
        pi_cols = [col for col in df.columns if col.endswith("_pi")]
        df_pi = df[pi_cols] if pi_cols else pd.DataFrame(index=df.index)

        return df_qc, lats, lons, elevs, stations, df_pi

    def _resolve_model_background(self, df_qc: pd.DataFrame, lats, lons, elevs):
        """Load the NWP background and decide which test set is active.

        Returns
        -------
        df_mod : pd.DataFrame
            Model values at station locations (NaN when no GRIB is available).
        df_diff : pd.DataFrame
            Observation innovation ``df_qc - df_mod`` (NaN when no GRIB is available).
        active_tests : set[str]
            ``obs_only_tests`` when no model is loaded; extended with
            ``{"buddy_diff", "fgt"}`` when the model background is available.
        """
        if self.model_grib_path is not None:
            try:
                df_mod, df_diff = self._load_model_at_stations(df_qc, lats, lons, elevs)
                LOG.info("Model background loaded; extended test set active")
                return df_mod, df_diff, _qc_config.obs_only_tests | {"buddy_diff", "fgt"}
            except Exception as exc:
                LOG.warning(
                    "Failed to load model GRIB (%s); falling back to obs-only tests", exc
                )
        df_mod, df_diff = self._nan_frames(df_qc)
        return df_mod, df_diff, _qc_config.obs_only_tests

    def _flag_station_columns(self, df, df_qc, station, para, source, **extra):
        """Set all parquet columns for *(station, para)* to NaN and record each flag.

        Parameters
        ----------
        df : pd.DataFrame
            Raw observation DataFrame (modified in-place).
        df_qc : pd.DataFrame
            QC frame used to read the QC-space value for logging.
        station : str
            Station ``nat_abbr`` to flag.
        para : str
            QC parameter name (e.g. ``"T_2M"``).
        source : str
            ``"qc_test"`` or ``"hard_blacklist"``.
        **extra
            Additional keys appended to each ``_flagged`` entry
            (e.g. ``tests_positive=[...]`` for ``"qc_test"``).
        """
        # Record the QC-space value (e.g. wind speed scalar, not u/v)
        # that triggered the flag, for traceability in the JSON output.
        sta_row = df_qc.index[df_qc["sta_name"] == station]
        qc_val = (
            float(df_qc.loc[sta_row[0], para])
            if len(sta_row) and para in df_qc.columns
            else None
        )
        for parquet_col in _qc_config.qc_to_parquet.get(para, []):
            if parquet_col in df.columns and station in df.index:
                self._flagged.append({
                    "station": station,
                    "column": parquet_col,
                    "qc_parameter": para,
                    "qc_value": qc_val,
                    "source": source,
                    **extra,
                })
                df.loc[station, parquet_col] = np.nan

    @staticmethod
    def _merge_blacklist(my_dict, test_name, blacklist):
        """Append a test's flagged stations into the accumulator dict."""
        if not blacklist:
            return
        nb = len(my_dict[test_name]["ID"])
        for k in range(len(blacklist["Station"])):
            nb += 1
            my_dict[test_name]["ID"].append(nb)
            my_dict[test_name]["Station"].append(blacklist["Station"][k])
            my_dict[test_name]["Time"].append(blacklist["Time"][k])
            my_dict[test_name]["Parameter"].append(blacklist["Parameter"][k])

    @classmethod
    def _spatial_fns(cls):
        """Return the spatial test function implementations to use.

        Subclasses override this to swap in alternative implementations
        (e.g. pure-Python replacements that do not require titanlib).
        """
        return {
            "buddy_check":       _buddy_check,
            "first_guess_test":  _first_guess_test,
            "isolation_check":   _isolation_check,
            "spacial_ct_dual":   _spacial_ct_dual,
            "spacial_ct_resistant": _spacial_ct_resistant,
        }

    @classmethod
    def _build_test_runners(
        cls, tests_to_do, df_obs, values, diffs, mods,
        stations, lats, lons, elevs,
        para, current_f, df_pi, obs_path_in,
    ):
        """Return a dict mapping each requested test name to a zero-arg callable.

        Only runners for tests present in *tests_to_do* are built, so config
        dicts are only accessed for tests that will actually run.
        Spatial functions are resolved via ``_spatial_fns`` so subclasses can
        swap implementations without overriding this method.
        """
        cfg = _qc_config
        fns = cls._spatial_fns()
        runners = {}

        if "hard" in tests_to_do:
            ind_var = cfg.obs_variables.index(para)
            pch_min = cfg.plausibility_thresholds['pch_min'][ind_var]
            pch_max = cfg.plausibility_thresholds['pch_max'][ind_var]
            runners["hard"] = lambda: _hard_test(df_obs, pch_min, pch_max, para, current_f)

        if "buddy_obs" in tests_to_do:
            b = cfg.buddy[para]
            runners["buddy_obs"] = lambda: fns["buddy_check"](
                stations, lats, lons, elevs, values, para, current_f,
                b["threshold"], b["max_elev_diff"], b["elev_gradient"],
                b["min_std"], b["num_iterations"], b["num_min"], b["radius"],
            )

        if "buddy_diff" in tests_to_do:
            bd = cfg.buddy_diff[para]
            runners["buddy_diff"] = lambda: fns["buddy_check"](
                stations, lats, lons, elevs, diffs, para, current_f,
                bd["threshold"], bd["max_elev_diff"], bd["elev_gradient"],
                bd["min_std"], bd["num_iterations"], bd["num_min"], bd["radius"],
            )

        if "fgt" in tests_to_do:
            fg = cfg.fgt[para]
            runners["fgt"] = lambda: fns["first_guess_test"](
                stations, lats, lons, elevs, values, mods, para, current_f,
                fg['background_elab_type'], fg['num_min_outer'], fg['num_max_outer'],
                fg['inner_radius'], fg['outer_radius'], fg['num_iterations'],
                fg['num_min_prof'], fg['min_elev_diff'], fg['min_horizontal_scale'],
                fg['max_horizontal_scale'], fg['kth_closest_obs_horizontal_scale'],
                bool(fg['debug']), bool(fg['basic']), fg['tpostneg'],
            )

        if "spt_resistant" in tests_to_do:
            sp = cfg.spt_resistant[para]
            runners["spt_resistant"] = lambda: fns["spacial_ct_resistant"](
                stations, lats, lons, elevs, values, para, current_f,
                sp['background_elab_type'], sp['num_min_outer'], sp['num_max_outer'],
                sp['inner_radius'], sp['outer_radius'], sp['num_iterations'],
                sp['num_min_prof'], sp['min_elev_diff'], sp['min_horizontal_scale'],
                sp['max_horizontal_scale'], sp['kth_closest_obs_horizontal_scale'],
                sp['vertical_scale'], sp['debug'], sp['basic'],
            )

        if "spt_dual" in tests_to_do:
            sd = cfg.sct_dual[para]
            runners["spt_dual"] = lambda: fns["spacial_ct_dual"](
                stations, lats, lons, elevs, values, para, current_f,
                sd['num_min_outer'], sd['num_max_outer'],
                sd['inner_radius'], sd['outer_radius'], sd['num_iterations'],
                sd['min_horizontal_scale'], sd['max_horizontal_scale'],
                sd['kth_closest_obs_horizontal_scale'], sd['vertical_scale'],
                bool(sd['debug']), sd['condition'],
                float(sd['event_thresholds']), float(sd['test_thresholds']),
            )

        if "DWH_flag" in tests_to_do:
            pi_col = cfg.par2pi.get(para, "")
            if df_pi is not None and pi_col and pi_col in df_pi.columns:
                pla_series = df_pi[pi_col]
            else:
                LOG.warning('DWH_flag: no pi column for %s, skipping', para)
                pla_series = pd.Series(dtype=float)
            runners["DWH_flag"] = lambda: _DWH_flag(current_f, para, stations, pla_series)

        if "plateau_test" in tests_to_do:
            pt = cfg.plateau_test[para]
            runners["plateau_test"] = lambda: _plateau_test(
                df_obs, pt['window'], pt['sd'], para, current_f,
                obs_path_in=obs_path_in, gran_minutes=pt['gran'],
            )

        if "isolation_check" in tests_to_do:
            ic = cfg.isolation_check[para]
            runners["isolation_check"] = lambda: fns["isolation_check"](
                stations, lats, lons, elevs, para, current_f,
                ic['num_min'], ic['radius'],
            )

        return runners

    @staticmethod
    def _write_flagged_json(json_path, flagged, tests_done, qc_duration, obs_timestamp):
        """Write the QC summary JSON to *json_path*."""
        n_per_para: dict = {}
        for entry in flagged:
            para = entry.get("qc_parameter", "unknown")
            n_per_para[para] = n_per_para.get(para, 0) + 1

        output = {
            "obs_timestamp": obs_timestamp,
            "duration_seconds": round(qc_duration, 3),
            "tests_done": tests_done,
            "n_flagged": len(flagged),
            "n_flagged_per_parameter": n_per_para,
            "flagged": [
                {k: round(v, 2) if isinstance(v, float) else v for k, v in e.items()}
                for e in flagged
            ],
        }
        with open(json_path, "w") as fh:
            json.dump(output, fh, indent=2)
            fh.write("\n")
        LOG.info("Wrote %d flagged entries to %s", len(flagged), json_path)

    @staticmethod
    def _nan_frames(df_qc: pd.DataFrame):
        """Return ``(df_mod, df_diff)`` with the same shape as *df_qc* but all parameter columns set to NaN."""
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
        """Interpolate model GRIB fields to station locations and compute obs-minus-model.

        Reads ``self.model_grib_path`` with earthkit.data, builds a spherical
        KD-tree, and interpolates each QC parameter to station locations using
        the strategy in ``self.model_interp``.

        Parameters
        ----------
        df_qc : pd.DataFrame
            QC observation frame (integer index, ``sta_name`` column, QC parameter columns).
        lats, lons, elevs : np.ndarray
            Station latitudes (°N), longitudes (°E), and elevations (m a.s.l.).

        Returns
        -------
        df_mod : pd.DataFrame
            Model values interpolated to station locations, same shape as *df_qc*.
        df_diff : pd.DataFrame
            Observation innovation: ``df_qc - df_mod`` (used by ``buddy_diff`` and ``fgt``).
        """
        from scipy.spatial import cKDTree

        # QC parameter -> GRIB shortName candidates (first match wins)
        _QC_TO_GRIB = {
            "T_2M":    ["T_2M", "2t"],
            "TD_2M":   ["TD_2M", "2d"],
            "VMAX10M": ["VMAX_10M", "vmax"],
            # SP_10M derived from U/V — handled separately below
        }
        _U_SHORTNAMES    = ["U_10M", "10u"]
        _V_SHORTNAMES    = ["V_10M", "10v"]
        _HSURF_SHORTNAMES = ["HSURF", "z"]

        LOG.info("Reading model GRIB: %s", self.model_grib_path)
        fs = ekd.from_source("file", self.model_grib_path)

        # Build a KD-tree on the unit sphere (3-D Cartesian coords) rather than
        # lat/lon directly, to avoid discontinuities at the date-line and poles.
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

        # For each station, pick the best model grid point.
        # min_elev_diff: among the 4 nearest points, take the one whose HSURF
        # elevation is closest to the station elevation.  This reduces the
        # temperature bias caused by interpolating across steep orography.
        if self.model_interp == "min_elev_diff":
            _, indices = tree.query(_xyz(lats, lons), k=4)  # (n_sta, 4)

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

        # SP_10M from U/V wind components
        if "SP_10M" in df_qc.columns:
            u_vals = _first_field_values(_U_SHORTNAMES)
            v_vals = _first_field_values(_V_SHORTNAMES)
            if u_vals is not None and v_vals is not None:
                df_mod["SP_10M"] = np.sqrt(
                    u_vals[station_idx] ** 2 + v_vals[station_idx] ** 2
                )
                LOG.debug("Derived SP_10M (U/V) for %d stations", len(station_idx))

        # df_diff = obs - model background (observation innovation).
        # buddy_diff and fgt use this instead of raw obs values so that
        # the NWP climatology is removed before spatial consistency checks.
        df_diff = df_qc.copy()
        for col in df_qc.columns:
            if col != "sta_name":
                df_diff[col] = df_qc[col].to_numpy() - df_mod[col].to_numpy()

        return df_mod, df_diff

    def _plot_station_maps(self, df: pd.DataFrame) -> None:
        """Save per-parameter QC PNG maps to the same directory as the output parquet.

        Delegates to ``prepare_and_plot_station_maps`` in ``clean_observation_plot``.
        Produces two PNG files per parameter (full domain and Switzerland zoom).
        """
        LOG.info("Generating QC station maps in %s", self.obs_path_out.parent)
        from clean_observation_plot import prepare_and_plot_station_maps
        prepare_and_plot_station_maps(
            getattr(self, "_flagged", []),
            getattr(self, "_qc_diagnostics", {}),
            df,
            self.obs_path_out,
            _qc_config.par2check,
            _qc_config.qc_to_parquet,
            _qc_config.parquet_to_qc,
        )
