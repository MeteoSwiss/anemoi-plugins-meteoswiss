import json
import logging
import math
import re
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import earthkit.data as ekd
import numpy as np
import pandas as pd
import yaml
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

# Tests that run without model/background data
_OBS_ONLY_TESTS = {"hard", "isolation_check", "buddy_obs", "DWH_flag", "plateau_test"}

# QC parameter → parquet plausibility column
_PAR2PI = {
    "T_2M": "2t_pi",
    "TD_2M": "2d_pi",
    "SP_10M": "ff_pi",
    "VMAX10M": "vmax_pi",
    "PS": "sp_pi",
    "PMSL": "msl_pi",
    "T_G": "skt_pi",
}

# parquet column → QC parameter (SP_10M derived from 10u/10v separately)
_PARQUET_TO_QC = {
    "2t": "T_2M",
    "2d": "TD_2M",
    "vmax": "VMAX10M",
    "sp": "PS",
    "msl": "PMSL",
    "skt": "T_G",
}

# QC parameter → parquet columns to set NaN when flagged
_QC_TO_PARQUET = {
    "T_2M": ["2t"],
    "TD_2M": ["2d"],
    "SP_10M": ["10u", "10v"],
    "VMAX10M": ["vmax"],
    "PS": ["sp"],
    "PMSL": ["msl"],
    "T_G": ["skt"],
}

_SEASON_MONTHS = {
    "winter": {12, 1, 2},
    "spring": {3, 4, 5},
    "fall": {9, 10, 11},
    "summer": {6, 7, 8},
}
_BUDDY_TESTS = {"buddy_obs", "buddy_diff"}
_INVERSION_BLOCKING_TESTS = {"hard", "isolation_check", "plateau_test", "DWH_flag"}

# Seasonal altitude-dependent hard-test thresholds.
# Config stores sea-level values for the most lenient month (winter for pch_min,
# summer for pch_max). Monthly deltas (K) tighten the bounds for each month,
# and the lapse-rate correction shifts them with altitude.
#
# Anchors used to derive the deltas:
#   pch_min: Jan/Feb at sea level ≈ 244 K, Nov at sea level ≈ 264 K, Jul ≈ 272 K
#   pch_max: Aug at sea level = config value, Jan/Feb ≈ config − 35 K
# pch_min uses dry-adiabatic-like lapse (steeper): cold pools form readily at altitude.
# pch_max uses moist-adiabatic lapse (ICAO standard): warm events follow slower gradient.
_HARD_LAPSE_MIN = 8.3  # K per 1000 m — lapse rate for pch_min altitude correction
_HARD_LAPSE_MAX = 6.5  # K per 1000 m — lapse rate for pch_max altitude correction
_HARD_MIN_FLOOR = 223.0  # K — absolute lower floor for pch_min (T_2M, T_G)
_HARD_MIN_FLOOR_TD = 213.0  # K — lower floor for TD_2M (−60°C); dew points at high altitude can be much colder

# Cosine seasonal deltas relative to the config sea-level anchor values.
# pch_min anchor = coldest month (Jan, delta=0); pch_max anchor = warmest month (Jul, delta=0).
# Both follow a symmetric cosine: equidistant months from the peak get equal deltas.
# Swing: pch_min 0→42 K (Jan→Jul); amplitude: pch_max 35→0 K (Jan→Jul).
# Added to sea-level pch_min_0 (delta=0 in Jan, increases toward summer)
_PCH_MIN_SEASONAL_DELTA: dict[int, float] = {
    1: 0,
    2: 3,
    3: 10,
    4: 21,
    5: 31,
    6: 39,
    7: 42,
    8: 39,
    9: 32,
    10: 21,
    11: 10,
    12: 3,
}
# Subtracted from sea-level pch_max_0 (delta=0 in Jul, increases toward winter)
_PCH_MAX_SEASONAL_DELTA: dict[int, float] = {
    1: 35,
    2: 33,
    3: 26,
    4: 18,
    5: 9,
    6: 2,
    7: 0,
    8: 2,
    9: 9,
    10: 17,
    11: 26,
    12: 33,
}
# Parameters for which altitude+seasonal correction applies (temperature-like, in K)
_HARD_ALTITUDE_PARAS = {"T_2M", "TD_2M", "T_G"}


def _seasonal_pch_min(base: float, month: int, alt_m: float, floor: float = _HARD_MIN_FLOOR) -> float:
    """Lower plausibility limit: sea-level base + seasonal delta − altitude lapse (8.3 K/km)."""
    delta = _PCH_MIN_SEASONAL_DELTA.get(month, 0)
    return max(base + delta - _HARD_LAPSE_MIN * (alt_m / 1000), floor)


_HARD_MAX_FLOOR = 273.15 + 25  # K — pch_max never drops below 25°C (Foeh protection)


def _seasonal_pch_max(base: float, month: int, alt_m: float) -> float:
    """Upper plausibility limit: seasonal + altitude lapse (6.5 K/km), floored at 20°C.

    Uses moist-adiabatic lapse rate (ICAO standard): warm events decrease more slowly
    with altitude. Floor at 20°C ensures Foeh events at any elevation are never flagged.
    """
    delta = _PCH_MAX_SEASONAL_DELTA.get(month, 0)
    return max(base - delta - _HARD_LAPSE_MAX * (alt_m / 1000), _HARD_MAX_FLOOR)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _load_inversion_stations(yaml_path: Path) -> dict:
    """Return {station: {ref, alt, ref_alt, alt_diff, ref2, ref2_alt, alt_diff2}} from YAML."""
    result = {}
    try:
        data = yaml.safe_load(yaml_path.read_text()).get("inversion_stations", {})
        for station, info in data.items():
            if not info.get("q99"):
                continue
            dh = int(info["ref_alt"]) - int(info["alt"])
            if dh == 0:
                continue
            entry = {
                "ref": info["ref"],
                "alt": int(info["alt"]),
                "ref_alt": int(info["ref_alt"]),
                "alt_diff": dh,
                "ref2": info.get("ref2"),
                "ref2_alt": int(info["ref2_alt"]) if info.get("ref2_alt") is not None else None,
                "alt_diff2": (int(info["ref2_alt"]) - int(info["alt"])) if info.get("ref2_alt") is not None else None,
            }
            result[station] = entry
        LOG.info("Inversion protection: loaded %d station pairs from %s", len(result), yaml_path)
    except (FileNotFoundError, OSError) as exc:
        LOG.warning("Inversion protection file not found — protection disabled: %s", exc)
    return result


def _check_inversion_protection(
    station: str, para: str, positive_tests: list, df, month: int, qc_config: SimpleNamespace
) -> tuple:
    """Return (protected, lapse_rate, confirming_stations) for this flag candidate.

    protected            – True if the flag should be suppressed due to inversion conditions.
    lapse_rate           – measured lapse rate [K/m] when protected, else None.
    confirming_stations  – list of nearby station names that also showed inversion.

    Conditions (all must hold):
      - para is in the configured parameters list (default: T_2M, TD_2M)
      - month is in one of the configured active seasons
      - station is in the inversion stations dict
      - at least one of {buddy_obs, buddy_diff} fired
      - none of {hard, isolation_check, plateau_test, DWH_flag} also fired
      - actual lapse rate (T_ref - T_sta) / (H_ref - H_sta) is within
        [lapse_rate_min, lapse_rate_max]
      - at least min_confirming_stations other inversion stations within
        confirming_radius_km also show a lapse rate in [lapse_rate_min, lapse_rate_max]
    """
    inv_cfg = getattr(qc_config, "inversion_protection", {}) or {}
    protected_paras = set(inv_cfg.get("parameters", ["T_2M", "TD_2M"]))
    if para not in protected_paras:
        return False, None, []
    seasons = inv_cfg.get("seasons", ["winter", "spring", "fall"])
    active_months: set = set()
    for s in seasons:
        active_months |= _SEASON_MONTHS.get(s, set())
    if month not in active_months:
        return False, None, []

    inv_stations = getattr(qc_config, "inversion_stations", {})
    if not inv_stations:
        return False, None, []
    info = inv_stations.get(station)
    if not info:
        return False, None, []

    if not any(t in _BUDDY_TESTS for t in positive_tests):
        return False, None, []
    blocking = set(inv_cfg.get("blocking_tests", _INVERSION_BLOCKING_TESTS))
    if any(t in blocking for t in positive_tests):
        return False, None, []

    if station not in df.index:
        return False, None, []

    t_sta = df.at[station, "2t"]
    if pd.isna(t_sta):
        return False, None, []

    def _lapse_rate(ref_name, alt_diff):
        if ref_name not in df.index or alt_diff == 0:
            return None
        t_ref = df.at[ref_name, "2t"]
        if pd.isna(t_ref):
            return None
        return (t_ref - t_sta) / alt_diff

    lr = _lapse_rate(info["ref"], info["alt_diff"])
    if lr is None and info.get("ref2") and info.get("alt_diff2"):
        lr = _lapse_rate(info["ref2"], info["alt_diff2"])
    if lr is None:
        return False, None, []

    lr_min = float(inv_cfg.get("lapse_rate_min", 0.006))
    lr_max = float(inv_cfg.get("lapse_rate_max", 0.1))
    if not lr_min <= lr <= lr_max:
        return False, None, []

    # Spatial confirmation: require at least min_confirming_stations other
    # inversion stations within confirming_radius_km that also show inversion.
    min_confirming = int(inv_cfg.get("min_confirming_stations", 0))
    confirming_stations: list = []
    if min_confirming > 0:
        radius_km = float(inv_cfg.get("confirming_radius_km", 100.0))
        try:
            sta_lat = float(df.at[station, "latitude"])
            sta_lon = float(df.at[station, "longitude"])
        except (KeyError, ValueError):
            return False, None, []
        _confirming_with_dist: list = []
        for other, other_info in inv_stations.items():
            if other == station or other not in df.index:
                continue
            try:
                other_lat = float(df.at[other, "latitude"])
                other_lon = float(df.at[other, "longitude"])
            except (KeyError, ValueError):
                continue
            dist = _haversine_km(sta_lat, sta_lon, other_lat, other_lon)
            if dist > radius_km:
                continue
            t_other = df.at[other, "2t"]
            if pd.isna(t_other):
                continue

            def _other_lr(oinfo, t_o):
                r = oinfo["ref"]
                if r not in df.index or oinfo["alt_diff"] == 0:
                    return None
                tv = df.at[r, "2t"]
                if pd.isna(tv):
                    r2 = oinfo.get("ref2")
                    dh2 = oinfo.get("alt_diff2")
                    if r2 and dh2 and r2 in df.index:
                        tv = df.at[r2, "2t"]
                        if pd.isna(tv):
                            return None
                        return (tv - t_o) / dh2
                    return None
                return (tv - t_o) / oinfo["alt_diff"]

            other_lr = _other_lr(other_info, t_other)
            if other_lr is not None and lr_min <= other_lr <= lr_max:
                _confirming_with_dist.append((dist, other))
        if len(_confirming_with_dist) < min_confirming:
            return False, None, []
        _confirming_with_dist.sort()
        confirming_stations = [
            {"station": s, "distance_km": round(d, 1)} for d, s in _confirming_with_dist[:min_confirming]
        ]

    return True, lr, confirming_stations


def _load_qc_config(
    config_file=None,
    hard_blacklist_file=None,
    inversion_protection_file=None,
) -> SimpleNamespace:
    """Load QC configuration, with optional path overrides for per-run customisation.

    Parameters
    ----------
    config_file : str or Path, optional
        Path to the main QC config YAML. Defaults to ``clean_observation_config.yaml``
        in the package directory.
    hard_blacklist_file : str or Path, optional
        Path to the hard-blacklist YAML. Overrides ``hard_blacklist_file`` in the
        main config when provided.
    inversion_protection_file : str or Path, optional
        Path to the inversion-stations YAML. Overrides ``inversion_protection.file``
        in the main config when provided.
    """
    default_dir = Path(__file__).parent
    cfg_path = Path(config_file) if config_file is not None else default_dir / "clean_observation_config.yaml"
    # Resolve sibling paths (blacklist, inversion stations) relative to the config file itself,
    # so that an external config_file can carry its own relative references.
    config_dir = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text())

    def _identity(x):
        return x

    ns = SimpleNamespace(**cfg)
    ns.fgt_excluded = cfg.get("fgt_excluded", {})
    ns.obs_only_tests = _OBS_ONLY_TESTS
    ns.par2pi = _PAR2PI
    ns.parquet_to_qc = {col: (qp, _identity) for col, qp in _PARQUET_TO_QC.items()}
    ns.qc_to_parquet = _QC_TO_PARQUET

    if hard_blacklist_file is not None:
        blacklist_path = Path(hard_blacklist_file)
    else:
        blacklist_file = cfg.get("hard_blacklist_file")
        blacklist_path = (config_dir / blacklist_file) if blacklist_file else None
    if blacklist_path is not None:
        blacklist_cfg = yaml.safe_load(blacklist_path.read_text())
        ns.hard_blacklist = blacklist_cfg["hard_blacklist"]

    inv_prot = cfg.get("inversion_protection", {})
    ns.inversion_protection = inv_prot
    if inversion_protection_file is not None:
        inv_path = Path(inversion_protection_file)
    else:
        inv_file = inv_prot.get("file") if inv_prot else None
        inv_path = (config_dir / inv_file) if inv_file else None
    ns.inversion_stations = _load_inversion_stations(inv_path) if inv_path else {}

    return ns


_qc_config = None
try:
    _qc_config = _load_qc_config()
except FileNotFoundError:
    # Default config not present in package — config_file= must be passed explicitly.
    pass

try:
    from .clean_observation_tests import DWH_flag as _DWH_flag
    from .clean_observation_tests import buddy_check as _buddy_check
    from .clean_observation_tests import first_guess_test as _first_guess_test
    from .clean_observation_tests import hard_test as _hard_test
    from .clean_observation_tests import isolation_check as _isolation_check
    from .clean_observation_tests import plateau_test as _plateau_test
    from .clean_observation_tests import spacial_ct_dual as _spacial_ct_dual
    from .clean_observation_tests import spacial_ct_resistant as _spacial_ct_resistant
    from .clean_observation_tests import tests_summary as _tests_summary

    _QC_AVAILABLE = True
except ImportError as _e:
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
        ``tests_positive`` (list of test names that voted to flag).
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
    residuals_dir : str, optional
        Directory containing pre-computed obs-minus-model residual parquet files
        named ``residuals_{YYYYMMDDhhmm}.parquet``.  When set, residuals are
        read directly instead of loading a GRIB file and interpolating.  Takes
        priority over ``model_grib_path``.  The parquet must have the station
        name as index and QC parameter names (``T_2M``, ``TD_2M``, ``SP_10M``,
        …) as columns.  The model background is recovered as ``obs − diff`` so
        that ``fgt`` can still run.
    config_file : str, optional
        Path to the main QC config YAML. Defaults to ``clean_observation_config.yaml``
        shipped with the package.
    hard_blacklist_file : str, optional
        Path to the hard-blacklist YAML. Overrides the ``hard_blacklist_file``
        entry in the main config when provided.
    inversion_protection_file : str, optional
        Path to the inversion-stations YAML. Overrides the
        ``inversion_protection.file`` entry in the main config when provided.
    """

    def __init__(
        self,
        obs_path_in: str,
        obs_path_out: str,
        model_grib_path: str = None,
        model_interp: str = "nearest",
        residuals_dir: str = None,
        config_file: str = None,
        hard_blacklist_file: str = None,
        inversion_protection_file: str = None,
    ):
        if model_interp not in ("nearest", "min_elev_diff"):
            raise ValueError(f"model_interp must be 'nearest' or 'min_elev_diff', got {model_interp!r}")
        self.obs_path_in = Path(obs_path_in)
        self.obs_path_out = Path(obs_path_out)
        self.model_grib_path = model_grib_path
        self.model_interp = model_interp
        self.residuals_dir = Path(residuals_dir) if residuals_dir is not None else None
        if config_file is not None or hard_blacklist_file is not None or inversion_protection_file is not None:
            self._qc_config = _load_qc_config(
                config_file=config_file,
                hard_blacklist_file=hard_blacklist_file,
                inversion_protection_file=inversion_protection_file,
            )
        elif _qc_config is not None:
            self._qc_config = _qc_config
        else:
            raise RuntimeError(
                "No QC config available: the default clean_observation_config.yaml is not "
                "present in the package. Pass config_file= explicitly in the inference config."
            )
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
        df[[c for c in df.columns if c not in pi_cols]].to_parquet(self.obs_path_out)
        LOG.info("Saved %d cleaned stations to %s", len(df), self.obs_path_out)

        ts_match = re.search(r"\d{12}", self.obs_path_in.stem)
        obs_timestamp = ts_match.group() if ts_match else None
        json_path = self.obs_path_out.with_name(self.obs_path_out.stem + "_flagged.json")
        self._write_flagged_json(
            json_path,
            getattr(self, "_flagged", []),
            getattr(self, "_tests_done", {}),
            getattr(self, "_qc_duration", 0.0),
            obs_timestamp,
        )

        inv_prot_cfg = getattr(self._qc_config, "inversion_protection", {}) or {}
        if inv_prot_cfg.get("report") and obs_timestamp:
            protected = getattr(self, "_inversion_protected", [])
            if protected:
                inv_json_path = json_path.parent / f"inversion_protection_{obs_timestamp}.json"
                self._write_inversion_protection_json(inv_json_path, protected, obs_timestamp)

        return data

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply QC tests and set suspected values to NaN.

        Two stages:

        1. **Hard blacklist** (``hard_blacklist``): station/parameter pairs
           permanently set to NaN before any test runs, so the spatial QC
           algorithms never see these values and cannot be influenced by them.

        2. **Automated QC** (``par2check`` parameters): for each parameter the
           configured tests are run via ``make_tests``.  Obs-only tests
           (``hard``, ``isolation_check``, ``buddy_obs``, ``DWH_flag``,
           ``plateau_test``) always run; model-dependent tests (``buddy_diff``,
           ``fgt``) are added only when ``model_grib_path`` is set.  Each test
           returns a blacklist; the weighted score from ``tests_summary`` is
           compared against ``titan_ntests_threshold[para]["threshold_summary"]``.
           Stations above the threshold are set to NaN unless listed in
           ``stations_excluded[para]``.

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
        # Inversion-protected entries: flags that were suppressed by inversion_protection
        self._inversion_protected: list = []

        if not _QC_AVAILABLE:
            LOG.warning("QC modules unavailable, skipping cleaning")
            return df

        # --- Stage 1: hard blacklist — applied before QC so tests never see these values ----
        for entry in self._qc_config.hard_blacklist.values():
            station = entry["station"]
            if station not in df.index:
                continue
            for para in entry["paras"]:
                LOG.info("Hard blacklist (pre-QC): setting %s / %s to NaN", station, para)
                for parquet_col in self._qc_config.qc_to_parquet.get(para, []):
                    if parquet_col not in df.columns or station not in df.index:
                        continue
                    df.loc[station, parquet_col] = np.nan

        df_qc, lats, lons, elevs, stations, df_pi = self._build_df_qc(df)
        df_mod, df_diff, active_tests = self._resolve_model_background(df_qc, lats, lons, elevs)

        current_f = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        _t0 = time.monotonic()

        # Index of hard-blacklisted (station, para) pairs — used below to suppress
        # duplicate logging when automated QC also flags them.
        _hard_blacklisted: dict[str, set] = {}
        for _entry in self._qc_config.hard_blacklist.values():
            for _p in _entry["paras"]:
                _hard_blacklisted.setdefault(_p, set()).add(_entry["station"])

        # Month of the current observation file — used for inversion protection
        _ts = re.search(r"\d{12}", self.obs_path_in.stem)
        _obs_month = int(_ts.group()[4:6]) if _ts else 0

        # Pre-load historical parquets once for all plateau_test calls
        _historical_raw = self._load_historical_parquets()

        # --- Stage 2: automated QC tests per parameter --------------------
        for para in self._qc_config.par2check:
            if para not in df_qc.columns:
                LOG.warning("Skipping %s: not available in parquet", para)
                continue

            tests_QC = self._qc_config.titan_ntests_threshold[para]["tests_QC"]
            all_tests = list(tests_QC.keys())
            all_weights = tests_QC  # {test: weight}
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
                    para,
                    skipped,
                )

            # Blacklist accumulator expected by make_tests:
            #   'n'    : number of planned tests
            #   'tests': list of planned test names
            #   <name> : per-test sub-dict with lists 'ID', 'Station', 'Time', 'Parameter'
            my_dict = {"n": len(tests_to_do), "tests": tests_to_do}
            for test in tests_to_do:
                my_dict[test] = {"ID": [], "Station": [], "Time": [], "Parameter": []}

            with np.errstate(invalid="ignore", divide="ignore"):
                blacklist, _, executed_tests = self.make_tests(
                    current_f,
                    df_qc,
                    df_diff,
                    df_mod,
                    para,
                    stations.copy(),
                    lats.copy(),
                    lons.copy(),
                    elevs.copy(),
                    0,
                    0,
                    my_dict,
                    tests_to_do,
                    df_pi=df_pi,
                    obs_path_in=self.obs_path_in,
                    historical_raw=_historical_raw,
                )

            # executed_tests may be shorter than tests_to_do (e.g. plateau_test skipped)
            # or longer (isolation_check always runs but is not in tests_QC).
            # Score uses only the tests that are in tests_QC (have configured weights).
            # isolation_check stations are flagged directly after the score loop.
            score_tests = [t for t in executed_tests if t in all_tests]
            self._tests_done[para] = executed_tests
            blacklist["tests"] = score_tests
            blacklist["n"] = len(score_tests)
            weights = [all_weights[t] for t in score_tests]

            # Score = sum(weight_i for flagging tests) / n_score_tests.
            # A station is blacklisted when score > threshold_summary (default 0.2).
            qc_summary = _tests_summary(blacklist, weights, score_tests)
            threshold = self._qc_config.titan_ntests_threshold[para]["threshold_summary"]

            flagged_stations = []

            # inversion_protection: suppress buddy-only T_2M flags when the
            # measured lapse rate to the reference station confirms an inversion.
            if self._qc_config.inversion_stations:
                for station in list(qc_summary.columns):
                    score = qc_summary[station].iloc[0]
                    if score <= threshold:
                        continue
                    pos = [t for t in executed_tests if station in blacklist.get(t, {}).get("Station", [])]
                    _protected, _lr, _confirming = _check_inversion_protection(
                        station, para, pos, df, _obs_month, self._qc_config
                    )
                    if _protected:
                        LOG.info(
                            "Inversion protection: suppressing %s buddy flag for %s (lr=%.4f K/m)", para, station, _lr
                        )
                        qc_summary = qc_summary.drop(columns=[station])
                        _obs_cols = [c for c in self._qc_config.qc_to_parquet.get(para, []) if c in df.columns]
                        _obs_val = (
                            round(float(df.at[station, _obs_cols[0]]), 4)
                            if _obs_cols and not pd.isna(df.at[station, _obs_cols[0]])
                            else None
                        )
                        self._inversion_protected.append(
                            {
                                "station": station,
                                "qc_parameter": para,
                                "obs_value": _obs_val,
                                "lapse_rate": round(_lr, 6),
                                "tests_suppressed": pos,
                                "confirming_stations": _confirming,
                            }
                        )

            # stations_excluded: trusted stations (e.g. reference stations) that are
            # never blacklisted by automated QC, regardless of their score.
            excluded = self._qc_config.stations_excluded.get(para, {}).get("stations", [])
            for station in qc_summary.columns:
                score = qc_summary[station].iloc[0]
                if score <= threshold:
                    continue
                if station in excluded:
                    LOG.info("QC flagged %s for station %s but station is excluded — skipping", para, station)
                    continue
                if station in _hard_blacklisted.get(para, set()):
                    LOG.debug(
                        "QC also flagged %s / %s (score=%.2f) — already in hard blacklist, skipping",
                        para,
                        station,
                        score,
                    )
                    continue
                LOG.info("QC flagged %s for station %s (score=%.2f)", para, station, score)
                flagged_stations.append(station)
                # Which individual tests voted to flag this station?
                positive_tests = [t for t in executed_tests if station in blacklist.get(t, {}).get("Station", [])]
                # A QC parameter may map to multiple parquet columns
                # (e.g. SP_10M → 10u and 10v); set all of them to NaN.
                self._flag_station_columns(df, df_qc, station, para, tests_positive=positive_tests)

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

        # --- Done: hard blacklist (Stage 1) + automated QC (Stage 2) complete -------
        self._qc_duration: float = time.monotonic() - _t0
        LOG.info("QC tests completed in %.2f s", self._qc_duration)
        return df

    def _load_historical_parquets(self) -> dict:
        """Load historical raw parquets needed by plateau_test across all parameters.

        Computes the union of all offsets (minutes back from current) required by
        every configured plateau_test, reads each unique file exactly once, and
        returns {offset_minutes: raw_DataFrame}.  plateau_test then selects only
        the entries matching its own gran_minutes without any further I/O.
        """
        needed: set = set()
        for para in self._qc_config.par2check:
            pt_cfg = (getattr(self._qc_config, "plateau_test", None) or {}).get(para)
            if pt_cfg is None:
                continue
            gran = int(pt_cfg.get("gran", 60))
            window = int(pt_cfg.get("window", 24))
            for step in range(1, int(window * 60 / gran) + 1):
                needed.add(step * gran)
        if not needed:
            return {}

        ts_match = re.search(r"\d{12}", self.obs_path_in.stem)
        if not ts_match:
            return {}
        current = datetime.strptime(ts_match.group(), "%Y%m%d%H%M")
        stem_pattern = self.obs_path_in.stem.replace(ts_match.group(), "{}")

        result = {}
        for offset_min in sorted(needed):
            t = current - timedelta(minutes=offset_min)
            fn = self.obs_path_in.parent / (stem_pattern.format(t.strftime("%Y%m%d%H%M")) + self.obs_path_in.suffix)
            if fn.exists():
                try:
                    result[offset_min] = pd.read_parquet(fn)
                except Exception as exc:
                    LOG.warning("plateau_test preload: could not read %s: %s", fn, exc)

        LOG.info("plateau_test: preloaded %d/%d historical parquet files", len(result), len(needed))
        return result

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
        historical_raw=None,
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
        LOG.info("make_tests %s: %d stations (non-NaN)", para, len(stations))

        runners = self._build_test_runners(
            tests_to_do,
            df_obs,
            values,
            diffs,
            mods,
            stations,
            lats,
            lons,
            elevs,
            para,
            current_f,
            df_pi,
            obs_path_in,
            historical_raw=historical_raw,
        )

        for test in tests_to_do:
            if test not in runners:
                continue

            raw = runners[test]()

            if test == "DWH_flag":
                blacklist, freq = raw
            elif test == "plateau_test":
                if raw is None:
                    LOG.warning("plateau_test skipped — no historical files available")
                    continue
                blacklist = raw
            else:
                blacklist = raw

            self._merge_blacklist(my_dict, test, blacklist)
            n = len((blacklist or {}).get("Station", []))
            LOG.info("%-18s %s blacklisted stations: %d", test, para, n)
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
        for parquet_col, (qc_para, converter) in self._qc_config.parquet_to_qc.items():
            if parquet_col in df.columns:
                df_qc[qc_para] = converter(df[parquet_col].to_numpy())

        # SP_10M is wind speed derived from the U/V components; the scalar speed
        # is what titanlib spatial tests operate on (not the vector components).
        if "10u" in df.columns and "10v" in df.columns:
            df_qc["SP_10M"] = np.sqrt(df["10u"].to_numpy() ** 2 + df["10v"].to_numpy() ** 2)

        lats = df["latitude"].to_numpy(dtype=float)
        lons = df["longitude"].to_numpy(dtype=float)
        # altitude may be missing for synthetic test data; default to 0 m a.s.l.
        elevs = df["elevation"].to_numpy(dtype=float) if "elevation" in df.columns else np.zeros(len(df), dtype=float)
        df_qc["altitude"] = elevs
        stations = np.array(df.index.to_list())

        # Plausibility frame for DWH_flag: index = station name, columns = *_pi.
        # Each *_pi column contains a DWH plausibility value (0–1) pre-fetched by
        # RetrieveObservation.  Stations with pi < dwh_plausibility_thr are flagged.
        pi_cols = [col for col in df.columns if col.endswith("_pi")]
        df_pi = df[pi_cols] if pi_cols else pd.DataFrame(index=df.index)

        return df_qc, lats, lons, elevs, stations, df_pi

    def _resolve_model_background(self, df_qc: pd.DataFrame, lats, lons, elevs):
        """Load the NWP background and decide which test set is active.

        Priority: ``residuals_dir`` > ``model_grib_path`` > obs-only fallback.

        Returns
        -------
        df_mod : pd.DataFrame
            Model values at station locations (NaN when no background is available).
        df_diff : pd.DataFrame
            Observation innovation ``df_qc - df_mod`` (NaN when no background is available).
        active_tests : set[str]
            ``obs_only_tests`` when no model is loaded; extended with
            ``{"buddy_diff", "fgt"}`` when the model background is available.
        """
        if self.residuals_dir is not None:
            try:
                df_mod, df_diff = self._load_residuals_at_stations(df_qc)
                LOG.info("Pre-computed residuals loaded from %s; extended test set active", self.residuals_dir)
                return df_mod, df_diff, self._qc_config.obs_only_tests | {"buddy_diff", "fgt"}
            except Exception as exc:
                LOG.warning("Failed to load pre-computed residuals (%s); falling back to obs-only tests", exc)
        elif self.model_grib_path is not None:
            try:
                df_mod, df_diff = self._load_model_at_stations(df_qc, lats, lons, elevs)
                LOG.info("Model background loaded from GRIB; extended test set active")
                return df_mod, df_diff, self._qc_config.obs_only_tests | {"buddy_diff", "fgt"}
            except Exception as exc:
                LOG.warning("Failed to load model GRIB (%s); falling back to obs-only tests", exc)
        df_mod, df_diff = self._nan_frames(df_qc)
        return df_mod, df_diff, self._qc_config.obs_only_tests

    def _flag_station_columns(self, df, df_qc, station, para, **extra):
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
        **extra
            Additional keys appended to each ``_flagged`` entry
            (e.g. ``tests_positive=[...]`` for ``"qc_test"``).
        """
        # Record the QC-space value (e.g. wind speed scalar, not u/v)
        # that triggered the flag, for traceability in the JSON output.
        cols = [c for c in self._qc_config.qc_to_parquet.get(para, []) if c in df.columns and station in df.index]
        if cols:
            qc_values = [round(float(df.at[station, c]), 4) if not pd.isna(df.at[station, c]) else None for c in cols]
            self._flagged.append(
                {
                    "station": station,
                    "column": cols,
                    "qc_parameter": para,
                    "qc_value": qc_values,
                    **extra,
                }
            )
            for c in cols:
                df.loc[station, c] = np.nan

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
            "buddy_check": _buddy_check,
            "first_guess_test": _first_guess_test,
            "isolation_check": _isolation_check,
            "spacial_ct_dual": _spacial_ct_dual,
            "spacial_ct_resistant": _spacial_ct_resistant,
        }

    def _build_test_runners(
        self,
        tests_to_do,
        df_obs,
        values,
        diffs,
        mods,
        stations,
        lats,
        lons,
        elevs,
        para,
        current_f,
        df_pi,
        obs_path_in,
        historical_raw=None,
    ):
        """Return a dict mapping each requested test name to a zero-arg callable.

        Only runners for tests present in *tests_to_do* are built, so config
        dicts are only accessed for tests that will actually run.
        Spatial functions are resolved via ``_spatial_fns`` so subclasses can
        swap implementations without overriding this method.
        """
        cfg = self._qc_config
        fns = self._spatial_fns()
        runners = {}

        if "hard" in tests_to_do:
            pch_min = cfg.plausibility_thresholds[para]["pch_min"]
            pch_max = cfg.plausibility_thresholds[para]["pch_max"]
            _hard_limits = None
            if para in _HARD_ALTITUDE_PARAS and obs_path_in is not None:
                _ts = re.search(r"\d{12}", Path(obs_path_in).stem)
                _month = int(_ts.group()[4:6]) if _ts else None
                if _month and "altitude" in df_obs.columns:
                    _min_floor = _HARD_MIN_FLOOR_TD if para == "TD_2M" else _HARD_MIN_FLOOR
                    _hard_limits = {
                        row["sta_name"]: (
                            _seasonal_pch_min(pch_min, _month, row["altitude"], floor=_min_floor),
                            _seasonal_pch_max(pch_max, _month, row["altitude"]),
                        )
                        for _, row in df_obs[["sta_name", "altitude"]].iterrows()
                    }
            runners["hard"] = lambda: _hard_test(df_obs, pch_min, pch_max, para, current_f, limits=_hard_limits)

        if "buddy_obs" in tests_to_do:
            b = cfg.buddy[para]
            runners["buddy_obs"] = lambda: fns["buddy_check"](
                stations,
                lats,
                lons,
                elevs,
                values,
                para,
                current_f,
                b["threshold"],
                b["max_elev_diff"],
                b["elev_gradient"],
                b["min_std"],
                b["num_iterations"],
                b["num_min"],
                b["radius"],
            )

        if "buddy_diff" in tests_to_do:
            bd = cfg.buddy_diff[para]
            runners["buddy_diff"] = lambda: fns["buddy_check"](
                stations,
                lats,
                lons,
                elevs,
                diffs,
                para,
                current_f,
                bd["threshold"],
                bd["max_elev_diff"],
                bd["elev_gradient"],
                bd["min_std"],
                bd["num_iterations"],
                bd["num_min"],
                bd["radius"],
            )

        if "fgt" in tests_to_do:
            fg = cfg.fgt[para]
            _fgt_excl = set(cfg.fgt_excluded.get(para, {}).get("stations", []))

            def _run_fgt(
                _stations=stations,
                _lats=lats,
                _lons=lons,
                _elevs=elevs,
                _values=values,
                _mods=mods,
                _fg=fg,
                _excl=_fgt_excl,
            ):
                if _excl:
                    mask = [s not in _excl for s in _stations]
                    idx = [i for i, m in enumerate(mask) if m]
                    _s = [_stations[i] for i in idx]
                    _la = _lats[idx] if hasattr(_lats, "__getitem__") else [_lats[i] for i in idx]
                    _lo = _lons[idx] if hasattr(_lons, "__getitem__") else [_lons[i] for i in idx]
                    _el = _elevs[idx] if hasattr(_elevs, "__getitem__") else [_elevs[i] for i in idx]
                    _v = _values[idx] if hasattr(_values, "__getitem__") else [_values[i] for i in idx]
                    _m = _mods[idx] if hasattr(_mods, "__getitem__") else [_mods[i] for i in idx]
                else:
                    _s, _la, _lo, _el, _v, _m = _stations, _lats, _lons, _elevs, _values, _mods
                return fns["first_guess_test"](
                    _s,
                    _la,
                    _lo,
                    _el,
                    _v,
                    _m,
                    para,
                    current_f,
                    _fg["background_elab_type"],
                    _fg["num_min_outer"],
                    _fg["num_max_outer"],
                    _fg["inner_radius"],
                    _fg["outer_radius"],
                    _fg["num_iterations"],
                    _fg["num_min_prof"],
                    _fg["min_elev_diff"],
                    _fg["min_horizontal_scale"],
                    _fg["max_horizontal_scale"],
                    _fg["kth_closest_obs_horizontal_scale"],
                    bool(_fg["debug"]),
                    bool(_fg["basic"]),
                    _fg["tpostneg"],
                )

            runners["fgt"] = _run_fgt

        if "spt_resistant" in tests_to_do:
            sp = cfg.spt_resistant[para]
            runners["spt_resistant"] = lambda: fns["spacial_ct_resistant"](
                stations,
                lats,
                lons,
                elevs,
                values,
                para,
                current_f,
                sp["background_elab_type"],
                sp["num_min_outer"],
                sp["num_max_outer"],
                sp["inner_radius"],
                sp["outer_radius"],
                sp["num_iterations"],
                sp["num_min_prof"],
                sp["min_elev_diff"],
                sp["min_horizontal_scale"],
                sp["max_horizontal_scale"],
                sp["kth_closest_obs_horizontal_scale"],
                sp["vertical_scale"],
                sp["debug"],
                sp["basic"],
            )

        if "spt_dual" in tests_to_do:
            sd = cfg.sct_dual[para]
            runners["spt_dual"] = lambda: fns["spacial_ct_dual"](
                stations,
                lats,
                lons,
                elevs,
                values,
                para,
                current_f,
                sd["num_min_outer"],
                sd["num_max_outer"],
                sd["inner_radius"],
                sd["outer_radius"],
                sd["num_iterations"],
                sd["min_horizontal_scale"],
                sd["max_horizontal_scale"],
                sd["kth_closest_obs_horizontal_scale"],
                sd["vertical_scale"],
                bool(sd["debug"]),
                sd["condition"],
                float(sd["event_thresholds"]),
                float(sd["test_thresholds"]),
            )

        if "DWH_flag" in tests_to_do:
            pi_col = cfg.par2pi.get(para, "")
            if df_pi is not None and pi_col and pi_col in df_pi.columns:
                pla_series = df_pi[pi_col]
            else:
                LOG.warning("DWH_flag: no pi column for %s, skipping", para)
                pla_series = pd.Series(dtype=float)
            runners["DWH_flag"] = lambda: _DWH_flag(current_f, para, stations, pla_series, qc_config=cfg)

        if "plateau_test" in tests_to_do:
            pt = cfg.plateau_test[para]
            runners["plateau_test"] = lambda: _plateau_test(
                df_obs,
                pt["window"],
                pt["sd"],
                para,
                current_f,
                obs_path_in=obs_path_in,
                gran_minutes=pt["gran"],
                historical_raw=historical_raw,
                qc_config=cfg,
            )

        if "isolation_check" in tests_to_do:
            ic = cfg.isolation_check[para]
            runners["isolation_check"] = lambda: fns["isolation_check"](
                stations,
                lats,
                lons,
                elevs,
                para,
                current_f,
                ic["num_min"],
                ic["radius"],
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
            "flagged": [{k: round(v, 2) if isinstance(v, float) else v for k, v in e.items()} for e in flagged],
        }
        with open(json_path, "w") as fh:
            json.dump(output, fh, indent=2)
            fh.write("\n")
        LOG.info("Wrote %d flagged entries to %s", len(flagged), json_path)

    @staticmethod
    def _write_inversion_protection_json(json_path: Path, protected: list, obs_timestamp: str) -> None:
        """Write inversion-protection suppression report to *json_path*."""

        def _entry(e: dict) -> str:
            return (
                "    {\n"
                f'      "station": {json.dumps(e["station"])},\n'
                f'      "qc_parameter": {json.dumps(e["qc_parameter"])},\n'
                f'      "obs_value": {json.dumps(e.get("obs_value"))},\n'
                f'      "lapse_rate": {e["lapse_rate"]},\n'
                f'      "tests_suppressed": {json.dumps(e["tests_suppressed"])},\n'
                f'      "confirming_stations": {json.dumps(e.get("confirming_stations", []))}\n'
                "    }"
            )

        entries = ",\n".join(_entry(e) for e in protected)
        body = (
            "{\n"
            f'  "obs_timestamp": {json.dumps(obs_timestamp)},\n'
            f'  "n_protected": {len(protected)},\n'
            '  "protected": [\n' + (entries + "\n" if entries else "") + "  ]\n}\n"
        )
        json_path.write_text(body)
        LOG.info(
            "Inversion protection report: %d suppressed flag(s) → %s",
            len(protected),
            json_path,
        )

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

    def _load_residuals_at_stations(self, df_qc: pd.DataFrame):
        """Load pre-computed obs-minus-model residuals and reconstruct the model background.

        Reads ``residuals_{ts}.parquet`` from ``self.residuals_dir``.  The file
        must have the station name as index and QC parameter names (``T_2M``,
        ``TD_2M``, ``SP_10M``, …) as columns.

        The model background is recovered as ``df_mod = df_qc − df_diff`` so that
        ``fgt`` can use per-station model values even without the GRIB file.

        Returns
        -------
        df_mod : pd.DataFrame
            Reconstructed model values at station locations.
        df_diff : pd.DataFrame
            Observation innovation (obs − model) aligned to ``df_qc`` rows.
        """
        ts_match = re.search(r"\d{12}", self.obs_path_in.stem)
        if not ts_match:
            raise ValueError(f"Cannot extract timestamp from {self.obs_path_in.stem!r}")
        ts = ts_match.group()

        matches = list(self.residuals_dir.glob(f"*{ts}*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No residuals parquet matching *{ts}*.parquet in {self.residuals_dir}")
        residuals_path = matches[0]
        if len(matches) > 1:
            LOG.warning("Multiple residuals files match %s — using %s", ts, residuals_path.name)

        LOG.info("Reading residuals: %s", residuals_path)
        df_res = pd.read_parquet(residuals_path)

        # Rename parquet column names (2t, 2d, skt, …) to QC parameter names (T_2M, TD_2M, T_G, …)
        col_to_qc = {col: para for col, (para, _) in self._qc_config.parquet_to_qc.items()}
        df_res = df_res.rename(columns=col_to_qc)
        LOG.debug("Residuals columns after rename: %s", list(df_res.columns))

        station_names = df_qc["sta_name"].to_numpy()
        df_diff, _ = self._nan_frames(df_qc)

        for para in df_qc.columns:
            if para == "sta_name" or para not in df_res.columns:
                continue
            diff_vals = np.array(
                [float(df_res.at[stn, para]) if stn in df_res.index else np.nan for stn in station_names],
                dtype=float,
            )
            df_diff[para] = diff_vals

        # Reconstruct model: obs − diff = model background
        df_mod = df_qc.copy()
        for col in df_qc.columns:
            if col != "sta_name":
                df_mod[col] = df_qc[col].to_numpy() - df_diff[col].to_numpy()

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
            "T_2M": ["T_2M", "2t"],
            "TD_2M": ["TD_2M", "2d"],
            "VMAX10M": ["VMAX_10M", "vmax"],
            # SP_10M derived from U/V — handled separately below
        }
        _U_SHORTNAMES = ["U_10M", "10u"]
        _V_SHORTNAMES = ["V_10M", "10v"]
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
                [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)],
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
                station_idx = np.array(
                    [nbrs[np.argmin(np.abs(grid_hsurf[nbrs] - sta_elev))] for nbrs, sta_elev in zip(indices, elevs)]
                )
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
                df_mod["SP_10M"] = np.sqrt(u_vals[station_idx] ** 2 + v_vals[station_idx] ** 2)
                LOG.debug("Derived SP_10M (U/V) for %d stations", len(station_idx))

        # df_diff = obs - model background (observation innovation).
        # buddy_diff and fgt use this instead of raw obs values so that
        # the NWP climatology is removed before spatial consistency checks.
        df_diff = df_qc.copy()
        for col in df_qc.columns:
            if col != "sta_name":
                df_diff[col] = df_qc[col].to_numpy() - df_mod[col].to_numpy()

        return df_mod, df_diff
