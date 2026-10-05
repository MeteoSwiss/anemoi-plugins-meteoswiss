"""
NudgeTowardObservation — Interpolation of Residuals (IoR).

Station residuals (background − observation) are spread onto the ICON grid
and subtracted from the background, nudging it toward the observations.

Algorithm description (applied to each variable)
----------------------
1. Project the ICON grid and stations to Swiss LV95 (EPSG:2056), in km.
2. Restrict the grid points (POIs) to the station bounding box, buffered by
   max_dist.
3. Reduce observations to the model orography with a lapse rate (T_2M, PS)
   and compute residuals at the nearest grid cell of each station.
4. Look up the barrier- and elevation-aware effective distance d_eff of each
   (POI, station) pair in a precomputed cache (built by
   ``scripts/generate_d_eff_cache.py``).
5. Optionally shrink each station's influence radius by its reliability.
6. Weight each station by 1 / d_eff^weight_power and normalise
   (``ned_interp``).
7. Multiply the normalised weights by a linear taper in Euclidean distance
   that reaches zero at each station's own radius.
8. Subtract the weighted sum of residuals from the background.

The taper is applied after normalisation: a POI with a single contributing
station would otherwise always get weight 1, whatever its distance.

Station reliability (optional)
------------------------------
Each station's residual is predicted from all other stations (leave-one-out)
with the same weighting. The discrepancy e = actual − predicted is turned into
a robust z-score u = (e − median(e)) / (1.4826 · MAD(e)) and then into a Tukey
biweight reliability = max(0, 1 − (u / number_of_std)²)² in [0, 1]. Stations
with no neighbour within max_dist get reliability 1 and are excluded from the
median/MAD. Each station's radius becomes
min_dist + (max_dist − min_dist) · reliability, with
min_dist = reliability_min_dist_frac · max_dist. The leave-one-out prediction
and the domain buffer use the full max_dist.
"""

import logging
from pathlib import Path
from typing import Optional
from typing import Union

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from pyproj import Transformer
from scipy.spatial import cKDTree

LOG = logging.getLogger(__name__)

_ICON_TO_ECMWF_NAME = {
    "T_2M": "2t",  # 2 m temperature         [K]
    "TD_2M": "2d",  # 2 m dewpoint            [K]
    "U_10M": "10u",  # 10 m U wind component   [m/s]
    "V_10M": "10v",  # 10 m V wind component   [m/s]
    "PMSL": "msl",  # mean sea-level pressure  [Pa]
    "PS": "sp",  # station-level (unreduced) surface pressure [Pa]
    "VMAX_10M": "vmax",  # 10 m wind gust           [m/s]
}

_DEFAULT_TEMPERATURE_LAPSE_RATE = 0.0065  # [K/m]
_DEFAULT_TEMPERATURE_LAPSE_RATE_VARS = frozenset({"T_2M"})
_DEFAULT_PRESSURE_LAPSE_RATE = 11.5  # [Pa/m]
_DEFAULT_PRESSURE_LAPSE_RATE_VARS = frozenset({"PS"})

_M_PER_KM = 1000.0
# Scales the median absolute deviation to a standard deviation for Gaussian data.
_MAD_TO_STD = 1.4826
# Stations below this reliability are counted as flagged in the log.
_FLAG_RELIABILITY = 0.5


def ned_interp(
    sta_res: xr.Dataset,
    ned_sta_poi: xr.DataArray,
    max_dist: Optional[Union[float, xr.DataArray]] = None,
    weight_power: float = 1,
    lim_effective: float = 0,
    taper: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Spread station residuals to POIs by inverse-distance weighting (IDW).

    Parameters
    ----------
    sta_res : xr.Dataset
        Residuals per variable, dim ``sta``.
    ned_sta_poi : xr.DataArray
        Effective distances, dims ``poi`` x ``sta``.
    max_dist : float or xr.DataArray, optional
        Pairs with ``ned_sta_poi >= max_dist`` are excluded; a DataArray over
        ``sta`` gives a per-station radius.
    weight_power : float
        IDW exponent: weight = 1 / distance^weight_power.
    lim_effective : float
        Added to the weight-normalisation denominator, acting as a virtual
        zero-residual station that damps corrections where stations are sparse.
    taper : xr.DataArray, optional
        Factor in [0, 1] per (poi, sta), applied after normalisation so that
        it also acts on POIs with a single contributing station.

    Returns the weighted residual sum per POI; NaN where no station contributes.
    """
    # NaN distance -> NaN IDW weight -> effectively excluded after normalisation.
    if max_dist is not None:
        ned_sta_poi = ned_sta_poi.where(ned_sta_poi < max_dist)

    w_ned = 1 / np.power(ned_sta_poi, weight_power)
    w_ned /= w_ned.sum("sta") + lim_effective

    if taper is not None:
        w_ned = w_ned * taper

    # min_count=1: a POI with no contributing station returns NaN rather than 0.
    return (w_ned * sta_res).sum("sta", min_count=1)


class NudgeTowardObservation(Filter):
    """Nudge the forecast initial condition toward surface station observations.

    Interpolation of Residuals with barrier-aware effective distances and an
    optional station reliability check; see the module docstring for the
    algorithm.

    Reads station observations from a Parquet file written by
    RetrieveObservation and CleanObservation. Nudging is applied once, to the initial-condition
    time step; later calls pass all fields through unchanged.

    Parameters
    ----------
    obs_path : str
        Cleaned station observations Parquet file. Required columns:
        ``latitude``, ``longitude``, ``elevation``, and one column per nudged
        variable (e.g. ``2t``, ``10u``), all in SI units. Stations with a NaN
        ``elevation`` are skipped for variables with an elevation correction
    icon_grid_file : str
        ICON grid NetCDF (variables ``clat``/``clon`` in radians)
    d_eff_file : str
        Precomputed effective distances (d_eff) for the station catalog:
        ``d_eff_poi`` (dims ``poi``, ``sta``) and ``d_eff_sta`` (dims
        ``sta_i``, ``sta``, self-distances +inf, used by the reliability
        check). Built by ``scripts/generate_d_eff_cache.py``. A POI or
        station missing from the cache raises ``ValueError``.
    icon_orog_file : str
        ICON extpar NetCDF (variable ``topography_c``), the model's own
        native orography, on the same grid and cell order as the ICON grid.
        Used as model elevation in the lapse-rate correction.
    weight_power : float
        IDW distance-decay exponent; higher concentrates weight on the
        nearest station.
    max_dist : float
        Station influence radius [m].
    variable_overrides : dict, optional
        Per-variable *d_eff_file* and/or *max_dist* [m], keyed by GRIB
        shortName, e.g. ``{"U_10M": {"d_eff_file": "...", "max_dist": 20000.0}}``.
    lim_effective : float
        Virtual zero-residual station weight added to the normalisation
        denominator; 0 = pure IDW.
    use_reliability_check : bool
        If ``True``, scale each station's influence radius by a leave-one-out
        spatial-consistency check (see module docstring, "Station
        reliability"). If ``False``, every station uses *max_dist*.
    number_of_std : float
        Tukey biweight rejection threshold in robust (median/MAD) standard
        deviations — a station at or beyond this many robust sigmas gets
        reliability=0. Only used when *use_reliability_check* is ``True``.
    reliability_min_dist_frac : float
        Minimum fraction of *max_dist* every station keeps even at
        reliability=0. Must be in [0, 1]. Only used when
        *use_reliability_check* is ``True``.
    reliability_eps : float
        Lower bound of the robust (MAD-based) scale, avoiding division by
        zero. Only used when *use_reliability_check* is ``True``.
    temperature_lapse_rate : float
        Standard-atmosphere lapse rate [K/m] reducing station observations to
        the model's elevation before differencing:
        ``obs_corrected = obs - temperature_lapse_rate * (elev_model_at_cell - elev_sta)``.
        Applied only to *temperature_lapse_rate_vars*.
    temperature_lapse_rate_vars : list of str, optional
        GRIB shortNames *temperature_lapse_rate* applies to.
    pressure_lapse_rate : float
        Vertical pressure gradient [Pa/m] used like *temperature_lapse_rate*,
        applied only to *pressure_lapse_rate_vars* (default: 11.5 Pa/m).
    pressure_lapse_rate_vars : list of str, optional
        GRIB shortNames *pressure_lapse_rate* applies to.
    run_mode : str
        ``'depl'``: ref_time = minimum valid_time across all fields.
        ``'devt'``: ref_time = valid_time of the first field.
    holdout_fraction : float, optional
        Fraction of stations to withhold for cross-validation. Mutually
        exclusive with *exclude_stations*.
    holdout_seed : int
        RNG seed for station holdout (default 42).
    exclude_stations : list of str, optional
        Station nat_abbr identifiers to unconditionally exclude. Mutually
        exclusive with *holdout_fraction*.
    write_diagnostics : bool
        If ``True``, write one NetCDF per nudged variable with the station
        residuals before/after nudging, the reliability check results and the
        gridded correction (see ``_write_diagnostics``), for plotting offline.
        Failures are logged and do not affect the nudging.
    diagnostics_dir : str
        Directory for the diagnostics files,
        ``nudging_diag_{shortname}_{ref_time:%Y%m%d%H%M}.nc``. Created when
        the first file is written. Only used when *write_diagnostics* is
        ``True``.
    """

    def __init__(
        self,
        obs_path: str,
        icon_grid_file: str,
        d_eff_file: str,
        icon_orog_file: str,
        weight_power: float = 2.0,
        max_dist: float = 50000.0,
        variable_overrides: Optional[dict] = None,
        lim_effective: float = 0.0,
        use_reliability_check: bool = False,
        number_of_std: float = 5.0,
        reliability_min_dist_frac: float = 0.05,
        reliability_eps: float = 1e-6,
        write_diagnostics: bool = False,
        diagnostics_dir: str = "nudging_diagnostics",
        temperature_lapse_rate: float = _DEFAULT_TEMPERATURE_LAPSE_RATE,
        temperature_lapse_rate_vars: Optional[list] = None,
        pressure_lapse_rate: float = _DEFAULT_PRESSURE_LAPSE_RATE,
        pressure_lapse_rate_vars: Optional[list] = None,
        nudge_variables: Optional[list] = None,
        run_mode: str = "depl",
        holdout_fraction: Optional[float] = None,
        holdout_seed: int = 42,
        exclude_stations: Optional[list] = None,
    ):
        if run_mode not in ("devt", "depl"):
            raise ValueError(f"run_mode must be 'devt' or 'depl', got {run_mode!r}")
        if holdout_fraction is not None and exclude_stations is not None:
            raise ValueError("holdout_fraction and exclude_stations are mutually exclusive.")
        if holdout_fraction is not None and not (0.0 <= holdout_fraction <= 1.0):
            raise ValueError(f"holdout_fraction must be in [0, 1], got {holdout_fraction!r}")
        if not (0.0 <= reliability_min_dist_frac <= 1.0):
            raise ValueError(f"reliability_min_dist_frac must be in [0, 1], got {reliability_min_dist_frac!r}")
        if number_of_std <= 0:
            raise ValueError(f"number_of_std must be > 0, got {number_of_std!r}")

        if max_dist <= 0:
            raise ValueError(f"max_dist must be > 0, got {max_dist!r}")
        if weight_power <= 0:
            raise ValueError(f"weight_power must be > 0, got {weight_power!r}")
        if lim_effective < 0:
            raise ValueError(f"lim_effective must be >= 0, got {lim_effective!r}")

        self.obs_path = Path(obs_path)
        self.icon_grid_file = Path(icon_grid_file)
        self.d_eff_file = Path(d_eff_file)
        self.icon_orog_file = Path(icon_orog_file)
        self.weight_power = weight_power
        self.max_dist = max_dist / _M_PER_KM  # m -> km; distances are in km internally
        self.lim_effective = lim_effective
        self.use_reliability_check = use_reliability_check
        self.number_of_std = number_of_std
        self.reliability_min_dist_frac = reliability_min_dist_frac
        self.reliability_eps = reliability_eps
        self.write_diagnostics = write_diagnostics
        self.diagnostics_dir = Path(diagnostics_dir)
        self.temperature_lapse_rate = temperature_lapse_rate
        self.temperature_lapse_rate_vars = (
            frozenset(temperature_lapse_rate_vars)
            if temperature_lapse_rate_vars is not None
            else _DEFAULT_TEMPERATURE_LAPSE_RATE_VARS
        )
        self.pressure_lapse_rate = pressure_lapse_rate
        self.pressure_lapse_rate_vars = (
            frozenset(pressure_lapse_rate_vars)
            if pressure_lapse_rate_vars is not None
            else _DEFAULT_PRESSURE_LAPSE_RATE_VARS
        )
        self.run_mode = run_mode
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed
        self.exclude_stations = list(exclude_stations) if exclude_stations is not None else None
        self._nudging_done = False
        self._reliability_diag = {}

        if nudge_variables is not None:
            unknown = set(nudge_variables) - _ICON_TO_ECMWF_NAME.keys()
            if unknown:
                raise ValueError(f"Unknown nudge variables: {unknown}. Valid: {list(_ICON_TO_ECMWF_NAME)}")
            self.ecmwf_names = {v: _ICON_TO_ECMWF_NAME[v] for v in nudge_variables}
        else:
            self.ecmwf_names = dict(_ICON_TO_ECMWF_NAME)

        # ── Per-variable d_eff_file/max_dist ───────────────────────────────
        self.variable_overrides = dict(variable_overrides) if variable_overrides is not None else {}
        unknown_override_vars = set(self.variable_overrides) - set(self.ecmwf_names)
        if unknown_override_vars:
            raise ValueError(
                f"variable_overrides references variable(s) not in the active "
                f"nudge set: {sorted(unknown_override_vars)}. Active variables: "
                f"{list(self.ecmwf_names)}"
            )
        for _var, _override in self.variable_overrides.items():
            _unknown_keys = set(_override) - {"d_eff_file", "max_dist"}
            if _unknown_keys:
                raise ValueError(
                    f"variable_overrides[{_var!r}] has unknown key(s) "
                    f"{sorted(_unknown_keys)}; only 'd_eff_file' and 'max_dist' "
                    "are supported."
                )

        self._max_dist_by_var = {
            var: (
                self.variable_overrides[var]["max_dist"] / _M_PER_KM
                if "max_dist" in self.variable_overrides.get(var, {})
                else self.max_dist
            )
            for var in self.ecmwf_names
        }
        for _var, _md in self._max_dist_by_var.items():
            if _md <= 0:
                raise ValueError(f"variable_overrides[{_var!r}]['max_dist'] must be > 0, got {_md!r}")
        self._d_eff_file_by_var = {
            var: Path(self.variable_overrides.get(var, {}).get("d_eff_file", self.d_eff_file))
            for var in self.ecmwf_names
        }

        # Static data, loaded once.
        self._load_icon_grid()
        self._load_icon_orog()
        self._project_icon_grid()
        # One entry per distinct cache file; several variables may share one.
        self._d_eff_caches = {path: self._load_d_eff_cache(path) for path in set(self._d_eff_file_by_var.values())}

        LOG.info(
            "NudgeTowardObservation initialised: variables=%s, max_dist=%s km, "
            "weight_power=%.1f, d_eff_file=%s, "
            "temperature_lapse_rate=%.5f K/m (vars=%s), pressure_lapse_rate=%.2f Pa/m (vars=%s), "
            "use_reliability_check=%s, number_of_std=%.2f, "
            "reliability_min_dist_frac=%.3f, min radius=%s km",
            list(self.ecmwf_names.keys()),
            {v: self._max_dist_by_var[v] for v in self.ecmwf_names},
            self.weight_power,
            {v: str(self._d_eff_file_by_var[v]) for v in self.ecmwf_names},
            self.temperature_lapse_rate,
            sorted(self.temperature_lapse_rate_vars),
            self.pressure_lapse_rate,
            sorted(self.pressure_lapse_rate_vars),
            self.use_reliability_check,
            self.number_of_std,
            self.reliability_min_dist_frac,
            {v: round(self.reliability_min_dist_frac * self._max_dist_by_var[v], 2) for v in self.ecmwf_names},
        )
        super().__init__()

    def _elevation_rate(self, shortname: str) -> Optional[float]:
        """Elevation-reduction rate for *shortname*'s observations (see
        *temperature_lapse_rate_vars*/*pressure_lapse_rate_vars*), or ``None`` if this
        variable gets no elevation correction."""
        if shortname in self.temperature_lapse_rate_vars:
            return self.temperature_lapse_rate
        if shortname in self.pressure_lapse_rate_vars:
            return self.pressure_lapse_rate
        return None

    # ── Static data loaders ───────────────────────────────────────────────────

    def _load_icon_grid(self) -> None:
        ds = xr.open_dataset(self.icon_grid_file)
        # clat/clon are stored in radians in the ICON grid file.
        self._lat_icon = np.degrees(ds["clat"].values).ravel()
        self._lon_icon = np.degrees(ds["clon"].values).ravel()
        ds.close()
        LOG.info(
            "ICON grid loaded: %d cells from %s",
            len(self._lat_icon),
            self.icon_grid_file,
        )

    def _load_icon_orog(self) -> None:
        """Load ICON's native orography, indexed positionally by ICON cell."""
        ds_orog = xr.open_dataset(self.icon_orog_file)
        self._icon_orog = ds_orog["topography_c"].values.astype(np.float32).ravel()
        ds_orog.close()
        # Checks the cell count only; a different cell order would go undetected.
        if len(self._icon_orog) != len(self._lat_icon):
            raise ValueError(
                f"icon_orog_file {self.icon_orog_file} has {len(self._icon_orog)} cells "
                f"but the ICON grid ({self.icon_grid_file}) has {len(self._lat_icon)}."
            )
        LOG.info("ICON native orography loaded from %s", self.icon_orog_file)

    def _project_icon_grid(self) -> None:
        """Project the ICON grid to LV95 [km] and build the nearest-cell tree."""
        self._wgs84_to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

        grid_x, grid_y = self._wgs84_to_lv95.transform(self._lon_icon, self._lat_icon)
        self._grid_xy_km = np.c_[grid_x, grid_y] / _M_PER_KM

        # Nearest-cell lookup for stations.
        self._grid_tree = cKDTree(self._grid_xy_km)

        LOG.info(
            "ICON grid projected to LV95: x=[%.1f, %.1f] km, y=[%.1f, %.1f] km",
            self._grid_xy_km[:, 0].min(),
            self._grid_xy_km[:, 0].max(),
            self._grid_xy_km[:, 1].min(),
            self._grid_xy_km[:, 1].max(),
        )

    def _load_d_eff_cache(self, path: Path) -> dict:
        """Load a d_eff cache file into memory, with
        indexes for slicing by POI and station."""
        if not path.exists():
            raise FileNotFoundError(
                f"d_eff cache not found: {path}. Build it with "
                "scripts/generate_d_eff_cache.py and point d_eff_file (or "
                "variable_overrides) at it."
            )
        ds = xr.open_dataset(path)
        d_eff_poi_full = ds["d_eff_poi"].load()
        d_eff_sta_full = ds["d_eff_sta"].load()
        ds.close()
        LOG.info(
            "d_eff loaded from %s: POI x station %s, station x station %s",
            path,
            d_eff_poi_full.shape,
            d_eff_sta_full.shape,
        )
        return {
            "d_eff_poi_full": d_eff_poi_full,
            "d_eff_sta_full": d_eff_sta_full,
            "poi_index": pd.Index(d_eff_poi_full["poi"].values),
            "sta_index": pd.Index(d_eff_poi_full["sta"].values),
            "poi_set": set(d_eff_poi_full["poi"].values.tolist()),
            "sta_set": set(d_eff_poi_full["sta"].values.tolist()),
        }

    def _get_d_eff_poi(self, shortname: str, dom_idx: np.ndarray, sta_ids: list) -> xr.DataArray:
        """POI<->station d_eff for the given POIs and stations, from the cache
        configured for *shortname*."""
        d_eff_file = self._d_eff_file_by_var[shortname]
        cache = self._d_eff_caches[d_eff_file]
        missing_poi = set(dom_idx) - cache["poi_set"]
        if missing_poi:
            raise ValueError(
                f"{len(missing_poi)} POI(s) not covered by the precomputed "
                f"d_eff cache ({d_eff_file}) used for '{shortname}'. "
                "Rebuild the cache for this domain and ICON grid."
            )
        missing_sta = set(sta_ids) - cache["sta_set"]
        if missing_sta:
            raise ValueError(
                f"Station(s) {sorted(missing_sta)} not covered by the "
                f"precomputed d_eff cache ({d_eff_file}) used for "
                f"'{shortname}' — the station catalog changed since the "
                "cache was built. Rebuild the cache."
            )
        # Positional lookup, in the requested order.
        poi_pos = cache["poi_index"].get_indexer(dom_idx)
        sta_pos = cache["sta_index"].get_indexer(sta_ids)
        values = cache["d_eff_poi_full"].values[np.ix_(poi_pos, sta_pos)]
        return xr.DataArray(values, dims=["poi", "sta"], coords={"poi": dom_idx, "sta": sta_ids})

    def _get_d_eff_sta(self, shortname: str, sta_ids: list) -> xr.DataArray:
        """Station<->station d_eff for the reliability check, from the cache
        configured for *shortname*."""
        d_eff_file = self._d_eff_file_by_var[shortname]
        cache = self._d_eff_caches[d_eff_file]
        missing = set(sta_ids) - cache["sta_set"]
        if missing:
            raise ValueError(
                f"Station(s) {sorted(missing)} not covered by the "
                f"precomputed d_eff cache ({d_eff_file}) used for "
                f"'{shortname}' — the station catalog changed since the "
                "cache was built. Rebuild the cache."
            )
        return cache["d_eff_sta_full"].sel(sta_i=list(sta_ids), sta=list(sta_ids)).rename({"sta_i": "poi"})

    # ── Filter entry point ────────────────────────────────────────────────────

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply IoR nudging to the initial-condition fields; pass subsequent calls through."""
        if self._nudging_done:
            return data

        ref_time = (
            data[0].datetime()["valid_time"]
            if self.run_mode == "devt"
            else min(f.datetime()["valid_time"] for f in data)
        )
        LOG.info("Nudging initial condition at %s", ref_time)

        all_stations = self._load_stations()
        LOG.info("Stations loaded: %d", len(all_stations))
        stations = self._apply_holdout(all_stations)
        LOG.info("Stations after holdout: %d", len(stations))
        # Used only for the diagnostics files.
        held_out_stations = all_stations.drop(index=stations.index)

        nudged = {}
        for field in data.sel(shortName=list(self.ecmwf_names.keys())):
            shortname = field.metadata("shortName")

            if field.datetime()["valid_time"] != ref_time:
                continue

            ecmwf_name = self.ecmwf_names[shortname]
            if ecmwf_name not in stations.columns or stations[ecmwf_name].isna().all():
                LOG.warning("No observations for '%s', skipping", shortname)
                continue

            corrected = self._nudge_field(field, stations, shortname, ecmwf_name, ref_time, held_out_stations)
            nudged[shortname] = new_field_from_numpy(
                corrected,
                template=field,
                validityDate=field.metadata("validityDate"),
                validityTime=field.metadata("validityTime"),
                dataDate=field.metadata("dataDate"),
                dataTime=field.metadata("dataTime"),
            )

        result = [nudged.get(f.metadata("shortName"), f) if f.datetime()["valid_time"] == ref_time else f for f in data]
        self._nudging_done = True
        LOG.info("Nudging complete: %d/%d fields updated", len(nudged), len(result))
        return new_fieldlist_from_list(result)

    # ── Core algorithm ────────────────────────────────────────────────────────

    def _usable_stations(
        self,
        df: pd.DataFrame,
        ecmwf_name: str,
        shortname: str,
        *,
        log: bool = False,
    ) -> pd.Series:
        """Rows of `df` with an observation in `ecmwf_name` and, for variables with
        an elevation correction, a known station elevation."""
        usable = df[ecmwf_name].notna()
        if self._elevation_rate(shortname) is not None:
            no_elev = usable & df["elevation"].isna()
            if no_elev.any():
                if log:
                    LOG.warning(
                        "'%s': skipping %d station(s) with unknown elevation: %s",
                        shortname,
                        int(no_elev.sum()),
                        df.index[no_elev].tolist(),
                    )
                usable &= ~no_elev
        return usable

    def _reduce_obs_to_model_elevation(
        self,
        shortname: str,
        obs: np.ndarray,
        grid_index: np.ndarray,
        sta_elev: np.ndarray,
        *,
        log: bool = False,
    ) -> np.ndarray:
        """Reduce `obs` to the model's elevation at each station's snapped ICON
        cell (`grid_index`) via *temperature_lapse_rate*/*pressure_lapse_rate*,
        so the residual reflects model bias rather than the
        elevation mismatch between station and ICON's (smoothed) orography.
        Returns `obs` unchanged for variables with no configured rate.
        """
        obs_lr = obs
        rate = self._elevation_rate(shortname)
        if rate is not None:
            elev_model = self._icon_orog[grid_index]
            obs_lr = obs - rate * (elev_model - sta_elev)
            if log:
                LOG.debug(
                    "Elevation-reduction correction for '%s' (rate=%.5f): mean "
                    "elev_model−elev_sta = %.0f m, mean |correction| = %.3f",
                    shortname,
                    rate,
                    float(np.mean(elev_model - sta_elev)),
                    float(np.mean(np.abs(obs_lr - obs))),
                )
        return obs_lr

    def _nudge_field(
        self,
        field: ekd.Field,
        stations: pd.DataFrame,
        shortname: str,
        ecmwf_name: str,
        ref_time=None,
        held_out_stations: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Nudge a single field toward the station observations; return the
        corrected 1-D array.

        ``ref_time`` and ``held_out_stations`` are only used for the
        diagnostics files.
        """
        B_flat = np.asarray(field.values, dtype=np.float32).ravel()

        max_dist = self._max_dist_by_var[shortname]

        # ── Valid stations ─────────────────────────────────────────────────
        valid = self._usable_stations(stations, ecmwf_name, shortname, log=True)
        st_lat = stations.loc[valid, "latitude"].to_numpy()
        st_lon = stations.loc[valid, "longitude"].to_numpy()
        st_obs = stations.loc[valid, ecmwf_name].to_numpy()
        st_elev = stations.loc[valid, "elevation"].to_numpy(dtype=float)
        sta_ids = stations.loc[valid].index.tolist()

        # ── Coordinate projection ──────────────────────────────────────────
        grid_xy = self._grid_xy_km
        sta_x, sta_y = self._wgs84_to_lv95.transform(st_lon, st_lat)
        sta_xy = np.c_[sta_x, sta_y] / _M_PER_KM

        # ── POI domain ─────────────────────────────────────────────────────
        # Grid cells within max_dist of the station bounding box.
        sta_x_min = sta_xy[:, 0].min() - max_dist
        sta_x_max = sta_xy[:, 0].max() + max_dist
        sta_y_min = sta_xy[:, 1].min() - max_dist
        sta_y_max = sta_xy[:, 1].max() + max_dist
        dom_mask = (
            (grid_xy[:, 0] >= sta_x_min)
            & (grid_xy[:, 0] <= sta_x_max)
            & (grid_xy[:, 1] >= sta_y_min)
            & (grid_xy[:, 1] <= sta_y_max)
        )
        dom_idx = np.where(dom_mask)[0]
        poi_xy = grid_xy[dom_idx]
        n_poi = len(dom_idx)

        # ── Residuals at stations ──────────────────────────────────────────
        # Residuals at each station's nearest grid cell.
        _, gi = self._grid_tree.query(sta_xy, k=1)

        st_obs_lr = self._reduce_obs_to_model_elevation(shortname, st_obs, gi, st_elev, log=True)

        r_at_st = B_flat[gi] - st_obs_lr  # positive when model > observation

        sta_res = xr.Dataset(
            {shortname: xr.DataArray(r_at_st.astype(np.float32), dims=["sta"], coords={"sta": sta_ids})}
        )

        # ── Euclidean distance matrix ──────────────────────────────────────
        # float32 to halve the memory of the n_poi x n_sta matrix.
        d_euc_mat = np.sqrt(
            ((poi_xy[:, None, :].astype(np.float32) - sta_xy[None, :, :].astype(np.float32)) ** 2).sum(axis=-1)
        ).astype(np.float32)

        # ── Barrier-aware distances ────────────────────────────────────────
        ned_sta_poi = self._get_d_eff_poi(shortname, dom_idx, sta_ids)

        # ── Station reliability → per-station influence radius ─────────────
        if self.use_reliability_check:
            reliability = self._compute_reliability(
                shortname,
                sta_ids,
                r_at_st,
                sta_res,
                max_dist,
            )
            min_dist = self.reliability_min_dist_frac * max_dist
            station_max_dist = min_dist + (max_dist - min_dist) * reliability
        else:
            station_max_dist = max_dist

        # ── Per-station linear taper ───────────────────────────────────────
        d_euc_da = xr.DataArray(d_euc_mat, dims=["poi", "sta"], coords={"poi": dom_idx, "sta": sta_ids})
        pair_taper = (1.0 - (d_euc_da / station_max_dist).clip(min=0.0, max=1.0)).astype(np.float32)

        # ── ned_interp ─────────────────────────────────────────────────────
        # The radius applies to d_eff, so a POI close to a station but behind
        # a ridge may get no correction from it.
        result = ned_interp(
            sta_res,
            ned_sta_poi,
            max_dist=station_max_dist,
            weight_power=self.weight_power,
            lim_effective=self.lim_effective,
            taper=pair_taper,
        )

        # NaN (no station in range) -> no correction.
        correction = np.nan_to_num(result[shortname].values, nan=0.0)

        corrected_flat = B_flat.copy()
        corrected_flat[dom_idx] -= correction

        if self.write_diagnostics:
            self._write_diagnostics(
                shortname,
                ecmwf_name,
                ref_time,
                stations,
                held_out_stations,
                B_flat,
                corrected_flat,
                dom_idx,
                correction,
                sta_ids,
                st_lat,
                st_lon,
                gi,
                st_obs_lr,
                station_max_dist,
            )

        if self.use_reliability_check:
            n_shrunk = int((station_max_dist < max_dist).sum())
            reliability_note = (
                f"reliability check: number_of_std={self.number_of_std:.2f}, "
                f"{n_shrunk}/{len(st_lat)} station(s) with a shrunk radius"
            )
        else:
            reliability_note = "reliability check: disabled"
        LOG.info(
            "Nudged '%s': %d stations, %d POIs, max |correction| = %.4f (%s)",
            shortname,
            len(st_lat),
            n_poi,
            float(np.abs(correction).max()),
            reliability_note,
        )

        return corrected_flat

    def _write_diagnostics(
        self,
        shortname: str,
        ecmwf_name: str,
        ref_time,
        stations: pd.DataFrame,
        held_out_stations: Optional[pd.DataFrame],
        B_flat: np.ndarray,
        corrected_flat: np.ndarray,
        dom_idx: np.ndarray,
        correction: np.ndarray,
        sta_ids: list,
        st_lat: np.ndarray,
        st_lon: np.ndarray,
        gi: np.ndarray,
        st_obs_lr: np.ndarray,
        station_max_dist: Union[float, xr.DataArray],
    ) -> None:
        """Write everything needed to plot the nudging of one variable to
        ``{diagnostics_dir}/nudging_diag_{shortname}_{ref_time:%Y%m%d%H%M}.nc``.

        Dimensions:
        - ``station``: stations used for nudging (holdin) followed by those
          withheld by ``_apply_holdout`` (holdout, an independent check), with
          the residuals before (background − obs) and after (corrected − obs)
          nudging. ``reliability``, ``loo_prediction``, ``loo_discrepancy``
          and ``robust_z`` are NaN for holdout stations and when the
          reliability check is disabled; ``radius_km`` is the influence radius
          used (NaN for holdout stations).
        - ``cell``: ICON cells of the POI domain, with the applied correction
          (background − corrected); it is zero outside this domain.
        - ``qc_station``: stations whose value was removed by QC.

        Never raises: failures are logged so they cannot affect the nudging.
        """
        try:
            ids = list(sta_ids)
            lat = [st_lat]
            lon = [st_lon]
            res_pre = [B_flat[gi] - st_obs_lr]
            res_post = [corrected_flat[gi] - st_obs_lr]

            if held_out_stations is not None and len(held_out_stations) and ecmwf_name in held_out_stations.columns:
                ho_valid = self._usable_stations(held_out_stations, ecmwf_name, shortname)
                if ho_valid.any():
                    ho_lat = held_out_stations.loc[ho_valid, "latitude"].to_numpy()
                    ho_lon = held_out_stations.loc[ho_valid, "longitude"].to_numpy()
                    ho_obs = held_out_stations.loc[ho_valid, ecmwf_name].to_numpy()
                    ho_elev = held_out_stations.loc[ho_valid, "elevation"].to_numpy(dtype=float)

                    ho_x, ho_y = self._wgs84_to_lv95.transform(ho_lon, ho_lat)
                    _, ho_gi = self._grid_tree.query(np.c_[ho_x, ho_y] / _M_PER_KM, k=1)
                    ho_obs_lr = self._reduce_obs_to_model_elevation(shortname, ho_obs, ho_gi, ho_elev)

                    ids += held_out_stations.index[ho_valid].tolist()
                    lat.append(ho_lat)
                    lon.append(ho_lon)
                    res_pre.append(B_flat[ho_gi] - ho_obs_lr)
                    res_post.append(corrected_flat[ho_gi] - ho_obs_lr)

            n_in = len(sta_ids)
            n_ho = len(ids) - n_in

            def _holdin_only(values) -> np.ndarray:
                # Holdin values followed by NaN for the holdout stations.
                return np.concatenate([np.asarray(values, dtype=np.float32), np.full(n_ho, np.nan, np.float32)])

            rel = self._reliability_diag.get(shortname, {}) if self.use_reliability_check else {}
            nan_in = np.full(n_in, np.nan)
            radius = np.broadcast_to(np.asarray(station_max_dist, dtype=np.float32), (n_in,))

            # QC sets values to NaN but keeps the row, so coordinates are available.
            qc_flag_name = f"{ecmwf_name}_qc_dropped"
            if qc_flag_name in stations.columns:
                qc_mask = stations[qc_flag_name].fillna(False).astype(bool)
            else:
                qc_mask = pd.Series(False, index=stations.index)

            ds = xr.Dataset(
                {
                    "station_latitude": ("station", np.concatenate(lat)),
                    "station_longitude": ("station", np.concatenate(lon)),
                    "is_holdout": ("station", np.r_[np.zeros(n_in, np.int8), np.ones(n_ho, np.int8)]),
                    "residual_pre": ("station", np.concatenate(res_pre).astype(np.float32)),
                    "residual_post": ("station", np.concatenate(res_post).astype(np.float32)),
                    "reliability": ("station", _holdin_only(rel.get("reliability", nan_in))),
                    "loo_prediction": ("station", _holdin_only(rel.get("r_hat", nan_in))),
                    "loo_discrepancy": ("station", _holdin_only(rel.get("e", nan_in))),
                    "robust_z": ("station", _holdin_only(rel.get("u", nan_in))),
                    "radius_km": ("station", _holdin_only(radius)),
                    "cell_latitude": ("cell", self._lat_icon[dom_idx].astype(np.float32)),
                    "cell_longitude": ("cell", self._lon_icon[dom_idx].astype(np.float32)),
                    "correction": ("cell", np.asarray(correction, dtype=np.float32)),
                    "qc_latitude": ("qc_station", stations.loc[qc_mask, "latitude"].to_numpy()),
                    "qc_longitude": ("qc_station", stations.loc[qc_mask, "longitude"].to_numpy()),
                },
                coords={
                    "station": np.array(ids, dtype=str),
                    "cell": dom_idx,
                    "qc_station": np.array(stations.index[qc_mask].tolist(), dtype=str),
                },
                attrs={
                    "variable": shortname,
                    "ref_time": ref_time.isoformat() if ref_time is not None else "unknown",
                    "residual_pre_definition": "background - observation (observation reduced to model elevation)",
                    "residual_post_definition": "corrected - observation (observation reduced to model elevation)",
                    "correction_definition": "background - corrected",
                    "max_dist_km": float(self._max_dist_by_var[shortname]),
                    "weight_power": float(self.weight_power),
                    "lim_effective": float(self.lim_effective),
                    "use_reliability_check": int(self.use_reliability_check),
                    "number_of_std": float(self.number_of_std),
                    "reliability_min_dist_frac": float(self.reliability_min_dist_frac),
                    "d_eff_file": str(self._d_eff_file_by_var[shortname]),
                },
            )

            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            ref_time_str = ref_time.strftime("%Y%m%d%H%M") if ref_time is not None else "unknown"
            out_path = self.diagnostics_dir / f"nudging_diag_{shortname}_{ref_time_str}.nc"
            ds.to_netcdf(out_path)
            LOG.info("Wrote nudging diagnostics for '%s' to %s", shortname, out_path)
        except Exception:
            LOG.exception(
                "Writing nudging diagnostics failed for '%s'; continuing without them "
                "(the nudging correction itself is unaffected).",
                shortname,
            )

    def _compute_reliability(
        self,
        shortname: str,
        sta_ids: list,
        r_at_st: np.ndarray,
        sta_res: xr.Dataset,
        max_dist: float,
    ) -> xr.DataArray:
        """Leave-one-out spatial-consistency check (see module docstring,
        "Station reliability").

        Scores how well each station's residual agrees with the IDW prediction
        from its neighbours.

        Returns an xr.DataArray (dims=["sta"]) of reliability in [0, 1].
        """
        n_sta = len(sta_ids)

        # ── Station <-> station barrier-aware distances ────────────────────
        ned_ss = self._get_d_eff_sta(shortname, sta_ids)

        # ── Leave-one-out neighbour prediction ──────────────────────────────
        # All other stations are trusted equally here; the station itself is
        # excluded by its infinite self-distance.
        r_hat = ned_interp(
            sta_res,
            ned_ss,
            max_dist=max_dist,
            weight_power=self.weight_power,
            lim_effective=self.lim_effective,
        )
        r_hat_at_st = r_hat[shortname].rename({"poi": "sta"}).sel(sta=sta_ids).values
        e = r_at_st - r_hat_at_st

        finite = np.isfinite(e)
        n_isolated = int((~finite).sum())
        if n_isolated:
            LOG.warning(
                "compute_reliability('%s'): %d/%d station(s) have no neighbour "
                "within max_dist=%.2f km for the leave-one-out check (e=NaN); "
                "left at reliability=1.0, excluded from the robust median/MAD "
                "so they don't corrupt every other station's reliability.",
                shortname,
                n_isolated,
                n_sta,
                max_dist,
            )
        if finite.any():
            med_e = float(np.median(e[finite]))
            mad = float(np.median(np.abs(e[finite] - med_e)))
        else:
            med_e, mad = 0.0, 0.0
        scale = max(_MAD_TO_STD * mad, self.reliability_eps)

        u = np.zeros(n_sta, dtype=np.float64)
        u[finite] = (e[finite] - med_e) / scale
        reliability_vals = np.ones(n_sta, dtype=np.float64)
        reliability_vals[finite] = np.clip(1.0 - (u[finite] / self.number_of_std) ** 2, 0.0, None) ** 2

        n_flagged = int((reliability_vals < _FLAG_RELIABILITY).sum())
        LOG.debug(
            "compute_reliability('%s'): %d stations, median(e)=%.4f, MAD(e)=%.4f, "
            "scale=%.4f, %d station(s) with reliability < %.2f",
            shortname,
            n_sta,
            med_e,
            mad,
            scale,
            n_flagged,
            _FLAG_RELIABILITY,
        )

        # For the diagnostics files.
        self._reliability_diag[shortname] = {
            "r_hat": r_hat_at_st,
            "e": e,
            "u": u,
            "reliability": reliability_vals,
        }

        return xr.DataArray(
            reliability_vals.astype(np.float32),
            dims=["sta"],
            coords={"sta": sta_ids},
        )

    # ── Holdout and station loading ───────────────────────────────────────────

    def _apply_holdout(self, stations: pd.DataFrame) -> pd.DataFrame:
        """Remove stations from the nudging set according to holdout configuration."""
        if self.exclude_stations is not None:
            before = len(stations)
            missing = [s for s in self.exclude_stations if s not in stations.index]
            if missing:
                LOG.warning("Excluded station IDs not found in observations: %s", missing)
            stations = stations.drop(index=[s for s in self.exclude_stations if s in stations.index])
            LOG.info(
                "Excluded %d station(s) by ID: %s",
                before - len(stations),
                self.exclude_stations,
            )

        elif self.holdout_fraction is not None:
            if self.holdout_fraction == 0.0:
                LOG.info("holdout_fraction=0: all stations used.")
            elif self.holdout_fraction == 1.0:
                LOG.info("holdout_fraction=1: all stations withheld, nudging will have no effect.")
                stations = stations.iloc[0:0]
            else:
                n_holdout = round(len(stations) * self.holdout_fraction)
                if n_holdout == 0:
                    LOG.warning(
                        "holdout_fraction=%.4f rounds to 0 station(s) held out "
                        "of %d — no cross-validation holdout set will be "
                        "available this run.",
                        self.holdout_fraction,
                        len(stations),
                    )
                rng = np.random.default_rng(self.holdout_seed)
                held_out = rng.choice(stations.index, size=n_holdout, replace=False)
                stations = stations.drop(index=held_out)
                LOG.info(
                    "Held out %d/%d station(s) (%.0f%%, seed=%d): %s",
                    n_holdout,
                    n_holdout + len(stations),
                    self.holdout_fraction * 100,
                    self.holdout_seed,
                    list(held_out),
                )

        return stations

    def _load_stations(self) -> pd.DataFrame:
        """Read pre-fetched station observations from the configured Parquet file."""
        if not self.obs_path.exists():
            raise FileNotFoundError(f"Observation file not found: {self.obs_path}")
        stations = pd.read_parquet(self.obs_path)
        missing_cols = {"latitude", "longitude", "elevation"} - set(stations.columns)
        if missing_cols:
            raise ValueError(
                f"Observations Parquet {self.obs_path} is missing required column(s) {sorted(missing_cols)}."
            )
        return stations
