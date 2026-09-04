"""
NudgeTowardObservation — Interpolation of Residuals (IoR): spreads station
residuals (background − observation) onto the ICON grid via ned_interp
(barrier-aware IDW + topographic similarity) and subtracts the result,
nudging the background toward observations.

Algorithm per variable
----------------------
1. Project ICON grid + station coords to Swiss LV95 (EPSG:2056, exact
   metric CRS, km) rather than an equirectangular approximation.
2. Restrict POIs to a buffer around the station bounding box (max_dist).
3. Compute the Euclidean distance matrix (n_poi x n_sta), km.
4. Look up each (POI, station) pair's barrier- and elevation-aware
   effective distance (d_eff) from the precomputed cache (d_eff_file) —
   never computed live; see barrier_distances(), run offline to build it
   (notebooks/nudging_analysis_v11.ipynb, "Precomputed d_eff cache").
5. Compute topographic similarity per (POI, station) pair from TPI/slope/
   elevation descriptors, weighted by each descriptor's |Pearson corr|
   with the station residuals (see ned_interp).
6. Spread residuals to the POI grid via ned_interp: IDW
   (1 / d_eff^weight_power) x topographic similarity, floored at
   min_topo_w.
7. Multiply each station's normalised weight by a linear taper fading to
   zero at THAT station's own max_dist (v4.1, see _nudge_field).
8. Subtract the resulting correction from the background field.

The taper (step 7) uses raw Euclidean distance, not d_eff, and is applied
AFTER ned_interp's weight normalisation, not merged into the raw weights:
with only one contributing station the normalised weight is always exactly
1 regardless of the raw weight, so a pre-normalisation taper would cancel
out silently (see ned_interp's "v4.1" note).

Units: max_dist is given in **meters** at the constructor boundary and
converted to km once in __init__ (self.max_dist / self._max_dist_by_var are
km from then on — everywhere else in this file, including barrier_distances
and the d_eff cache, stays km-only, unaffected by this input-side conversion).
elev_scale/elev_diff_scale are no longer parameters here — they're baked
into the offline-built d_eff cache.

Station reliability (v4, optional; ported unchanged from
notebooks/nudging_analysis_v10.ipynb)
---------------------------------------------------------------------------
Before spreading residuals, each station's influence radius may be scaled
by a leave-one-out spatial-consistency check (_compute_reliability): each
station's residual is predicted from every OTHER station (self-distance
forced to +inf); the discrepancy e = actual − predicted is turned into a
robust z-score u = (e − median(e)) / (1.4826 * MAD(e)) — median/MAD so a
few bad stations can't skew the reference point — and then into a Tukey
biweight reliability = clip(1 − (u/number_of_std)^2, 0, None)^2 in [0, 1].
A station with no neighbour within max_dist gets e=NaN, is excluded from
median(e)/MAD(e) (np.median is not NaN-safe), and is left at reliability=1.
Each station's max_dist is then scaled to
min_dist + (max_dist − min_dist) * reliability, min_dist =
reliability_min_dist_frac * max_dist, so influence never drops to zero.

The leave-one-out search and the domain buffer always use the GLOBAL
max_dist (reliability must be computed before any per-station radius can
be derived from it); only the final taper is reliability-dependent (v4.1).
use_reliability_check=False reproduces the pre-v4 (v3) behaviour exactly.
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
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

LOG = logging.getLogger(__name__)

# Variables that are fetched for reference but must never be nudged.
_NO_NUDGE = frozenset({"TOT_PREC"})

# GRIB shortName (COSMO/ICON) -> (station Parquet column, offset added to obs before residuals).
PARAM_MAP = {
    "T_2M":    ("2t",   0.0),   # 2 m temperature         [K]
    "TD_2M":   ("2d",   0.0),   # 2 m dewpoint            [K]
    "U_10M":   ("10u",  0.0),   # 10 m U wind component   [m/s]
    "V_10M":   ("10v",  0.0),   # 10 m V wind component   [m/s]
    "PMSL":    ("msl",  0.0),   # mean sea-level pressure  [Pa]
    "PS":      ("sp",   0.0),   # station-level (unreduced) surface pressure [Pa]
    "TOT_PREC":("tp",   0.0),   # hourly precipitation     [kg m-2]
    "VMAX_10M":("vmax", 0.0),   # 10 m wind gust           [m/s]
}

_DEFAULT_TOPO_VARS = [
    "TPI_500M", "TPI_4000M_SMTH", "SN_DERIV_2000M", "WE_DERIV_2000M", "ICON_OROG"
]
_DEFAULT_TOPO_FILE = (
    "/scratch/mch/llanzila/sruc/aux_files/topo_descriptors_icon_R19B08.nc"
)
_DEFAULT_DEM_BARRIER_FILE = (
    "/store_new/mch/msclim/appclim/data/grids/topodem/v2/topo/radar_100/topo_DEM_1000M.nc"
)
# Precomputed d_eff cache — see NudgeTowardObservation's d_eff_file parameter.
_DEFAULT_D_EFF_FILE = "/scratch/mch/llanzila/sruc/aux_files/d_eff_cache_v11.nc"
# ICON's own native orography (extpar); barrier_distances itself keeps using the
# finer external DEM (dem_barrier_file) for the barrier term.
_DEFAULT_ICON_OROG_FILE = (
    "/scratch/mch/icontest/testing-input-data/c2sm/icon-1/"
    "external_parameter_icon_grid_0001_R19B08_mch.nc"
)

_DEFAULT_TEMPERATURE_LAPSE_RATE = 0.0065  # K/m
_DEFAULT_TEMPERATURE_LAPSE_RATE_VARS = frozenset({"T_2M"})
_DEFAULT_PRESSURE_LAPSE_RATE = 11.5  # Pa/m
_DEFAULT_PRESSURE_LAPSE_RATE_VARS = frozenset({"PS"})

# Cosmetic only — never affects the nudging correction itself.
_DIAG_COLORBAR_PERCENTILE = 90


def _cast_astype(ds: xr.Dataset, dtypes: dict) -> xr.Dataset:
    """Cast each variable in ds to the dtype given in dtypes. Verbatim from
    data4web_pipelines/utils.py::cast_astype."""
    dtypes_unique = list(set(dtypes.values()))
    if len(dtypes_unique) == 1:
        return ds.astype(dtypes_unique[0])
    ds_out = ds.copy()
    for var, da in ds_out.items():
        if var in dtypes:
            ds_out[var] = da.astype(dtypes[var])
    return ds_out


def _normalize(ds: xr.Dataset) -> xr.Dataset:
    """Scale each variable in ds to [0, 1] using its own min/max. Verbatim from
    data4web_pipelines/utils.py::normalize."""
    dtypes = {name: da.dtype for name, da in ds.items()}
    min_val = _cast_astype(ds.min(), dtypes)
    max_val = _cast_astype(ds.max(), dtypes)
    return (ds - min_val) / (max_val - min_val)


def ned_interp(
    sta_res: xr.Dataset,
    ned_sta_poi: xr.DataArray,
    sta_topo: Optional[xr.Dataset] = None,
    poi_topo: Optional[xr.Dataset] = None,
    max_dist: Optional[Union[float, xr.DataArray]] = None,
    weight_power: float = 1,
    min_topo_w: float = 0.2,
    lim_effective: float = 0,
    taper: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Spread station residuals to POIs via IDW with optional topographic similarity.

    Adapted from data4web_pipelines/utils.py; the "ned" prefix (``ned_interp``,
    ``ned_sta_poi``, ``ned_ss``) is inherited naming from that module (expansion
    unknown), kept verbatim for continuity with the notebooks this was ported from.

    With topo descriptors: stations beyond max_dist are masked out (NaN distance);
    sta_topo/poi_topo are each separately scaled to [0, 1] using their own min/max
    (data4web's reference behaviour, kept for exact equivalence); descriptor
    importance is |Pearson corr(residuals, descriptor)| across stations, normalised
    to sum to 1; per-(POI, station) topo similarity is the importance-weighted
    average of (1 − |sta_val − poi_val|), floored at min_topo_w; the result multiplies
    the IDW weight. Weights are then normalised to sum to 1 per POI (lim_effective > 0
    adds a virtual zero-residual station to the denominator, shrinking corrections in
    data-sparse regions), and finally `taper` is applied — see the v4.1 note below.
    Without topo descriptors (sta_topo/poi_topo None), falls back to pure IDW.

    v4.1: `taper`, if given, is a per-(poi, sta) DataArray in [0, 1] multiplied into
    the weights AFTER normalisation, not merged into the raw weights beforehand —
    if a POI has only one contributing station, its normalised weight is always
    exactly 1 regardless of the raw weight, so a pre-normalisation taper would be
    exactly cancelled out by the division and have no effect in that (common, e.g.
    an isolated low-reliability station) case.
    """
    # NaN distance -> NaN IDW weight -> effectively excluded after normalisation.
    if max_dist is not None:
        ned_sta_poi = ned_sta_poi.where(ned_sta_poi < max_dist)

    if sta_topo is None or poi_topo is None:
        w_ned = 1 / np.power(ned_sta_poi, weight_power)
    else:
        sta_topo = _normalize(sta_topo)
        poi_topo = _normalize(poi_topo)

        delta_topo = 1 - abs(sta_topo - poi_topo)

        w_topo = (
            abs(
                xr.corr(
                    sta_res.to_array("data_var"),
                    sta_topo.to_array("topo"),
                    dim="sta",
                ).astype(np.float32)
            )
            .transpose("topo", ...)
            .to_dataset("data_var")
        )
        w_topo /= w_topo.sum("topo")

        for _var in w_topo.data_vars:
            if bool(w_topo[_var].isnull().all()):
                LOG.warning(
                    "ned_interp: descriptor importance is entirely NaN for "
                    "'%s' (likely zero-variance topo descriptors across too "
                    "few stations) — topographic similarity weighting "
                    "contributes nothing for this variable this call.",
                    _var,
                )

        w_topo = (w_topo * delta_topo.to_array("topo")).sum("topo")

        w_ned = w_topo.clip(min=min_topo_w) * (1 / np.power(ned_sta_poi, weight_power))

    w_ned /= w_ned.sum("sta") + lim_effective

    if taper is not None:
        w_ned = w_ned * taper

    # min_count=1: a POI with every station masked returns NaN, not 0; the caller
    # nan_to_num's this to a zero correction for those POIs.
    return (w_ned * sta_res).sum("sta", min_count=1)


def barrier_distances(
    poi_lon: np.ndarray,
    poi_lat: np.ndarray,
    sta_lon: np.ndarray,
    sta_lat: np.ndarray,
    d_euc: np.ndarray,
    max_dist: float,
    sta_elev: np.ndarray,
    dem_rgi: RegularGridInterpolator,
    wgs84_to_lv95: Transformer,
    n_samples: int = 50,
    elev_scale: float = 50,  # fit offline; see generate_d_eff_cache.py, not derived analytically
    elev_diff_scale: float = 100,  # fit offline; see generate_d_eff_cache.py, not derived analytically
    n_barrier_width_samples: int = 3,
    barrier_width: float = 1500.0,
) -> np.ndarray:
    """Replace Euclidean distances with elevation-aware effective distances.

    For each (POI, station) pair with d_euc < max_dist:
        d_eff = sqrt(d_euc² + (barrier / elev_scale)² + (elev_diff / elev_diff_scale)²)

    d_euc/max_dist are expected in km; elev_scale/elev_diff_scale in m/km. The
    function is unit-agnostic (output unit matches input), but
    NudgeTowardObservation always calls this with km.

    Barrier term: at each of n_samples interior points along the straight-line
    path (endpoints excluded — at t=0/1 the perpendicular sample grid would be
    centred on the POI/station itself, sampling terrain beside it rather than
    between the two points), n_barrier_width_samples DEM points span a
    ±barrier_width corridor perpendicular to the path, combined via a
    Gaussian-weighted mean (sigma = barrier_width / 2, so off-axis terrain is
    down-weighted and the direct path stays dominant). The 95th percentile of
    those means along the path (robust to DEM spikes) gives the effective ridge
    height above the higher endpoint:
        barrier = max(0, percentile_95(gauss_mean_cross) − max(elev_poi, elev_sta))

    elev_diff term: |elev_poi − elev_sta| penalises pairs at very different
    altitudes even with no intervening ridge (different vertical atmospheric
    regimes, e.g. valley station vs. high-altitude POI). The two terms are
    added in quadrature, so a large ridge and a large elevation gap compound.

    sta_elev should come from DWH station metadata (more accurate than DEM
    interpolation at station locations); elev_poi is read from the 1 km DEM.
    Not called by ``NudgeTowardObservation`` at runtime — invoked offline by
    ``generate_d_eff_cache.py`` to precompute the ``d_eff_file`` cache.
    """
    close_mask = d_euc < max_dist
    pi_idx, si_idx = np.where(close_mask)

    if len(pi_idx) == 0:
        return d_euc

    u_poi, inv_poi = np.unique(pi_idx, return_inverse=True)
    u_sta, inv_sta = np.unique(si_idx, return_inverse=True)
    
    poi_x_u, poi_y_u = wgs84_to_lv95.transform(poi_lon[u_poi], poi_lat[u_poi])
    sta_x_u, sta_y_u = wgs84_to_lv95.transform(sta_lon[u_sta], sta_lat[u_sta])

    elev_poi = dem_rgi(np.c_[poi_y_u, poi_x_u])[inv_poi]
    elev_sta = sta_elev[si_idx]

    ref_elev = np.maximum(elev_poi, elev_sta)

    poi_xp = poi_x_u[inv_poi]
    poi_yp = poi_y_u[inv_poi]
    sta_xp = sta_x_u[inv_sta]
    sta_yp = sta_y_u[inv_sta]

    t = np.linspace(0, 1, n_samples + 2)[1:-1]
    x_path = poi_xp[None, :] + t[:, None] * (sta_xp - poi_xp)[None, :]  # (n_samples, n_close)
    y_path = poi_yp[None, :] + t[:, None] * (sta_yp - poi_yp)[None, :]

    # Unit vector 90° to the path: rotate (dx, dy) -> (-dy, dx), then normalise.
    dx = sta_xp - poi_xp
    dy = sta_yp - poi_yp
    path_len = np.sqrt(dx ** 2 + dy ** 2)
    safe_len = np.where(path_len > 0, path_len, 1.0)  # avoid /0 for co-located pairs
    perp_x = -dy / safe_len
    perp_y =  dx / safe_len

    perp_offsets = np.linspace(-barrier_width, barrier_width, n_barrier_width_samples)

    # sigma=0 (barrier_width=0, single centre sample) -> uniform weight of 1.
    sigma = barrier_width / 2.0
    if sigma > 0:
        gauss_w = np.exp(-0.5 * (perp_offsets / sigma) ** 2)
    else:
        gauss_w = np.ones(n_barrier_width_samples)
    gauss_w /= gauss_w.sum()

    # (n_samples, n_perp, n_close): LV95 position of each along-path/corridor sample point.
    x_slab = x_path[:, None, :] + perp_offsets[None, :, None] * perp_x[None, None, :]
    y_slab = y_path[:, None, :] + perp_offsets[None, :, None] * perp_y[None, None, :]

    # RGI expects (northing, easting).
    n_perp  = n_barrier_width_samples
    n_close = len(pi_idx)
    elev_slab = dem_rgi(
        np.c_[y_slab.ravel(), x_slab.ravel()]
    ).reshape(n_samples, n_perp, n_close)

    elev_mean_cross = (elev_slab * gauss_w[None, :, None]).sum(axis=1)

    barrier = np.maximum(
        0.0, np.percentile(elev_mean_cross, 95, axis=0) - ref_elev
    ).astype(np.float32)

    elev_diff = np.abs(elev_poi - elev_sta).astype(np.float32)

    d_eff = d_euc.copy()
    d_eff[pi_idx, si_idx] = np.sqrt(
        d_euc[pi_idx, si_idx] ** 2
        + (barrier    / elev_scale)      ** 2
        + (elev_diff  / elev_diff_scale) ** 2
    ).astype(np.float32)

    n_blocked = int((d_eff[pi_idx, si_idx] >= max_dist).sum())
    LOG.debug(
        "barrier_distances: %d close pairs → %d newly blocked by barrier+elev_diff (%.1f%%)",
        len(pi_idx), n_blocked, 100.0 * n_blocked / max(len(pi_idx), 1),
    )
    return d_eff


class NudgeTowardObservation(Filter):
    """Nudge the forecast initial condition toward surface station observations.

    Implements v3 Interpolation of Residuals using ned_interp with topographic
    similarity weighting and barrier-aware effective distances (DEM path sampling).

    Reads pre-fetched station observations from a Parquet file written by
    RetrieveObservation (must include ``elevation``, as produced by the current
    version of that filter). Nudging is applied once, to the initial-condition
    time step; later calls pass all fields through unchanged.

    Parameters
    ----------
    obs_path : str
        Cleaned station observations Parquet file. Required columns:
        ``latitude``, ``longitude``, ``elevation``, and one column per nudged
        variable (e.g. ``2t``, ``10u``), all in SI units.
    icon_grid_dir : str
        Directory containing ``icon_grid_0001_R19B08_mch.nc``.
    topo_file : str
        Topographic descriptor NetCDF on the ICON R19B08 grid. Must contain
        ``lon``/``lat`` and all of *topo_vars* except ``ICON_OROG``, which is
        injected separately from *icon_orog_file*.
    dem_barrier_file : str
        1 km DEM NetCDF (variable ``DEM_1000M``, coords ``x``/``y`` in LV95
        metres). No longer used to compute d_eff at run time (see
        *d_eff_file*) — only as a fallback to sample station elevation when
        missing from the observations Parquet.
    d_eff_file : str
        Precomputed NetCDF of barrier- and elevation-aware effective
        distances (d_eff) for the full station catalog: ``d_eff_poi`` (dims
        ``poi``, ``sta``) and ``d_eff_sta`` (dims ``sta_i``, ``sta``,
        self-distances +inf, for the reliability check). Built offline via
        ``barrier_distances()`` and reused as long as the station catalog,
        DEM, ICON grid, and barrier hyperparameters are unchanged. This
        filter never calls ``barrier_distances()`` itself — a POI/station not
        covered by the cache raises ``ValueError``; rebuild the cache.
    icon_orog_file : str
        ICON extpar NetCDF (variable ``topography_c``), the model's own
        native orography. Used only as the ``ICON_OROG`` topo descriptor —
        distinct from *dem_barrier_file*, which drives the barrier distance.
    weight_power : float
        IDW distance-decay exponent; higher concentrates weight on the
        nearest station (notebook: ``WEIGHT_POWER = 4``).
    max_dist : float
        Default station influence radius, in **meters** (converted to km
        once here in __init__; every other use of this radius in the file —
        barrier-distance cutoff, domain-buffer size, and — when
        *use_reliability_check* is ``False`` — the shared taper radius — is
        km, unaffected by this input-side unit), overridable per-variable via
        *variable_overrides*. When *use_reliability_check* is ``True``, this
        is the upper bound each station's reliability-scaled radius shrinks
        from. Default: 50000 (50 km).
    variable_overrides : dict, optional
        Per-variable override of *d_eff_file* and/or *max_dist* (also in
        meters, converted the same way), e.g.
        ``{"U_10M": {"d_eff_file": "...", "max_dist": 20000.0}}``. Keys must be
        GRIB shortNames in the active nudge set; near-surface wind bias has a
        shorter decorrelation length than other variables, so wind typically
        uses a smaller-radius cache (notebooks/d_eff_generator.ipynb).
        Defaults to ``None`` (every variable uses the top-level default).
    min_topo_w : float
        Minimum topographic similarity weight floor so nearby stations always
        contribute (notebook: ``MIN_TOPO_W = 0.2``).
    lim_effective : float
        Virtual zero-residual station weight added to the normalisation
        denominator; 0 = pure IDW (notebook: ``LIM_EFFECTIVE = 0.0``).
    use_reliability_check : bool
        If ``True``, scale each station's influence radius by a leave-one-out
        spatial-consistency check (see module docstring, "Station
        reliability"). Defaults to ``False`` (pre-v4/v3 behaviour: a single
        global *max_dist* for every station; notebook default is ``True``).
    number_of_std : float
        Tukey biweight rejection threshold in robust (median/MAD) standard
        deviations — a station at or beyond this many robust sigmas gets
        reliability=0. Only used when *use_reliability_check* is ``True``
        (notebook: ``RELIABILITY_REJECT_C``, default 4.0).
    reliability_min_dist_frac : float
        Minimum fraction of *max_dist* every station keeps even at
        reliability=0 (never shrinks to zero). Must be in [0, 1]. Only used
        when *use_reliability_check* is ``True`` (notebook:
        ``RELIABILITY_MIN_DIST_FRAC``, default 0.1).
    reliability_eps : float
        Numerical floor on the robust (MAD-based) scale estimate, avoiding
        division by zero (notebook: ``RELIABILITY_EPS``, default 1e-6). Only
        used when *use_reliability_check* is ``True``.
    temperature_lapse_rate : float
        Standard-atmosphere lapse rate [K/m] reducing station observations to
        the model's elevation before differencing:
        ``obs_corrected = obs - temperature_lapse_rate * (elev_model_at_cell - elev_sta)``.
        Applied only to *temperature_lapse_rate_vars* (default: 0.0065, i.e. 6.5 K/km).
    temperature_lapse_rate_vars : list of str, optional
        GRIB shortNames *temperature_lapse_rate* applies to. Defaults to
        ``["T_2M"]``: dewpoint doesn't follow a fixed lapse rate, pressure/wind
        don't apply here (see *pressure_lapse_rate_vars*).
    pressure_lapse_rate : float
        Same idea as *temperature_lapse_rate* but for ``PS`` (station-level, unreduced
        pressure); a separate parameter since pressure's near-surface
        gradient (~10-12 Pa/m) is ~3 orders of magnitude steeper than
        temperature's (default: 11.5 Pa/m — see
        *_DEFAULT_PRESSURE_LAPSE_RATE*). Applied only to
        *pressure_lapse_rate_vars*.
    pressure_lapse_rate_vars : list of str, optional
        GRIB shortNames *pressure_lapse_rate* applies to. Defaults to
        ``["PS"]``; ``PMSL`` is excluded as it's already sea-level-reduced.
    topo_vars : list of str, optional
        Topographic descriptor names, from *topo_file* plus the injected
        ``ICON_OROG``. Defaults to ``["TPI_500M", "TPI_4000M_SMTH",
        "SN_DERIV_2000M", "WE_DERIV_2000M", "ICON_OROG"]``. Ignored when
        *use_topo_descriptors* is ``False``.
    use_topo_descriptors : bool
        If ``True`` (default), weight stations by topographic similarity to
        each POI on top of the barrier-aware IDW distance; if ``False``, fall
        back to pure IDW for both POI spreading and the reliability check.
        *topo_file*/*icon_orog_file* are still read either way, since
        ``ICON_OROG`` also feeds the lapse-rate correction.
    nudge_variables : list of str, optional
        GRIB shortNames to nudge (subset of PARAM_MAP keys). Defaults to all
        non-precipitation variables.
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
    enable_plotting : bool
        Master on/off switch for the reliability diagnostic plot, independent
        of *plot_dir* (lets a deployment keep *plot_dir* set while skipping
        the matplotlib/cartopy rendering cost). ``True`` by default; when
        ``False``, *plot_dir* is not even created.
    plot_dir : str, optional
        If given (and *enable_plotting*), save a 6-panel reliability
        diagnostic PNG per (variable, ref_time) — see
        ``_plot_reliability_diagnostic``:
        ``{plot_dir}/reliability_diag_{shortname}_{ref_time:%Y%m%d%H%M}.png``.
        Only effective when *use_reliability_check* is ``True``. Defaults to
        ``None``. Plotting failures are logged and swallowed.
    plot_extent : list of float, optional
        ``[lon_min, lon_max, lat_min, lat_max]`` map extent for the
        diagnostic plot. Defaults to ``[5.8, 10.8, 45.7, 47.9]``
        (Switzerland).
    """

    def __init__(
        self,
        obs_path: str,
        icon_grid_dir: str = "/scratch/mch/llanzila/sruc/aux_files",
        topo_file: str = _DEFAULT_TOPO_FILE,
        dem_barrier_file: str = _DEFAULT_DEM_BARRIER_FILE,
        d_eff_file: str = _DEFAULT_D_EFF_FILE,
        icon_orog_file: str = _DEFAULT_ICON_OROG_FILE,
        weight_power: float = 2.0,
        max_dist: float = 50000.0,  # m
        variable_overrides: Optional[dict] = None,
        min_topo_w: float = 0.2,
        lim_effective: float = 0.0,
        use_reliability_check: bool = False,
        number_of_std: float = 5.0,
        reliability_min_dist_frac: float = 0.05,
        reliability_eps: float = 1e-6,
        enable_plotting: bool = True,
        plot_dir: Optional[str] = None,
        plot_extent: Optional[list] = None,
        temperature_lapse_rate: float = _DEFAULT_TEMPERATURE_LAPSE_RATE,
        temperature_lapse_rate_vars: Optional[list] = None,
        pressure_lapse_rate: float = _DEFAULT_PRESSURE_LAPSE_RATE,
        pressure_lapse_rate_vars: Optional[list] = None,
        topo_vars: Optional[list] = None,
        use_topo_descriptors: bool = True,
        nudge_variables: Optional[list] = None,
        run_mode: str = "depl",
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
        if not (0.0 <= reliability_min_dist_frac <= 1.0):
            raise ValueError(
                f"reliability_min_dist_frac must be in [0, 1], got {reliability_min_dist_frac!r}"
            )
        if number_of_std <= 0:
            raise ValueError(f"number_of_std must be > 0, got {number_of_std!r}")

        if max_dist <= 0:
            raise ValueError(f"max_dist must be > 0, got {max_dist!r}")
        if weight_power <= 0:
            raise ValueError(f"weight_power must be > 0, got {weight_power!r}")
        if min_topo_w < 0:
            raise ValueError(f"min_topo_w must be >= 0, got {min_topo_w!r}")
        if lim_effective < 0:
            raise ValueError(f"lim_effective must be >= 0, got {lim_effective!r}")

        self.obs_path = Path(obs_path)
        self.icon_grid_dir = Path(icon_grid_dir)
        self.topo_file = Path(topo_file)
        self.dem_barrier_file = Path(dem_barrier_file)
        self.d_eff_file = Path(d_eff_file)
        self.icon_orog_file = Path(icon_orog_file)
        self.weight_power = weight_power
        self.max_dist = max_dist / 1000.0  # max_dist is given in meters (see docstring); converted to km once here
        self.min_topo_w = min_topo_w
        self.lim_effective = lim_effective
        self.use_reliability_check = use_reliability_check
        self.number_of_std = number_of_std
        self.reliability_min_dist_frac = reliability_min_dist_frac
        self.reliability_eps = reliability_eps
        self.enable_plotting = enable_plotting
        self.plot_dir = Path(plot_dir) if plot_dir is not None else None
        self.plot_extent = list(plot_extent) if plot_extent is not None else [5.8, 10.8, 45.7, 47.9]
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
        self.topo_vars = list(topo_vars) if topo_vars is not None else _DEFAULT_TOPO_VARS
        self.use_topo_descriptors = use_topo_descriptors
        self.run_mode = run_mode
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed
        self.exclude_stations = list(exclude_stations) if exclude_stations is not None else None
        self._nudging_done = False
        self._reliability_diag = {}

        if self.plot_dir is not None and self.enable_plotting:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            if not self.use_reliability_check:
                LOG.warning(
                    "plot_dir=%s is set but use_reliability_check=False — there is no "
                    "reliability to plot, so no diagnostic plots will be produced.",
                    self.plot_dir,
                )
        elif self.plot_dir is not None and not self.enable_plotting:
            LOG.info(
                "plot_dir=%s is set but enable_plotting=False — no diagnostic plots "
                "will be produced (the directory is not created).",
                self.plot_dir,
            )

        if nudge_variables is not None:
            unknown = set(nudge_variables) - PARAM_MAP.keys()
            if unknown:
                raise ValueError(
                    f"Unknown nudge variables: {unknown}. Valid: {list(PARAM_MAP)}"
                )
            self.param_map = {v: PARAM_MAP[v] for v in nudge_variables}
        else:
            self.param_map = dict(PARAM_MAP)
        self.param_map = {v: w for v, w in self.param_map.items() if v not in _NO_NUDGE}

        # ── Per-variable d_eff_file/max_dist (variable_overrides) ──────────
        self.variable_overrides = dict(variable_overrides) if variable_overrides is not None else {}
        unknown_override_vars = set(self.variable_overrides) - set(self.param_map)
        if unknown_override_vars:
            raise ValueError(
                f"variable_overrides references variable(s) not in the active "
                f"nudge set: {sorted(unknown_override_vars)}. Active variables: "
                f"{list(self.param_map)}"
            )
        for _var, _override in self.variable_overrides.items():
            _unknown_keys = set(_override) - {"d_eff_file", "max_dist"}
            if _unknown_keys:
                raise ValueError(
                    f"variable_overrides[{_var!r}] has unknown key(s) "
                    f"{sorted(_unknown_keys)}; only 'd_eff_file' and 'max_dist' "
                    "are supported."
                )

        # variable_overrides' max_dist 
        self._max_dist_by_var = {
            var: (
                self.variable_overrides[var]["max_dist"] / 1000.0
                if "max_dist" in self.variable_overrides.get(var, {})
                else self.max_dist
            )
            for var in self.param_map
        }
        for _var, _md in self._max_dist_by_var.items():
            if _md <= 0:
                raise ValueError(
                    f"variable_overrides[{_var!r}]['max_dist'] must be > 0, got {_md!r}"
                )
        self._d_eff_file_by_var = {
            var: Path(self.variable_overrides.get(var, {}).get("d_eff_file", self.d_eff_file))
            for var in self.param_map
        }

        # Load heavy static data once at construction time to avoid repeated I/O in forward().
        self._load_icon_grid()
        self._load_topo()
        self._load_dem()
        # Load each distinct d_eff cache file once (several variables may share one).
        self._d_eff_caches = {
            path: self._load_d_eff_cache(path) for path in set(self._d_eff_file_by_var.values())
        }

        LOG.info(
            "NudgeTowardObservation v5 initialised: variables=%s, max_dist=%s km, "
            "weight_power=%.1f, d_eff_file=%s, use_topo_descriptors=%s, "
            "temperature_lapse_rate=%.5f K/m (vars=%s), pressure_lapse_rate=%.2f Pa/m (vars=%s), "
            "use_reliability_check=%s, number_of_std=%.2f, "
            "reliability_min_dist_frac=%.3f, min radius=%s km",
            list(self.param_map.keys()),
            {v: self._max_dist_by_var[v] for v in self.param_map},
            self.weight_power,
            {v: str(self._d_eff_file_by_var[v]) for v in self.param_map},
            self.use_topo_descriptors,
            self.temperature_lapse_rate,
            sorted(self.temperature_lapse_rate_vars),
            self.pressure_lapse_rate,
            sorted(self.pressure_lapse_rate_vars),
            self.use_reliability_check,
            self.number_of_std,
            self.reliability_min_dist_frac,
            {v: round(self.reliability_min_dist_frac * self._max_dist_by_var[v], 2) for v in self.param_map},
        )
        super().__init__()

    def _elevation_rate(self, shortname: str) -> Optional[float]:
        """Elevation-reduction rate for *shortname*'s observations (see
        *temperature_lapse_rate_vars*/*pressure_lapse_rate_vars*), or ``None`` if this
        variable gets no elevation correction. *temperature_lapse_rate_vars* is checked
        first if a config were to make the two sets overlap."""
        if shortname in self.temperature_lapse_rate_vars:
            return self.temperature_lapse_rate
        if shortname in self.pressure_lapse_rate_vars:
            return self.pressure_lapse_rate
        return None

    # ── Static data loaders ───────────────────────────────────────────────────

    def _load_icon_grid(self) -> None:
        ds = xr.open_dataset(self.icon_grid_dir / "icon_grid_0001_R19B08_mch.nc")
        # clat/clon are stored in radians in the ICON grid file.
        self._lat_icon = np.degrees(ds["clat"].values).ravel()
        self._lon_icon = np.degrees(ds["clon"].values).ravel()
        ds.close()
        LOG.info(
            "ICON grid loaded: %d cells from %s",
            len(self._lat_icon),
            self.icon_grid_dir / "icon_grid_0001_R19B08_mch.nc",
        )

    def _load_topo(self) -> None:
        self._ds_topo = xr.open_dataset(self.topo_file)
        # This only catches a cell-COUNT mismatch; a same-count-different-ORDER
        # mismatch would silently misalign descriptors (_nudge_field's dom_idx
        # later indexes this dataset positionally) — partial safety net only.
        if self._ds_topo.sizes.get("cell") != len(self._lat_icon):
            raise ValueError(
                f"topo_file {self.topo_file} has "
                f"{self._ds_topo.sizes.get('cell')} 'cell' entries but the "
                f"ICON grid ({self.icon_grid_dir}) has {len(self._lat_icon)} "
                "cells — topo_file must be on the exact same grid as "
                "icon_grid_dir."
            )
        self._load_icon_orog()
        if not self.use_topo_descriptors:
            LOG.info(
                "use_topo_descriptors=False: topo descriptors from %s will not be "
                "used for similarity weighting (ICON_OROG from %s is still loaded "
                "for the lapse-rate correction).",
                self.topo_file, self.icon_orog_file,
            )
            return
        missing = [v for v in self.topo_vars if v not in self._ds_topo]
        if missing:
            raise ValueError(
                f"Topo variables {missing} not found in {self.topo_file} "
                f"(plus the injected ICON_OROG). Available: {list(self._ds_topo.data_vars)}"
            )
        LOG.info("Topo descriptors loaded from %s: %s", self.topo_file, self.topo_vars)

    def _load_icon_orog(self) -> None:
        """Inject ICON's native orography as the 'ICON_OROG' topo descriptor.

        'cell' ordering matches topo_file/icon_grid_dir exactly (same source grid
        file), so the array is attached positionally, without reindexing.
        """
        ds_orog = xr.open_dataset(self.icon_orog_file)
        self._ds_topo["ICON_OROG"] = ("cell", ds_orog["topography_c"].values.astype(np.float32))
        ds_orog.close()
        LOG.info("ICON native orography loaded from %s: ICON_OROG", self.icon_orog_file)

    def _load_dem(self) -> None:
        dem_ds = xr.open_dataset(self.dem_barrier_file)
        dem_shape = dem_ds["DEM_1000M"].shape
        # NaN (ocean/no-data) -> 0 m so out-of-domain path segments don't produce NaN barriers.
        dem_z = np.where(
            np.isnan(dem_ds["DEM_1000M"].values), 0.0, dem_ds["DEM_1000M"].values
        )
        # RGI axes: first=y (northing/rows), second=x (easting/cols); query as (y, x).
        self._dem_rgi = RegularGridInterpolator(
            (dem_ds["y"].values, dem_ds["x"].values),
            dem_z,
            method="linear",
            bounds_error=False,
            fill_value=0.0,  # extrapolate as 0 m for points outside the DEM extent
        )
        dem_ds.close()
        # always_xy=True: transform(lon, lat) -> (easting, northing).
        self._wgs84_to_lv95 = Transformer.from_crs(
            "EPSG:4326", "EPSG:2056", always_xy=True
        )
        LOG.info(
            "DEM loaded from %s: shape=%s",
            self.dem_barrier_file,
            dem_shape,
        )

        # ── Project the ICON grid and topo-descriptor grid to LV95 km, once ────
        # Computed once here and reused by every _nudge_field() call instead of
        # re-projecting each time.
        grid_x, grid_y = self._wgs84_to_lv95.transform(self._lon_icon, self._lat_icon)
        self._grid_xy_km = np.c_[grid_x, grid_y] / 1000.0

        topo_x, topo_y = self._wgs84_to_lv95.transform(
            self._ds_topo["lon"].values, self._ds_topo["lat"].values
        )
        self._topo_xy_km = np.c_[topo_x, topo_y] / 1000.0

        # Built once and reused by every _nudge_field() call (these never change
        # after __init__), rather than rebuilding per call.
        self._grid_tree = cKDTree(self._grid_xy_km)
        self._topo_tree = cKDTree(self._topo_xy_km)

        LOG.info(
            "ICON/topo grids projected to LV95: x=[%.1f, %.1f] km, y=[%.1f, %.1f] km",
            self._grid_xy_km[:, 0].min(), self._grid_xy_km[:, 0].max(),
            self._grid_xy_km[:, 1].min(), self._grid_xy_km[:, 1].max(),
        )

    def _load_d_eff_cache(self, path: Path) -> dict:
        """Load one precomputed d_eff cache file for the full station catalog.

        Built offline (notebooks/nudging_analysis_v11.ipynb, "Precomputed d_eff
        cache"); every call's own POI/station subset is sliced from it via
        _get_d_eff_poi()/_get_d_eff_sta(). Called once per distinct path in
        *_d_eff_file_by_var* — variables sharing a cache file (e.g. T_2M/TD_2M)
        only load it once.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"d_eff cache not found: {path}. This filter no longer "
                "computes barrier-aware effective distances itself — build "
                "the cache offline first (see notebooks/nudging_analysis_v11"
                ".ipynb, 'Precomputed d_eff cache') and point d_eff_file (or "
                "variable_overrides) at it."
            )
        ds = xr.open_dataset(path)
        d_eff_poi_full = ds["d_eff_poi"].load()
        d_eff_sta_full = ds["d_eff_sta"].load()
        ds.close()
        LOG.info(
            "d_eff loaded from %s: POI x station %s, station x station %s",
            path, d_eff_poi_full.shape, d_eff_sta_full.shape,
        )
        return {
            "d_eff_poi_full": d_eff_poi_full,
            "d_eff_sta_full": d_eff_sta_full,
            # Precomputed so _get_d_eff_poi can use pandas' get_indexer instead
            # of xarray's label-based .sel on every call.
            "poi_index": pd.Index(d_eff_poi_full["poi"].values),
            "sta_index": pd.Index(d_eff_poi_full["sta"].values),
            "poi_set": set(d_eff_poi_full["poi"].values.tolist()),
            "sta_set": set(d_eff_poi_full["sta"].values.tolist()),
        }

    def _get_d_eff_poi(self, shortname: str, dom_idx: np.ndarray, sta_ids: list) -> xr.DataArray:
        """POI<->station d_eff for this call's domain/station subset, sliced
        from the precomputed cache configured for *shortname* (see
        *d_eff_file*/*variable_overrides* and _load_d_eff_cache)."""
        d_eff_file = self._d_eff_file_by_var[shortname]
        cache = self._d_eff_caches[d_eff_file]
        missing_poi = set(dom_idx) - cache["poi_set"]
        if missing_poi:
            raise ValueError(
                f"{len(missing_poi)} POI(s) not covered by the precomputed "
                f"d_eff cache ({d_eff_file}) used for '{shortname}' — the "
                "cache was built for a smaller domain, or the ICON grid "
                "changed since. Rebuild the cache."
            )
        missing_sta = set(sta_ids) - cache["sta_set"]
        if missing_sta:
            raise ValueError(
                f"Station(s) {sorted(missing_sta)} not covered by the "
                f"precomputed d_eff cache ({d_eff_file}) used for "
                f"'{shortname}' — the station catalog changed since the "
                "cache was built. Rebuild the cache."
            )
        # Positional lookup instead of xarray's label-based .sel; get_indexer
        # preserves the requested order, matching .sel's ordering exactly.
        poi_pos = cache["poi_index"].get_indexer(dom_idx)
        sta_pos = cache["sta_index"].get_indexer(sta_ids)
        values = cache["d_eff_poi_full"].values[np.ix_(poi_pos, sta_pos)]
        return xr.DataArray(
            values, dims=["poi", "sta"], coords={"poi": dom_idx, "sta": sta_ids}
        )

    def _get_d_eff_sta(self, shortname: str, sta_ids: list) -> xr.DataArray:
        """Station<->station d_eff (for _compute_reliability's leave-one-out
        check), sliced from the precomputed cache configured for
        *shortname*."""
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
        return (
            cache["d_eff_sta_full"]
            .sel(sta_i=list(sta_ids), sta=list(sta_ids))
            .rename({"sta_i": "poi"})
        )

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
        # Kept only for the optional diagnostic plot; never used in the correction.
        held_out_stations = all_stations.drop(index=stations.index)

        nudged = {}
        for field in data.sel(shortName=list(self.param_map.keys())):
            shortname = field.metadata("shortName")

            if field.datetime()["valid_time"] != ref_time:
                continue

            col, offset = self.param_map[shortname]
            if col not in stations.columns or stations[col].isna().all():
                LOG.warning("No observations for '%s', skipping", shortname)
                continue

            corrected = self._nudge_field(
                field, stations, shortname, col, offset, ref_time, held_out_stations
            )
            nudged[shortname] = new_field_from_numpy(
                corrected,
                template=field,
                validityDate=field.metadata("validityDate"),
                validityTime=field.metadata("validityTime"),
                dataDate=field.metadata("dataDate"),
                dataTime=field.metadata("dataTime"),
            )
            LOG.info(
                "Nudged '%s' using %d stations",
                shortname,
                int(stations[col].notna().sum()),
            )

        result = [
            nudged.get(f.metadata("shortName"), f)
            if f.datetime()["valid_time"] == ref_time
            else f
            for f in data
        ]
        self._nudging_done = True
        LOG.info("Nudging complete: %d/%d fields updated", len(nudged), len(result))
        return new_fieldlist_from_list(result)

    # ── Core algorithm ────────────────────────────────────────────────────────

    def _station_elevation(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        lon: np.ndarray,
        lat: np.ndarray,
        *,
        log: bool = False,
    ) -> np.ndarray:
        """Station elevation for the `mask`-selected rows of `df` (with
        `lon`/`lat` already extracted for that same subset, same order).

        Uses DWH metadata (`df`'s ``elevation`` column) when available — more
        accurate than DEM interpolation at station locations — falling back to
        DEM sampling for individual NaN entries, or every row when the column
        is absent (e.g. an older Parquet predating ``RetrieveObservation``
        adding it). `log=False` keeps the holdout/diagnostic call site silent.
        """
        if "elevation" in df.columns:
            elev = df.loc[mask, "elevation"].to_numpy(dtype=float)
            nan_mask = np.isnan(elev)
            if nan_mask.any():
                x, y = self._wgs84_to_lv95.transform(lon[nan_mask], lat[nan_mask])
                elev[nan_mask] = self._dem_rgi(np.c_[y, x])
                if log:
                    LOG.debug(
                        "Filled %d/%d station elevations from DEM (NaN in metadata)",
                        int(nan_mask.sum()), len(elev),
                    )
        else:
            if log:
                LOG.warning(
                    "Station Parquet missing 'elevation' column; sampling DEM at "
                    "station locations. Upgrade RetrieveObservation to include "
                    "elevation."
                )
            x, y = self._wgs84_to_lv95.transform(lon, lat)
            elev = self._dem_rgi(np.c_[y, x])
        return elev

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
        cell (`grid_index`) via *temperature_lapse_rate*/*pressure_lapse_rate* (see
        *_elevation_rate*), so the residual reflects model bias rather than the
        elevation mismatch between station and ICON's (smoothed) orography.
        Returns `obs` unchanged for variables with no configured rate.
        `log=False` keeps the holdout call site silent.
        """
        obs_lr = obs
        rate = self._elevation_rate(shortname)
        if rate is not None:
            elev_model = self._ds_topo["ICON_OROG"].values[grid_index]
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
        col: str,
        offset: float,
        ref_time=None,
        held_out_stations: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Apply v3 IoR nudging to a single field; return the corrected 1-D array.

        ``ref_time`` and ``held_out_stations`` are only used for the optional
        reliability diagnostic plot (see *plot_dir*) and play no role in the
        nudging computation itself.
        """
        # float32: matches sta_res/d_eff elsewhere in this pipeline and is far
        # finer than physically meaningful for these variables — drops unused
        # precision, not signal.
        B_flat = np.asarray(field.values, dtype=np.float32).ravel()

        max_dist = self._max_dist_by_var[shortname]

        # ── Valid stations ─────────────────────────────────────────────────
        valid   = stations[col].notna()
        st_lat  = stations.loc[valid, "latitude"].to_numpy()
        st_lon  = stations.loc[valid, "longitude"].to_numpy()
        st_obs  = stations.loc[valid, col].to_numpy() + offset
        sta_ids = stations.loc[valid].index.tolist()

        st_elev = self._station_elevation(stations, valid, st_lon, st_lat, log=True)

        # ── Coordinate projection ──────────────────────────────────────────
        grid_xy = self._grid_xy_km
        sta_x, sta_y = self._wgs84_to_lv95.transform(st_lon, st_lat)
        sta_xy = np.c_[sta_x, sta_y] / 1000.0

        # ── POI domain ─────────────────────────────────────────────────────
        # LV95 is an isotropic metric CRS, so a simple symmetric km buffer suffices;
        # this cuts the n_poi x n_sta distance matrix from ~1.1M to ~100k rows.
        sta_x_min = sta_xy[:, 0].min() - max_dist
        sta_x_max = sta_xy[:, 0].max() + max_dist
        sta_y_min = sta_xy[:, 1].min() - max_dist
        sta_y_max = sta_xy[:, 1].max() + max_dist
        dom_mask = (
            (grid_xy[:, 0] >= sta_x_min) & (grid_xy[:, 0] <= sta_x_max) &
            (grid_xy[:, 1] >= sta_y_min) & (grid_xy[:, 1] <= sta_y_max)
        )
        dom_idx = np.where(dom_mask)[0]
        poi_xy  = grid_xy[dom_idx]
        n_poi   = len(dom_idx)

        # ── Residuals at stations ──────────────────────────────────────────
        # self._grid_tree was built once in _load_dem(), reused across every call.
        _, gi   = self._grid_tree.query(sta_xy, k=1)

        st_obs_lr = self._reduce_obs_to_model_elevation(shortname, st_obs, gi, st_elev, log=True)

        r_at_st = B_flat[gi] - st_obs_lr  # positive when model > observation

        sta_res = xr.Dataset(
            {shortname: xr.DataArray(
                r_at_st.astype(np.float32), dims=["sta"], coords={"sta": sta_ids}
            )}
        )

        # ── Euclidean distance matrix ──────────────────────────────────────
        # float32 only for this broadcast (poi_xy/sta_xy stay float64, still needed
        # below for the topo cKDTree query) — halves transient size; precision is
        # ample for a continuous IDW/taper weight.
        d_euc_mat = np.sqrt(
            (
                (poi_xy[:, None, :].astype(np.float32) - sta_xy[None, :, :].astype(np.float32)) ** 2
            ).sum(axis=-1)
        ).astype(np.float32)

        # ── Barrier-aware distances ────────────────────────────────────────
        ned_sta_poi = self._get_d_eff_poi(shortname, dom_idx, sta_ids)

        # ── Topographic descriptors at POIs and stations ───────────────────
        if self.use_topo_descriptors:
            poi_topo = (
                self._ds_topo[self.topo_vars]
                .isel(cell=dom_idx)
                .rename({"cell": "poi"})
                .assign_coords({"poi": dom_idx})
            )
            # self._topo_tree, like self._grid_tree, was built once in _load_dem().
            _, topo_gi = self._topo_tree.query(sta_xy, k=1)
            sta_topo = (
                self._ds_topo[self.topo_vars]
                .isel(cell=topo_gi)
                .rename({"cell": "sta"})
                .assign_coords({"sta": sta_ids})
            )
        else:
            poi_topo = None
            sta_topo = None

        # ── Station reliability → per-station influence radius (v4) ────────
        if self.use_reliability_check:
            reliability = self._compute_reliability(
                shortname, sta_ids, st_lon, st_lat, r_at_st, sta_res, sta_topo, max_dist,
            )
            min_dist = self.reliability_min_dist_frac * max_dist
            station_max_dist = min_dist + (max_dist - min_dist) * reliability
        else:
            station_max_dist = max_dist

        # ── Per-station linear taper (v4.1) ─────────────────────────────────
        d_euc_da = xr.DataArray(
            d_euc_mat, dims=["poi", "sta"], coords={"poi": dom_idx, "sta": sta_ids}
        )
        pair_taper = (1.0 - (d_euc_da / station_max_dist).clip(min=0.0, max=1.0)).astype(np.float32)

        # ── ned_interp ─────────────────────────────────────────────────────
        # max_dist cuts on barrier-aware distance here, so a POI that is
        # geographically close but behind a ridge may get zero correction.
        result = ned_interp(
            sta_res, ned_sta_poi,
            sta_topo=sta_topo, poi_topo=poi_topo,
            max_dist=station_max_dist,
            weight_power=self.weight_power,
            min_topo_w=self.min_topo_w,
            lim_effective=self.lim_effective,
            taper=pair_taper,
        )

        # POIs with no station within their radius return NaN from ned_interp -> no correction.
        correction = np.nan_to_num(result[shortname].values, nan=0.0)

        corrected_flat = B_flat.copy()
        corrected_flat[dom_idx] -= correction

        if self.enable_plotting and self.plot_dir is not None and self.use_reliability_check:
            self._record_reliability_diagnostics(
                shortname, col, offset, stations, held_out_stations,
                B_flat, corrected_flat, st_lat, st_lon, gi, st_obs_lr,
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
            shortname, len(st_lat), n_poi, float(np.abs(correction).max()), reliability_note,
        )

        if self.enable_plotting and self.plot_dir is not None and self.use_reliability_check:
            self._plot_reliability_diagnostic(shortname, ref_time, B_flat, corrected_flat)

        return corrected_flat

    def _record_reliability_diagnostics(
        self,
        shortname: str,
        col: str,
        offset: float,
        stations: pd.DataFrame,
        held_out_stations: Optional[pd.DataFrame],
        B_flat: np.ndarray,
        corrected_flat: np.ndarray,
        st_lat: np.ndarray,
        st_lon: np.ndarray,
        gi: np.ndarray,
        st_obs_lr: np.ndarray,
    ) -> None:
        """Record pre/post-nudging station error (holdin + holdout) into
        ``self._reliability_diag[shortname]``, for the diagnostic plot's
        residual/RMSE panels only — never feeds back into the correction,
        which has already been returned by the time this runs. Holdin
        stations validate the fit where it influenced the correction; holdout
        stations (excluded by ``_apply_holdout``) give an independent check.
        Only called when *enable_plotting*, *plot_dir*, and
        *use_reliability_check* are all set.
        """
        err_lat = [st_lat]
        err_lon = [st_lon]
        err_val_pre = [B_flat[gi] - st_obs_lr]
        err_val_post = [corrected_flat[gi] - st_obs_lr]
        err_is_holdout = [np.zeros(len(st_lat), dtype=bool)]

        if held_out_stations is not None and len(held_out_stations):
            ho_valid = held_out_stations[col].notna() if col in held_out_stations.columns else pd.Series(dtype=bool)
            if ho_valid.any():
                ho_lat = held_out_stations.loc[ho_valid, "latitude"].to_numpy()
                ho_lon = held_out_stations.loc[ho_valid, "longitude"].to_numpy()
                ho_obs = held_out_stations.loc[ho_valid, col].to_numpy() + offset

                ho_elev = self._station_elevation(held_out_stations, ho_valid, ho_lon, ho_lat, log=False)

                ho_x, ho_y = self._wgs84_to_lv95.transform(ho_lon, ho_lat)
                ho_xy = np.c_[ho_x, ho_y] / 1000.0
                _, ho_gi = self._grid_tree.query(ho_xy, k=1)

                ho_obs_lr = self._reduce_obs_to_model_elevation(shortname, ho_obs, ho_gi, ho_elev, log=False)

                err_lat.append(ho_lat)
                err_lon.append(ho_lon)
                err_val_pre.append(B_flat[ho_gi] - ho_obs_lr)
                err_val_post.append(corrected_flat[ho_gi] - ho_obs_lr)
                err_is_holdout.append(np.ones(len(ho_lat), dtype=bool))

        # One sample per station here, so this is algebraically just |error| — kept
        # as an explicit RMS formula in case this later aggregates multiple ref_times.
        residual_pre_nudge = np.concatenate(err_val_pre)

        # QC only nulls a value, never drops the row, so lat/lon are still present.
        qc_flag_col = f"{col}_qc_dropped"
        if qc_flag_col in stations.columns:
            qc_mask = stations[qc_flag_col].fillna(False).astype(bool)
        else:
            qc_mask = pd.Series(False, index=stations.index)

        self._reliability_diag.setdefault(shortname, {}).update({
            "err_latitude": np.concatenate(err_lat),
            "err_longitude": np.concatenate(err_lon),
            "residual_pre_nudge": residual_pre_nudge,
            "pre_nudge_rmse": np.sqrt(residual_pre_nudge ** 2),
            "post_nudge_rmse": np.sqrt(np.concatenate(err_val_post) ** 2),
            "err_is_holdout": np.concatenate(err_is_holdout),
            "qc_dropped_latitude": stations.loc[qc_mask, "latitude"].to_numpy(),
            "qc_dropped_longitude": stations.loc[qc_mask, "longitude"].to_numpy(),
            "qc_dropped_ids": stations.index[qc_mask].tolist(),
        })

    def _compute_reliability(
        self,
        shortname: str,
        sta_ids: list,
        st_lon: np.ndarray,
        st_lat: np.ndarray,
        r_at_st: np.ndarray,
        sta_res: xr.Dataset,
        sta_topo: Optional[xr.Dataset],
        max_dist: float,
    ) -> xr.DataArray:
        """Leave-one-out spatial-consistency check (v4, ported unchanged from
        notebooks/nudging_analysis_v10.ipynb — see module docstring, "Station
        reliability", for the full algorithm).

        Down-weights (via a shrunk influence radius, applied by the caller)
        stations whose residual disagrees with what their own neighbours
        predict, independent of any particular POI. ``sta_topo`` is ``None``
        when *use_topo_descriptors* is ``False``, in which case the
        leave-one-out prediction falls back to pure IDW. Reuses arrays already
        computed by ``_nudge_field`` for the same station set/variable rather
        than recomputing them (a pure dedup, no numerical effect).

        Returns an xr.DataArray (dims=["sta"]) of reliability in [0, 1].
        """
        n_sta = len(sta_ids)

        # ── Station <-> station barrier-aware distances ────────────────────
        ned_ss = self._get_d_eff_sta(shortname, sta_ids)

        # Stations stand in as both "sta" and "poi" for the leave-one-out prediction.
        poi_topo = (
            sta_topo.rename({"sta": "poi"}).assign_coords({"poi": sta_ids})
            if sta_topo is not None else None
        )

        # ── Leave-one-out neighbour prediction ──────────────────────────────
        # No reliability weighting here: this first pass must trust every OTHER
        # station equally, or a bad station could suppress its own signal.
        r_hat = ned_interp(
            sta_res, ned_ss,
            sta_topo=sta_topo, poi_topo=poi_topo,
            max_dist=max_dist, weight_power=self.weight_power,
            min_topo_w=self.min_topo_w, lim_effective=self.lim_effective,
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
                shortname, n_isolated, n_sta, max_dist,
            )
        if finite.any():
            med_e = float(np.median(e[finite]))
            mad   = float(np.median(np.abs(e[finite] - med_e)))
        else:
            med_e, mad = 0.0, 0.0
        scale = max(1.4826 * mad, self.reliability_eps)

        u = np.zeros(n_sta, dtype=np.float64)
        u[finite] = (e[finite] - med_e) / scale
        reliability_vals = np.ones(n_sta, dtype=np.float64)
        reliability_vals[finite] = np.clip(1.0 - (u[finite] / self.number_of_std) ** 2, 0.0, None) ** 2

        n_flagged = int((reliability_vals < 0.5).sum())
        LOG.debug(
            "compute_reliability('%s'): %d stations, median(e)=%.4f, MAD(e)=%.4f, "
            "scale=%.4f, %d station(s) with reliability < 0.5",
            shortname, n_sta, med_e, mad, scale, n_flagged,
        )

        # Side channel for the optional plot only; does not feed back into reliability.
        self._reliability_diag[shortname] = {
            "sta_ids": sta_ids, "r_at_st": r_at_st, "r_hat": r_hat_at_st,
            "e": e, "u": u, "reliability": reliability_vals,
            "latitude": st_lat, "longitude": st_lon,
        }

        return xr.DataArray(
            reliability_vals.astype(np.float32), dims=["sta"], coords={"sta": sta_ids},
        )

    def _plot_reliability_diagnostic(self, shortname, ref_time, background, corrected) -> None:
        """Save the 6-panel reliability diagnostic PNG for one variable/ref_time:
        station residual (holdin+holdout, pre-nudging), station reliability
        (holdin only), gridded correction (these three match
        notebooks/nudging_analysis_v10.ipynb's diagnostic cell), pre-/
        post-nudging station RMSE (holdin circles vs. holdout triangles), and
        stations CleanObservation's QC removed for this variable.

        Opt-in via *plot_dir* and *enable_plotting* (see ``_nudge_field``'s
        call site). Never raises: any failure is logged and swallowed so it
        can't affect the nudging correction, already computed by this point.
        """
        if shortname not in self._reliability_diag:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend — safe in headless/batch jobs
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            diag = self._reliability_diag[shortname]
            flagged = diag["reliability"] < 0.5
            is_ho = diag["err_is_holdout"]
            station_res = diag["residual_pre_nudge"]  # holdin + holdout, pre-nudging

            extent = self.plot_extent
            ch_mask = (
                (self._lon_icon >= extent[0]) & (self._lon_icon <= extent[1]) &
                (self._lat_icon >= extent[2]) & (self._lat_icon <= extent[3])
            )
            residual_full = background - corrected  # the correction actually applied
            lo_ch, la_ch = self._lon_icon[ch_mask], self._lat_icon[ch_mask]
            res_ch = residual_full[ch_mask]

            # Percentile per source, combined via max, so gridded points don't dilute
            # station outliers' influence on the scale, or vice versa.
            res_abs = max(
                float(np.percentile(np.abs(station_res), _DIAG_COLORBAR_PERCENTILE)) if len(station_res) else 0.0,
                float(np.percentile(np.abs(res_ch), _DIAG_COLORBAR_PERCENTILE)) if len(res_ch) else 0.0,
            ) or 1.0  # guard against an all-zero degenerate case
            vmin, vmax = -res_abs, res_abs

            # Shared colour scale for the two RMSE panels (same rationale as above).
            err_vmax = max(
                float(np.percentile(diag["pre_nudge_rmse"], _DIAG_COLORBAR_PERCENTILE)) if len(diag["pre_nudge_rmse"]) else 0.0,
                float(np.percentile(diag["post_nudge_rmse"], _DIAG_COLORBAR_PERCENTILE)) if len(diag["post_nudge_rmse"]) else 0.0,
            ) or 1.0

            def _base(ax):
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
                ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
                ax.add_feature(cfeature.LAKES, facecolor="lightblue", alpha=0.5)

            def _colorbar(ax, mappable, label):
                # plt.colorbar's fraction/pad doesn't track a cartopy GeoAxes' rendered aspect.
                cax = make_axes_locatable(ax).append_axes(
                    "right", size="5%", pad=0.3, axes_class=plt.Axes
                )
                return fig.colorbar(mappable, cax=cax, label=label)

            def _holdin_holdout_scatter(ax, lon, lat, values, s=45, **kwargs):
                sc = ax.scatter(
                    lon[~is_ho], lat[~is_ho], c=values[~is_ho], s=s,
                    marker="o", edgecolors="black", linewidths=0.4,
                    transform=ccrs.PlateCarree(), zorder=5, label="holdin", **kwargs,
                )
                if is_ho.any():
                    ax.scatter(
                        lon[is_ho], lat[is_ho], c=values[is_ho], s=s * 2,
                        marker="^", edgecolors="black", linewidths=0.6,
                        transform=ccrs.PlateCarree(), zorder=6, label="holdout", **kwargs,
                    )
                    ax.legend(loc="lower left", fontsize=8)
                return sc

            fig = plt.figure(figsize=(26, 9))

            ax0 = fig.add_subplot(2, 3, 1, projection=ccrs.PlateCarree())
            _base(ax0)
            sc0 = _holdin_holdout_scatter(
                ax0, diag["err_longitude"], diag["err_latitude"], station_res,
                cmap="RdBu_r", vmin=vmin, vmax=vmax,
            )
            _colorbar(ax0, sc0, "residual (background − obs)")
            ax0.set_title(f"Station residuals (holdin + holdout) — {shortname} ({ref_time})")

            ax1 = fig.add_subplot(2, 3, 2, projection=ccrs.PlateCarree())
            _base(ax1)
            sc1 = ax1.scatter(
                diag["longitude"][~flagged], diag["latitude"][~flagged],
                c=diag["reliability"][~flagged], cmap="RdYlGn", vmin=0, vmax=1, s=45,
                marker="o", edgecolors="black", linewidths=0.4,
                transform=ccrs.PlateCarree(), zorder=5, label="reliability ≥ 0.5",
            )
            if flagged.any():
                ax1.scatter(
                    diag["longitude"][flagged], diag["latitude"][flagged],
                    c=diag["reliability"][flagged], cmap="RdYlGn", vmin=0, vmax=1, s=90,
                    marker="X", edgecolors="black", linewidths=0.6,
                    transform=ccrs.PlateCarree(), zorder=6, label="flagged (reliability < 0.5)",
                )
                ax1.legend(loc="lower left", fontsize=8)
            _colorbar(ax1, sc1, "reliability")
            ax1.set_title(f"Station reliability (holdin only) — {shortname} ({ref_time})")

            ax2 = fig.add_subplot(2, 3, 3, projection=ccrs.PlateCarree())
            _base(ax2)

            levels = np.linspace(vmin, vmax, 51)
            tcf2 = ax2.tricontourf(
                lo_ch, la_ch, res_ch, levels=levels, cmap="RdBu_r",
                vmin=vmin, vmax=vmax, extend="both", transform=ccrs.PlateCarree(),
            )
            _colorbar(ax2, tcf2, "interpolated residual (background − corrected)")
            ax2.set_title(f"Interpolated residuals — {shortname} ({ref_time})")

            ax3 = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
            _base(ax3)
            sc3 = _holdin_holdout_scatter(
                ax3, diag["err_longitude"], diag["err_latitude"], diag["pre_nudge_rmse"],
                cmap="viridis", vmin=0, vmax=err_vmax,
            )
            _colorbar(ax3, sc3, "pre-nudging RMSE (station − background)")
            ax3.set_title(f"Pre-nudging station error — {shortname} ({ref_time})")

            ax4 = fig.add_subplot(2, 3, 5, projection=ccrs.PlateCarree())
            _base(ax4)
            sc4 = _holdin_holdout_scatter(
                ax4, diag["err_longitude"], diag["err_latitude"], diag["post_nudge_rmse"],
                cmap="viridis", vmin=0, vmax=err_vmax,
            )
            _colorbar(ax4, sc4, "post-nudging RMSE (station − corrected)")
            ax4.set_title(f"Post-nudging station error — {shortname} ({ref_time})")

            ax5 = fig.add_subplot(2, 3, 6, projection=ccrs.PlateCarree())
            _base(ax5)

            ax5.scatter(
                diag["err_longitude"], diag["err_latitude"],
                s=25, c="lightgrey", marker="o", edgecolors="black", linewidths=0.3,
                transform=ccrs.PlateCarree(), zorder=4, label="kept",
            )
            qc_lon = diag.get("qc_dropped_longitude", np.array([]))
            qc_lat = diag.get("qc_dropped_latitude", np.array([]))
            if len(qc_lon):
                ax5.scatter(
                    qc_lon, qc_lat,
                    s=90, c="red", marker="X", edgecolors="black", linewidths=0.6,
                    transform=ccrs.PlateCarree(), zorder=6, label="removed by QC",
                )
            ax5.legend(loc="lower left", fontsize=8)
            ax5.set_title(f"Stations removed by QC ({len(qc_lon)}) — {shortname} ({ref_time})")

            plt.tight_layout()
            ref_time_str = ref_time.strftime("%Y%m%d%H%M") if ref_time is not None else "unknown"
            out_path = self.plot_dir / f"reliability_diag_{shortname}_{ref_time_str}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            LOG.info("Saved reliability diagnostic plot for '%s' to %s", shortname, out_path)
        except Exception:
            LOG.exception(
                "Reliability diagnostic plot failed for '%s'; continuing without it "
                "(the nudging correction itself is unaffected).",
                shortname,
            )

    # ── Holdout and station loading ───────────────────────────────────────────

    def _apply_holdout(self, stations: pd.DataFrame) -> pd.DataFrame:
        """Remove stations from the nudging set according to holdout configuration."""
        if self.exclude_stations is not None:
            before = len(stations)
            missing = [s for s in self.exclude_stations if s not in stations.index]
            if missing:
                LOG.warning("Excluded station IDs not found in observations: %s", missing)
            stations = stations.drop(
                index=[s for s in self.exclude_stations if s in stations.index]
            )
            LOG.info(
                "Excluded %d station(s) by ID: %s",
                before - len(stations),
                self.exclude_stations,
            )

        elif self.holdout_fraction is not None:
            if self.holdout_fraction == 0.0:
                LOG.info("holdout_fraction=0: all stations used.")
            elif self.holdout_fraction == 1.0:
                LOG.info("holdout_fraction=1: all stations withheld.")
                stations = stations.iloc[0:0]
            else:
                n_holdout = round(len(stations) * self.holdout_fraction)
                if n_holdout == 0:
                    LOG.warning(
                        "holdout_fraction=%.4f rounds to 0 station(s) held out "
                        "of %d — no cross-validation holdout set will be "
                        "available this run.",
                        self.holdout_fraction, len(stations),
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
        missing_cols = {"latitude", "longitude"} - set(stations.columns)
        if missing_cols:
            raise ValueError(
                f"Observations Parquet {self.obs_path} is missing required "
                f"column(s) {sorted(missing_cols)}."
            )
        return stations
