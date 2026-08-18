"""
NudgeTowardObservation — v3 (distances in km)

Implements Interpolation of Residuals (IoR) using ned_interp combined with
barrier-aware effective distances derived from a 1 km-scale DEM.

Algorithm per variable
----------------------
1.  Project ICON grid and station coordinates to Swiss LV95 (EPSG:2056) — the
    same exact metric CRS already used for the barrier-distance DEM — and
    express distances in km. This replaces the (lon * cos(lat0), lat)
    equirectangular approximation used by earlier versions, which was only
    exactly isotropic near the domain's mean latitude.
2.  Restrict POIs to a buffer around the station bounding box derived from
    max_dist. LV95 is isotropic metric, so a simple symmetric km buffer
    suffices — no lat/lon asymmetry needed, unlike the old projection.
3.  Compute Euclidean distance matrix (n_poi × n_sta) in km.
4.  Inflate distances with a barrier term and an elevation-difference term
    (barrier_distances): the DEM is sampled along a perpendicular slab at each
    along-path step; a Gaussian-weighted mean across the corridor width and a
    95th-percentile along the path give the effective ridge height.
        d_eff = sqrt(d_euc² + (barrier/elev_scale)² + (elev_diff/elev_diff_scale)²)
    All terms are in km; elev_scale/elev_diff_scale are in m/km (a ridge of
    elev_scale metres now adds 1 km of effective distance).
5.  Compute topographic similarity per (POI, station) pair (TPI, slope
    derivatives, DEM/ICON elevation): each descriptor's importance is
    |Pearson corr| between it and the station residuals, normalised to sum
    to 1 per variable — a descriptor uncorrelated with this variable's
    residuals contributes ~nothing, one that tracks the bias pattern
    dominates the blend.
6.  Spread station residuals to the POI grid via ned_interp: IDW
    (1 / d_eff^weight_power) weighted by the topographic similarity from
    step 5, floored at min_topo_w so nearby stations always contribute.
7.  Multiply the correction by a linear taper that fades to zero at max_dist
    km from the nearest station.
8.  Subtract the tapered correction from the background field.

Weight computation, symbolically (per POI p, station s)
---------------------------------------------------------------------------
Step 1 — raw distance (km):     d_euc[p,s]
Step 2 — elevation-aware (km):  d_eff[p,s]     = barrier_distances(d_euc)          → "ned_sta_poi"
Step 3 — descriptor importance: importance[d]  = |corr_s(residual[s], descriptor_d[s])| / Σ_d |corr_s(...)|
                                 (one weight per descriptor d, computed across stations s; recomputed per variable)
Step 4 — topo similarity:       w_topo[p,s]    = Σ_d importance[d] * (1 - |sta_topo[d,s] - poi_topo[d,p]|), floored at min_topo_w
Step 5 — combine:               raw_w[p,s]     = w_topo[p,s] * (1 / d_eff[p,s]^weight_power)   [NaN if d_eff ≥ max_dist]
Step 6 — normalize per POI:     w_ned[p,s]     = raw_w[p,s] / (Σ_s raw_w[p,s] + lim_effective)
Step 7 — interpolate:           correction_raw[p] = Σ_s w_ned[p,s] * residual[s]
Step 8 — taper (separate!):     taper[p] = 1 - clip(nearest_station_raw_dist_km[p] / max_dist, 0, 1)
Step 9 — apply:                 background[p] -= correction_raw[p] * taper[p]

Note taper (step 8) uses raw d_euc (km), not d_eff — the geographic fade-out is
deliberately independent of the barrier logic (see barrier_distances docstring).

Unit history: earlier versions expressed max_dist/elev_scale/elev_diff_scale in
projected degrees and m/° respectively (1 projected degree ≈ 111.32 km near
Swiss latitudes). Default values below are those degree-tuned values converted
to km/m-per-km, so default behaviour is preserved rather than re-tuned;
deployment configs must supply km/m-per-km values directly (see the YAML
config for this filter).
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

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

# Maps GRIB shortName (COSMO/ICON) → (station Parquet column, unit offset applied to obs).
# The offset is added to the raw observation value before computing residuals.
PARAM_MAP = {
    "T_2M":    ("2t",   0.0),   # 2 m temperature         [K]
    "TD_2M":   ("2d",   0.0),   # 2 m dewpoint            [K]
    "U_10M":   ("10u",  0.0),   # 10 m U wind component   [m/s]
    "V_10M":   ("10v",  0.0),   # 10 m V wind component   [m/s]
    "PMSL":    ("msl",  0.0),   # mean sea-level pressure  [Pa]
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
# ICON's own native orography (extpar, ASTER-derived) on the same R19B08 grid — used
# only as the elevation *topo descriptor* for ned_interp's similarity weighting, since
# it is what the model itself "sees" as terrain. The elevation-aware barrier distance
# (barrier_distances) intentionally keeps using the finer external DEM (dem_barrier_file)
# instead, since that term needs the true terrain, not the model's (smoothed) view of it.
_DEFAULT_ICON_OROG_FILE = (
    "/scratch/mch/icontest/testing-input-data/c2sm/icon-1/"
    "external_parameter_icon_grid_0001_R19B08_mch.nc"
)

# Standard-atmosphere lapse rate, applied to reduce station observations to the
# model's elevation before differencing (see NudgeTowardObservation.lapse_rate).
_DEFAULT_LAPSE_RATE = 0.0065  # K/m
# T_2M only: dry-bulb temperature follows the standard-atmosphere lapse rate closely.
# TD_2M (dewpoint) does not decrease with elevation at the same fixed rate, so it is
# excluded by default; pressure is already sea-level-reduced and wind has no lapse rate.
_DEFAULT_LAPSE_RATE_VARS = frozenset({"T_2M"})


# ── ned_interp (adapted from data4web_pipelines/utils.py) ─────────────────────
# Direct import is not possible because data4web_pipelines depends on
# internal MeteoSwiss libraries not available in the production environment.


def ned_interp(
    sta_res: xr.Dataset,
    ned_sta_poi: xr.DataArray,
    sta_topo: Optional[xr.Dataset] = None,
    poi_topo: Optional[xr.Dataset] = None,
    max_dist: Optional[float] = None,
    weight_power: float = 1,
    min_topo_w: float = 0.2,
    lim_effective: float = 0,
) -> xr.Dataset:
    """Spread station residuals to POIs via IDW with optional topographic similarity.

    Source: data4web_pipelines/utils.py.

    Steps when topo descriptors are provided
    ----------------------------------------
    1.  Mask stations beyond max_dist: set their distance to NaN so their
        inverse-distance weight becomes NaN (effectively 0 after normalisation).
    2.  Joint normalisation: scale both sta_topo and poi_topo to [0, 1] using the
        combined min/max over both sets. This is critical — normalising them
        separately would put ~150 station values and ~400 k POI values on different
        [0, 1] scales, making delta_topo = 1 − |sta − poi| inconsistent.
    3.  Descriptor importance: |Pearson corr(residuals, descriptor)| across stations,
        normalised so descriptor weights sum to 1. Data-driven: if a descriptor has
        no correlation with the residuals it receives zero importance.
    4.  Topographic similarity per (POI, station) pair: weighted average of
        (1 − |sta_val − poi_val|) across descriptors.
    5.  Floor topo similarity at min_topo_w so IDW still applies to nearby stations
        even when they are topographically dissimilar (e.g. valley vs. ridge).
    6.  Normalise final weights so they sum to 1 per POI. lim_effective > 0 adds a
        virtual zero-residual station to the denominator, shrinking corrections in
        data-sparse regions.
    """
    # Step 1 — exclude stations beyond the cutoff radius.
    # ned_sta_poi has dims (poi, sta). Where the condition is False (distance ≥ max_dist),
    # the value is replaced by NaN. NaN distances → NaN IDW weights → effectively excluded.
    if max_dist is not None:
        ned_sta_poi = ned_sta_poi.where(ned_sta_poi < max_dist)

    if sta_topo is None or poi_topo is None:
        # Pure IDW fallback when no topographic descriptors are available.
        w_ned = 1 / np.power(ned_sta_poi, weight_power)
    else:
        # Step 2 — joint normalisation: iterate over each topo descriptor variable.
        # We modify copies to avoid mutating the caller's datasets.
        sta_topo = sta_topo.copy()
        poi_topo = poi_topo.copy()
        for var in list(sta_topo.data_vars):
            vmin = float(min(float(sta_topo[var].min()), float(poi_topo[var].min())))
            vmax = float(max(float(sta_topo[var].max()), float(poi_topo[var].max())))
            rng = (vmax - vmin) if (vmax - vmin) > 0 else 1.0  # guard against constant fields
            sta_topo[var] = ((sta_topo[var] - vmin) / rng).astype(sta_topo[var].dtype)
            poi_topo[var] = ((poi_topo[var] - vmin) / rng).astype(poi_topo[var].dtype)

        # Topographic similarity per (poi, sta, descriptor): 1 = identical, 0 = opposite extremes.
        # xarray broadcasts (sta,) and (poi,) dimensions automatically, yielding shape (poi, sta)
        # per variable — one entry for each POI–station pair.
        delta_topo = 1 - abs(sta_topo - poi_topo)

        # Step 3 — descriptor importance: |Pearson corr| across stations, normalised to sum = 1.
        # sta_res.to_array("data_var") → DataArray(data_var, sta)
        # sta_topo.to_array("topo")   → DataArray(topo, sta)
        # xr.corr(..., dim="sta")     → DataArray(data_var, topo), one corr value per (var, topo)
        # After abs and transpose: shape (topo, data_var). Converted back to Dataset so that
        # downstream multiplication with delta_topo (a Dataset) works per-variable.
        w_topo = (
            abs(
                xr.corr(
                    sta_res.to_array("data_var"),
                    sta_topo.to_array("topo"),
                    dim="sta",
                ).astype(np.float32)
            )
            .transpose("topo", ...)   # ensure "topo" leads for the division below
            .to_dataset("data_var")   # Dataset{var: DataArray(topo,)} — one weight per descriptor
        )
        w_topo /= w_topo.sum("topo")  # normalise: descriptor importances sum to 1 per variable

        # Step 4 — weighted-average topo similarity across descriptors → shape (poi, sta) per var.
        # delta_topo.to_array("topo") → DataArray(topo, poi, sta).
        # w_topo is Dataset{var: DataArray(topo,)}. The product broadcasts topo over (poi, sta),
        # then .sum("topo") collapses the descriptor axis.
        w_topo = (w_topo * delta_topo.to_array("topo")).sum("topo")

        # Step 5 — floor and combine: clip topo similarity from below so that nearby stations
        # always contribute even when topographically dissimilar; multiply by the IDW weight.
        w_ned = w_topo.clip(min=min_topo_w) * (1 / np.power(ned_sta_poi, weight_power))

    # Step 6 — normalise weights across stations so they sum to 1 per POI.
    # Adding lim_effective to the denominator introduces a virtual zero-residual station,
    # which shrinks the total correction in regions with few real stations.
    w_ned /= w_ned.sum("sta") + lim_effective

    # Weighted sum of residuals over stations → Dataset{var: DataArray(poi,)}.
    # min_count=1: POIs whose every station is masked (NaN weight) return NaN instead of 0.
    # The caller converts NaN to 0 via nan_to_num, giving zero correction for those POIs.
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
    n_samples: int = 35,
    elev_scale: float = 17.966,
    elev_diff_scale: float = 35.932,
    n_barrier_width_samples: int = 3,
    barrier_width_m: float = 1500.0,
) -> np.ndarray:
    """Replace Euclidean distances with elevation-aware effective distances.

    For each (POI, station) pair with d_euc < max_dist:
        d_eff = sqrt(d_euc² + (barrier / elev_scale)² + (elev_diff / elev_diff_scale)²)

    d_euc/max_dist are expected in km; elev_scale/elev_diff_scale in m/km (a
    ridge of elev_scale metres adds 1 km of effective distance). The function
    itself is unit-agnostic — whatever consistent distance unit d_euc/max_dist
    are given in is what d_eff comes out in — but NudgeTowardObservation always
    calls this with km.

    Barrier term
    ------------
    At each of n_samples interior points along the straight-line path (endpoints
    excluded — see note below), n_barrier_width_samples DEM points span a
    ±barrier_width_m corridor perpendicular to the path. A Gaussian-weighted mean
    (sigma = barrier_width_m / 2, centre-weighted) across the corridor is taken at
    each step. The 95th percentile of those means along the path gives the effective
    ridge height (robust to DEM spikes):
        barrier = max(0, percentile_95(gauss_mean_cross) − max(elev_poi, elev_sta))

    Why exclude endpoints: t = 0 and t = 1 place the perpendicular sample grid
    centred on the POI or station itself. The resulting DEM samples would represent
    terrain beside the endpoint (in a direction perpendicular to the path) rather than
    terrain between the two points, potentially counting a nearby hill as a barrier.

    Why Gaussian weighting: a uniform corridor would weight off-axis terrain equally.
    The Gaussian down-weights terrain at the corridor edges so the direct path remains
    dominant; sigma = barrier_width_m / 2 keeps ~95 % of the weight within the corridor.

    elev_diff term
    --------------
    |elev_poi − elev_sta| penalises pairs at very different altitudes even when no
    ridge intervenes, capturing different vertical atmospheric regimes (e.g. a valley
    station vs. a high-altitude POI).

    The two terms are added in quadrature (Pythagorean combination), so a large ridge
    and a large elevation difference compound each other.

    Implementation notes
    --------------------
    - Coordinates are projected to LV95 (Swiss metric CRS, EPSG:2056) so that path
      lengths and perpendicular offsets are in metres.
    - elev_poi is read from the 1 km DEM; sta_elev should come from DWH station
      metadata (more accurate than DEM interpolation at station locations).
    - Only pairs with d_euc < max_dist are processed; all others keep their d_euc.
    - Unique endpoints are projected only once (deduplication before the transform).
    """
    # Boolean mask of pairs to process; pi_idx/si_idx are the row/col flat indices.
    close_mask = d_euc < max_dist
    pi_idx, si_idx = np.where(close_mask)  # shape: (n_close,) each

    if len(pi_idx) == 0:
        return d_euc

    # Deduplicate before the expensive WGS84 → LV95 transform.
    # Many pairs share the same POI (one ICON cell near many stations) or the same station.
    # u_poi/u_sta: unique POI row indices / unique station col indices
    # inv_poi/inv_sta: mapping from unique back to the flat pair list
    u_poi, inv_poi = np.unique(pi_idx, return_inverse=True)
    u_sta, inv_sta = np.unique(si_idx, return_inverse=True)

    # Project unique WGS84 coordinates to LV95 (easting x, northing y) in metres.
    # always_xy=True means input is (longitude, latitude), output is (easting, northing).
    poi_x_u, poi_y_u = wgs84_to_lv95.transform(poi_lon[u_poi], poi_lat[u_poi])
    sta_x_u, sta_y_u = wgs84_to_lv95.transform(sta_lon[u_sta], sta_lat[u_sta])

    # POI elevation from DEM; DEM is more reliable than nearest-cell interpolation.
    # Station elevation from DWH metadata — instrument altitude, more accurate than DEM.
    elev_poi = dem_rgi(np.c_[poi_y_u, poi_x_u])[inv_poi]  # (n_close,)
    elev_sta = sta_elev[si_idx]                             # (n_close,)

    # Barrier is terrain above the *higher* endpoint (a peak below the higher end is not a barrier).
    ref_elev = np.maximum(elev_poi, elev_sta)

    # Expand unique LV95 coords to one entry per pair, matching pi_idx/si_idx order.
    poi_xp = poi_x_u[inv_poi]   # (n_close,)
    poi_yp = poi_y_u[inv_poi]
    sta_xp = sta_x_u[inv_sta]
    sta_yp = sta_y_u[inv_sta]

    # ── Along-path grid (interior points only) ────────────────────────────────
    t = np.linspace(0, 1, n_samples + 2)[1:-1]  # (n_samples,), excludes t=0 and t=1
    x_path = poi_xp[None, :] + t[:, None] * (sta_xp - poi_xp)[None, :]  # (n_samples, n_close)
    y_path = poi_yp[None, :] + t[:, None] * (sta_yp - poi_yp)[None, :]

    # ── Perpendicular-slab grid ───────────────────────────────────────────────
    # Unit vector 90° to the path direction: rotate (dx, dy) by 90° → (-dy, dx), then normalise.
    dx = sta_xp - poi_xp  # (n_close,)
    dy = sta_yp - poi_yp
    path_len = np.sqrt(dx ** 2 + dy ** 2)
    safe_len = np.where(path_len > 0, path_len, 1.0)  # avoid /0 for co-located pairs
    perp_x = -dy / safe_len  # (n_close,) unit perpendicular, x-component
    perp_y =  dx / safe_len  # (n_close,) unit perpendicular, y-component

    # Symmetric offsets across the corridor in metres: centred on the straight-line path.
    perp_offsets = np.linspace(-barrier_width_m, barrier_width_m, n_barrier_width_samples)

    # Gaussian weights: centre-weighted so off-axis terrain contributes less.
    # When barrier_width_m = 0 (single centre sample), sigma = 0 → uniform weight of 1.
    sigma = barrier_width_m / 2.0
    if sigma > 0:
        gauss_w = np.exp(-0.5 * (perp_offsets / sigma) ** 2)
    else:
        gauss_w = np.ones(n_barrier_width_samples)
    gauss_w /= gauss_w.sum()  # (n_perp,), normalised to sum = 1

    # Slab coordinates: for each along-path step and each corridor offset, compute the
    # LV95 position of the DEM sample point.
    # x_slab shape: (n_samples, n_perp, n_close)
    x_slab = x_path[:, None, :] + perp_offsets[None, :, None] * perp_x[None, None, :]
    y_slab = y_path[:, None, :] + perp_offsets[None, :, None] * perp_y[None, None, :]

    # Evaluate DEM at all slab points in one vectorised RGI call.
    # RGI expects (northing, easting) = (y, x) as the first axis.
    n_perp  = n_barrier_width_samples
    n_close = len(pi_idx)
    elev_slab = dem_rgi(
        np.c_[y_slab.ravel(), x_slab.ravel()]
    ).reshape(n_samples, n_perp, n_close)

    # Gaussian-weighted mean across the perpendicular corridor at each along-path step.
    # gauss_w broadcast: (n_perp,) → (1, n_perp, 1) to multiply over the middle axis.
    elev_mean_cross = (elev_slab * gauss_w[None, :, None]).sum(axis=1)  # (n_samples, n_close)

    # 95th percentile along the path direction: robust to DEM spikes while still
    # capturing the dominant ridge. Subtract ref_elev; clamp to ≥ 0 (a pass lower than
    # both endpoints is not a barrier).
    barrier = np.maximum(
        0.0, np.percentile(elev_mean_cross, 95, axis=0) - ref_elev
    ).astype(np.float32)

    # Elevation-difference term: altitude gap between endpoints, independent of terrain.
    elev_diff = np.abs(elev_poi - elev_sta).astype(np.float32)

    # Pythagorean combination: barrier and elevation-difference contribute independently.
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


# ── Production filter ─────────────────────────────────────────────────────────


class NudgeTowardObservation(Filter):
    """Nudge the forecast initial condition toward surface station observations.

    Implements v3 Interpolation of Residuals using ned_interp with topographic
    similarity weighting and barrier-aware effective distances (DEM path sampling).

    The filter reads pre-fetched station observations from a Parquet file written
    by RetrieveObservation (which must include ``elevation`` as produced by the
    current version of that filter). Nudging is applied once — to the
    initial-condition time step — and the filter passes all fields through
    unchanged on subsequent calls.

    Parameters
    ----------
    obs_path : str
        Path to the cleaned station observations Parquet file.
        Required columns: ``latitude``, ``longitude``, ``elevation``, and one
        column per nudged variable (e.g. ``2t``, ``10u``), all in SI units.
    icon_grid_dir : str
        Directory containing ``icon_grid_0001_R19B08_mch.nc``.
    topo_file : str
        Path to the topographic descriptor NetCDF on the ICON R19B08 grid
        (``topo_descriptors_icon_R19B08.nc``). Must contain ``lon`` and ``lat``
        coordinate variables and all variables listed in *topo_vars* except
        ``ICON_OROG``, which is injected separately from *icon_orog_file*.
    dem_barrier_file : str
        Path to the 1 km DEM NetCDF (variable ``DEM_1000M``, coordinates ``x``
        and ``y`` in LV95 metres) used for barrier path sampling.
    icon_orog_file : str
        Path to the ICON extpar NetCDF for the R19B08 grid (variable
        ``topography_c``) providing the model's own native orography. Used
        only as the ``ICON_OROG`` topo descriptor for ned_interp's
        topographic-similarity weighting — distinct from *dem_barrier_file*,
        which continues to drive the elevation-aware barrier distance.
    weight_power : float
        IDW distance-decay exponent. Higher values concentrate weight on the
        nearest station (notebook: ``WEIGHT_POWER = 4``).
    max_dist : float
        Station influence radius in km; both the barrier-distance cutoff and
        the linear taper radius (notebook v8: ``MAX_DIST_KM``; default here is
        the historical degree-tuned value (0.35°) converted to km, ≈ 38.96).
    n_barrier_samples : int
        DEM sample points along the straight-line path interior (endpoints
        excluded) (notebook: ``N_BARRIER_SAMPLES = 35``).
    n_barrier_width_samples : int
        Perpendicular samples per along-path step; odd values centre the
        slab on the straight line (notebook: ``N_BARRIER_WIDTH_SAMPLES = 3``).
    barrier_width_m : float
        Half-width of the perpendicular DEM corridor in metres
        (notebook: ``BARRIER_WIDTH_M = 1500``).
    elev_scale : float
        Ridge height in metres that adds 1 km to effective distance, i.e. m/km
        (notebook v8: ``ELEV_SCALE_KM``; default here is the historical
        degree-tuned value (2000 m/°) converted to m/km, ≈ 17.97).
    elev_diff_scale : float
        Endpoint elevation difference in metres that adds 1 km to effective
        distance — weaker penalty than a ridge, i.e. m/km (notebook v8:
        ``ELEV_DIFF_SCALE_KM``; default here is the historical degree-tuned
        value (4000 m/°) converted to m/km, ≈ 35.93).
    min_topo_w : float
        Minimum topographic similarity weight floor so nearby stations always
        contribute (notebook: ``MIN_TOPO_W = 0.2``).
    lim_effective : float
        Virtual zero-residual station weight added to the normalisation
        denominator; 0 = pure IDW (notebook: ``LIM_EFFECTIVE = 0.0``).
    lapse_rate : float
        Standard-atmosphere lapse rate [K/m] used to reduce station observations
        to the model's elevation before computing the residual, so the residual
        reflects model bias rather than the elevation mismatch between the true
        station altitude and ICON's (smoothed) orography at the nearest grid
        cell: ``obs_corrected = obs - lapse_rate * (elev_model_at_cell - elev_sta)``.
        Only applied to variables in *lapse_rate_vars* (default: 0.0065, i.e. 6.5 K/km).
    lapse_rate_vars : list of str, optional
        GRIB shortNames to which *lapse_rate* is applied. Restricted to
        ``T_2M`` by default: dry-bulb temperature follows the standard-atmosphere
        lapse rate closely, whereas dewpoint (``TD_2M``) does not decrease with
        elevation at the same fixed rate, pressure is already sea-level-reduced,
        and wind has no lapse rate. Defaults to ``["T_2M"]``.
    topo_vars : list of str, optional
        Topographic descriptor variable names, drawn from *topo_file* plus
        the injected ``ICON_OROG`` (see *icon_orog_file*). Defaults to
        ``["TPI_500M", "TPI_4000M_SMTH", "SN_DERIV_2000M",
        "WE_DERIV_2000M", "ICON_OROG"]``.
    nudge_variables : list of str, optional
        GRIB shortNames to nudge (subset of PARAM_MAP keys). Defaults to all
        non-precipitation variables.
    run_mode : str
        ``'depl'``: ref_time = minimum valid_time across all fields.
        ``'devt'``: ref_time = valid_time of the first field.
    holdout_fraction : float, optional
        Fraction of stations to withhold from nudging for cross-validation.
        Mutually exclusive with *exclude_stations*.
    holdout_seed : int
        RNG seed for station holdout (default 42).
    exclude_stations : list of str, optional
        Station nat_abbr identifiers to unconditionally exclude.
        Mutually exclusive with *holdout_fraction*.
    power : float, optional
        Deprecated alias for *weight_power*. Will be removed in a future version.
    k : int, optional
        Deprecated and ignored. The v3 algorithm uses all stations within
        *max_dist* rather than a fixed k-nearest subset.
    """

    def __init__(
        self,
        obs_path: str,
        icon_grid_dir: str = "/scratch/mch/llanzila/sruc/aux_files",
        topo_file: str = _DEFAULT_TOPO_FILE,
        dem_barrier_file: str = _DEFAULT_DEM_BARRIER_FILE,
        icon_orog_file: str = _DEFAULT_ICON_OROG_FILE,
        weight_power: float = 4.0,
        max_dist: float = 38.962,
        n_barrier_samples: int = 35,
        n_barrier_width_samples: int = 3,
        barrier_width_m: float = 1500.0,
        elev_scale: float = 17.966,
        elev_diff_scale: float = 35.932,
        min_topo_w: float = 0.2,
        lim_effective: float = 0.0,
        lapse_rate: float = _DEFAULT_LAPSE_RATE,
        lapse_rate_vars: Optional[list] = None,
        topo_vars: Optional[list] = None,
        nudge_variables: Optional[list] = None,
        run_mode: str = "depl",
        holdout_fraction: Optional[float] = None,
        holdout_seed: int = 42,
        exclude_stations: Optional[list] = None,
        # Deprecated parameters kept for backward compatibility
        power: Optional[float] = None,
        k: Optional[int] = None,
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

        if power is not None:
            warnings.warn(
                "The 'power' parameter is deprecated; use 'weight_power' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            weight_power = power
        if k is not None:
            warnings.warn(
                "The 'k' parameter is deprecated and ignored in v3. "
                "All stations within max_dist are used.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.obs_path = Path(obs_path)
        self.icon_grid_dir = Path(icon_grid_dir)
        self.topo_file = Path(topo_file)
        self.dem_barrier_file = Path(dem_barrier_file)
        self.icon_orog_file = Path(icon_orog_file)
        self.weight_power = weight_power
        self.max_dist = max_dist
        self.n_barrier_samples = n_barrier_samples
        self.n_barrier_width_samples = n_barrier_width_samples
        self.barrier_width_m = barrier_width_m
        self.elev_scale = elev_scale
        self.elev_diff_scale = elev_diff_scale
        self.min_topo_w = min_topo_w
        self.lim_effective = lim_effective
        self.lapse_rate = lapse_rate
        self.lapse_rate_vars = (
            frozenset(lapse_rate_vars) if lapse_rate_vars is not None else _DEFAULT_LAPSE_RATE_VARS
        )
        self.topo_vars = list(topo_vars) if topo_vars is not None else _DEFAULT_TOPO_VARS
        self.run_mode = run_mode
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed
        self.exclude_stations = list(exclude_stations) if exclude_stations is not None else None
        self._nudging_done = False

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

        # Load heavy static data once at construction time to avoid repeated I/O in forward().
        self._load_icon_grid()
        self._load_topo()
        self._load_dem()

        LOG.info(
            "NudgeTowardObservation v3 initialised: variables=%s, max_dist=%.2f km, "
            "weight_power=%.1f, elev_scale=%.2f m/km, elev_diff_scale=%.2f m/km, "
            "barrier_width_m=%.0f, n_barrier_samples=%d, n_barrier_width_samples=%d, "
            "lapse_rate=%.5f K/m (vars=%s)",
            list(self.param_map.keys()),
            self.max_dist,
            self.weight_power,
            self.elev_scale,
            self.elev_diff_scale,
            self.barrier_width_m,
            self.n_barrier_samples,
            self.n_barrier_width_samples,
            self.lapse_rate,
            sorted(self.lapse_rate_vars),
        )
        super().__init__()

    # ── Static data loaders ───────────────────────────────────────────────────

    def _load_icon_grid(self) -> None:
        ds = xr.open_dataset(self.icon_grid_dir / "icon_grid_0001_R19B08_mch.nc")
        # clat/clon are cell-centre coordinates stored in radians in the ICON grid file.
        self._lat_icon = np.degrees(ds["clat"].values).ravel()  # (n_cells,)
        self._lon_icon = np.degrees(ds["clon"].values).ravel()  # (n_cells,)
        LOG.info(
            "ICON grid loaded: %d cells from %s",
            len(self._lat_icon),
            self.icon_grid_dir / "icon_grid_0001_R19B08_mch.nc",
        )

    def _load_topo(self) -> None:
        self._ds_topo = xr.open_dataset(self.topo_file)
        self._load_icon_orog()
        missing = [v for v in self.topo_vars if v not in self._ds_topo]
        if missing:
            raise ValueError(
                f"Topo variables {missing} not found in {self.topo_file} "
                f"(plus the injected ICON_OROG). Available: {list(self._ds_topo.data_vars)}"
            )
        LOG.info("Topo descriptors loaded from %s: %s", self.topo_file, self.topo_vars)

    def _load_icon_orog(self) -> None:
        """Inject ICON's native orography as the 'ICON_OROG' topo descriptor.

        The 'cell' ordering matches topo_file / icon_grid_dir exactly (same source
        grid file), so the array can be attached positionally without reindexing.
        """
        ds_orog = xr.open_dataset(self.icon_orog_file)
        self._ds_topo["ICON_OROG"] = ("cell", ds_orog["topography_c"].values.astype(np.float32))
        LOG.info("ICON native orography loaded from %s: ICON_OROG", self.icon_orog_file)

    def _load_dem(self) -> None:
        dem_ds = xr.open_dataset(self.dem_barrier_file)
        # Replace NaN (ocean / no-data cells) with 0 m so that out-of-domain path
        # segments don't produce NaN barriers.
        dem_z = np.where(
            np.isnan(dem_ds["DEM_1000M"].values), 0.0, dem_ds["DEM_1000M"].values
        )
        # RGI axes must match the DEM array layout: first axis = y (northing/rows),
        # second axis = x (easting/cols). Query points must be passed as (y, x) = (northing, easting).
        self._dem_rgi = RegularGridInterpolator(
            (dem_ds["y"].values, dem_ds["x"].values),
            dem_z,
            method="linear",
            bounds_error=False,
            fill_value=0.0,  # extrapolate as 0 m for points outside the DEM extent
        )
        # always_xy=True: transform(lon, lat) → (easting, northing), matching the (x, y) convention.
        self._wgs84_to_lv95 = Transformer.from_crs(
            "EPSG:4326", "EPSG:2056", always_xy=True
        )
        LOG.info(
            "DEM loaded from %s: shape=%s",
            self.dem_barrier_file,
            dem_ds["DEM_1000M"].shape,
        )

        # ── Project the ICON grid and topo-descriptor grid to LV95 km, once ────
        # Computed here (not per-nudge-call, and not in a separate method) since
        # self._lon_icon/self._lat_icon (from _load_icon_grid) and self._ds_topo
        # (from _load_topo) are already set by this point in __init__, and
        # self._wgs84_to_lv95 was just created above. Replaces the
        # (lon * cos(lat0), lat) equirectangular approximation used by earlier
        # versions with this exact metric projection, reused by every
        # _nudge_field() call instead of being re-projected each time.
        grid_x, grid_y = self._wgs84_to_lv95.transform(self._lon_icon, self._lat_icon)
        self._grid_xy_km = np.c_[grid_x, grid_y] / 1000.0  # (n_cells, 2), km

        topo_x, topo_y = self._wgs84_to_lv95.transform(
            self._ds_topo["lon"].values, self._ds_topo["lat"].values
        )
        self._topo_xy_km = np.c_[topo_x, topo_y] / 1000.0  # (n_topo_cells, 2), km

        LOG.info(
            "ICON/topo grids projected to LV95: x=[%.1f, %.1f] km, y=[%.1f, %.1f] km",
            self._grid_xy_km[:, 0].min(), self._grid_xy_km[:, 0].max(),
            self._grid_xy_km[:, 1].min(), self._grid_xy_km[:, 1].max(),
        )

    # ── Filter entry point ────────────────────────────────────────────────────

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply IoR nudging to the initial-condition fields; pass subsequent calls through."""
        if self._nudging_done:
            return data

        # Reference time: the time step to which nudging is applied.
        # devt: first field's valid_time (useful for single-step development runs).
        # depl: earliest valid_time across the fieldlist (the true initial condition).
        ref_time = (
            data[0].datetime()["valid_time"]
            if self.run_mode == "devt"
            else min(f.datetime()["valid_time"] for f in data)
        )
        LOG.info("Nudging initial condition at %s", ref_time)

        stations = self._load_stations()
        LOG.info("Stations loaded: %d", len(stations))
        stations = self._apply_holdout(stations)
        LOG.info("Stations after holdout: %d", len(stations))

        nudged = {}
        for field in data.sel(shortName=list(self.param_map.keys())):
            shortname = field.metadata("shortName")

            # Only nudge the initial-condition time step; later steps pass through unchanged.
            if field.datetime()["valid_time"] != ref_time:
                continue

            col, offset = self.param_map[shortname]
            if col not in stations.columns or stations[col].isna().all():
                LOG.warning("No observations for '%s', skipping", shortname)
                continue

            corrected = self._nudge_field(field, stations, shortname, col, offset)
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

        # Rebuild the fieldlist: replace nudged fields at ref_time; keep all others unchanged.
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

    def _nudge_field(
        self,
        field: ekd.Field,
        stations: pd.DataFrame,
        shortname: str,
        col: str,
        offset: float,
    ) -> np.ndarray:
        """Apply v3 IoR nudging to a single field; return the corrected 1-D array."""
        # Background field values on the full ICON grid, shape (n_cells,).
        B_flat = np.asarray(field.values, dtype=float).ravel()

        # ── Valid stations ─────────────────────────────────────────────────
        # Filter to stations that have a non-NaN observation for this variable.
        valid   = stations[col].notna()
        st_lat  = stations.loc[valid, "latitude"].to_numpy()
        st_lon  = stations.loc[valid, "longitude"].to_numpy()
        st_obs  = stations.loc[valid, col].to_numpy() + offset
        sta_ids = stations.loc[valid].index.tolist()

        # Station elevation: use DWH metadata when available (more accurate than DEM).
        # Falls back to DEM sampling for missing values or when the column is absent entirely.
        if "elevation" in stations.columns:
            st_elev = stations.loc[valid, "elevation"].to_numpy(dtype=float)
            nan_mask = np.isnan(st_elev)
            if nan_mask.any():
                # Fill individual NaN entries by sampling the DEM at the station location.
                sta_x, sta_y = self._wgs84_to_lv95.transform(
                    st_lon[nan_mask], st_lat[nan_mask]
                )
                st_elev[nan_mask] = self._dem_rgi(np.c_[sta_y, sta_x])
                LOG.debug(
                    "Filled %d/%d station elevations from DEM (NaN in metadata)",
                    int(nan_mask.sum()), len(st_elev),
                )
        else:
            # Old Parquet files written before elevation was added to RetrieveObservation.
            LOG.warning(
                "Station Parquet missing 'elevation' column; sampling DEM at station "
                "locations. Upgrade RetrieveObservation to include elevation."
            )
            sta_x, sta_y = self._wgs84_to_lv95.transform(st_lon, st_lat)
            st_elev = self._dem_rgi(np.c_[sta_y, sta_x])

        # ── Coordinate projection ──────────────────────────────────────────
        # True metric LV95 projection (km): reuses the grid precomputed once in
        # _load_dem() and the same transformer used there for the DEM. This is
        # an exact projection — 1 km in x equals 1 km in y everywhere in the
        # domain — replacing the (lon * cos(lat0), lat) equirectangular
        # approximation used by earlier versions, which was only exactly
        # isotropic near the domain's mean latitude.
        grid_xy = self._grid_xy_km                                     # (n_cells, 2), km
        sta_x, sta_y = self._wgs84_to_lv95.transform(st_lon, st_lat)
        sta_xy = np.c_[sta_x, sta_y] / 1000.0                           # (n_sta, 2), km

        # ── POI domain ─────────────────────────────────────────────────────
        # Restrict processing to ICON cells within a buffer of the station bounding box.
        # This reduces the n_poi × n_sta distance matrix from ~1.1 M × n_sta to ~100 k × n_sta.
        #
        # LV95 is already isotropic metric (x and y both true km), so a simple
        # symmetric buffer suffices — no lat/lon asymmetry needed, unlike the old
        # cos(lat0)-projected-degree scheme.
        sta_x_min = sta_xy[:, 0].min() - self.max_dist
        sta_x_max = sta_xy[:, 0].max() + self.max_dist
        sta_y_min = sta_xy[:, 1].min() - self.max_dist
        sta_y_max = sta_xy[:, 1].max() + self.max_dist
        dom_mask = (
            (grid_xy[:, 0] >= sta_x_min) & (grid_xy[:, 0] <= sta_x_max) &
            (grid_xy[:, 1] >= sta_y_min) & (grid_xy[:, 1] <= sta_y_max)
        )
        dom_idx = np.where(dom_mask)[0]  # ICON cell indices inside the domain, shape (n_poi,)
        poi_xy  = grid_xy[dom_idx]       # (n_poi, 2), km
        n_poi   = len(dom_idx)

        # ── Residuals at stations ──────────────────────────────────────────
        # For each station, snap to the nearest ICON cell (k=1 nearest neighbour)
        # and compute background_at_cell − observation.
        # Positive residual: model is too high → we subtract a positive correction later.
        _, gi   = cKDTree(grid_xy).query(sta_xy, k=1)

        # Lapse-rate correction: reduce the observation to the model's elevation at
        # that cell before differencing, so the residual reflects model bias rather
        # than the elevation mismatch between the true station altitude and ICON's
        # (smoothed) orography. gi indexes directly into _ds_topo (same source grid
        # as the ICON grid — see _load_icon_orog). Only applied to temperature-like
        # variables (self.lapse_rate_vars); other variables use the raw observation.
        st_obs_lr = st_obs
        if shortname in self.lapse_rate_vars:
            elev_model_at_sta = self._ds_topo["ICON_OROG"].values[gi]
            st_obs_lr = st_obs - self.lapse_rate * (elev_model_at_sta - st_elev)
            LOG.debug(
                "Lapse-rate correction for '%s': mean elev_model−elev_sta = %.0f m, "
                "mean |correction| = %.3f",
                shortname,
                float(np.mean(elev_model_at_sta - st_elev)),
                float(np.mean(np.abs(st_obs_lr - st_obs))),
            )

        r_at_st = B_flat[gi] - st_obs_lr  # (n_sta,): positive when model > observation

        sta_res = xr.Dataset(
            {shortname: xr.DataArray(
                r_at_st.astype(np.float32), dims=["sta"], coords={"sta": sta_ids}
            )}
        )

        # ── Euclidean distance matrix ──────────────────────────────────────
        # Shape: (n_poi, n_sta), in km.
        d_euc_mat = np.sqrt(
            ((poi_xy[:, None, :] - sta_xy[None, :, :]) ** 2).sum(axis=-1)
        ).astype(np.float32)

        # ── Barrier-aware distances ────────────────────────────────────────
        # Inflate d_euc for pairs separated by a ridge or a large elevation step.
        # Pairs pushed beyond max_dist are effectively excluded from ned_interp.
        dist_mat = barrier_distances(
            self._lon_icon[dom_idx], self._lat_icon[dom_idx],  # POI WGS84 lon/lat
            st_lon, st_lat,                                     # station WGS84 lon/lat
            d_euc_mat, self.max_dist,
            st_elev,                                            # DWH instrument altitude [m a.s.l.]
            self._dem_rgi, self._wgs84_to_lv95,
            n_samples=self.n_barrier_samples,
            elev_scale=self.elev_scale,
            elev_diff_scale=self.elev_diff_scale,
            n_barrier_width_samples=self.n_barrier_width_samples,
            barrier_width_m=self.barrier_width_m,
        )

        # Wrap in a DataArray so ned_interp can use named dims for the xarray operations.
        ned_sta_poi = xr.DataArray(
            dist_mat,
            dims=["poi", "sta"],
            coords={"poi": dom_idx, "sta": sta_ids},
        )

        # ── Topographic descriptors at POIs and stations ───────────────────
        # POI descriptors: direct lookup by ICON cell index.
        poi_topo = (
            self._ds_topo[self.topo_vars]
            .isel(cell=dom_idx)
            .rename({"cell": "poi"})
            .assign_coords({"poi": dom_idx})
        )
        # Station descriptors: snap each station to the nearest ICON cell using
        # the same LV95-km projection as grid_xy/sta_xy (precomputed once in
        # _load_dem()) so the distance metric is consistent.
        _, topo_gi = cKDTree(self._topo_xy_km).query(sta_xy, k=1)
        sta_topo = (
            self._ds_topo[self.topo_vars]
            .isel(cell=topo_gi)
            .rename({"cell": "sta"})
            .assign_coords({"sta": sta_ids})
        )

        # ── ned_interp ─────────────────────────────────────────────────────
        # Spreads residuals to POIs using IDW weighted by topographic similarity.
        # The max_dist cutoff here acts on barrier-aware distances, so a POI that is
        # geographically close but behind a ridge may receive zero correction.
        result = ned_interp(
            sta_res, ned_sta_poi,
            sta_topo=sta_topo, poi_topo=poi_topo,
            max_dist=self.max_dist,
            weight_power=self.weight_power,
            min_topo_w=self.min_topo_w,
            lim_effective=self.lim_effective,
        )

        # ── Linear taper ───────────────────────────────────────────────────
        # Fade the correction linearly from 1 (at the nearest station) to 0 (at max_dist).
        # The taper uses raw Euclidean distance — not barrier-inflated distance — because
        # we want the geographic fade-out to be independent of the barrier logic.
        dmin_poi, _ = cKDTree(sta_xy).query(poi_xy, k=1)
        taper = (1.0 - np.clip(dmin_poi / self.max_dist, 0.0, 1.0)).astype(np.float32)

        # POIs with no station within max_dist return NaN from ned_interp → no correction.
        correction = np.nan_to_num(result[shortname].values, nan=0.0) * taper

        # ── Apply correction ───────────────────────────────────────────────
        # corrected ≈ background − (background − obs) = obs near stations.
        # Only POIs within the domain are modified; the rest of the field is unchanged.
        corrected_flat = B_flat.copy()
        corrected_flat[dom_idx] -= correction

        LOG.info(
            "Nudged '%s': %d stations, %d POIs, max |correction| = %.4f",
            shortname, len(st_lat), n_poi, float(np.abs(correction).max()),
        )
        return corrected_flat

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
        return pd.read_parquet(self.obs_path)
