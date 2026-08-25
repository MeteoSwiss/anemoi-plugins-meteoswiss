"""
NudgeTowardObservation — v5 (distances in km; v4 adds per-station reliability-based
influence radius; v4.1 makes the linear taper per-station too — both ported
unchanged from notebooks/nudging_analysis_v10.ipynb; v5 stops computing
barrier-aware effective distances (d_eff) live and instead reads them from a
precomputed cache file — see *d_eff_file* and step 4 below)

Implements Interpolation of Residuals (IoR) using ned_interp combined with
barrier-aware effective distances (d_eff), precomputed offline from a
1 km-scale DEM and read from disk at construction time (*d_eff_file*) rather
than computed by this filter.

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
4.  Look up the barrier- and elevation-aware effective distance for each
    (POI, station) pair from the precomputed cache (*d_eff_file*) instead of
    computing it live:
        d_eff = sqrt(d_euc² + (barrier/elev_scale)² + (elev_diff/elev_diff_scale)²)
    where the barrier term comes from sampling the DEM along a perpendicular
    slab at each along-path step (Gaussian-weighted mean across the corridor
    width, 95th-percentile along the path for the effective ridge height) and
    elev_diff is the endpoint elevation gap — see barrier_distances() for the
    full derivation. This filter never calls barrier_distances() itself; that
    function is only used offline to build the cache (see
    notebooks/nudging_analysis_v11.ipynb, "Precomputed d_eff cache").
5.  Compute topographic similarity per (POI, station) pair (TPI, slope
    derivatives, DEM/ICON elevation): each descriptor's importance is
    |Pearson corr| between it and the station residuals, normalised to sum
    to 1 per variable — a descriptor uncorrelated with this variable's
    residuals contributes ~nothing, one that tracks the bias pattern
    dominates the blend.
6.  Spread station residuals to the POI grid via ned_interp: IDW
    (1 / d_eff^weight_power) weighted by the topographic similarity from
    step 5, floored at min_topo_w so nearby stations always contribute.
7.  Multiply each station's own normalised weight by a linear taper that
    fades to zero at THAT station's own max_dist (v4.1 — see the
    "Per-station linear taper" note in _nudge_field: each station may have
    its own reliability-scaled max_dist, so a single taper shared by every
    station would no longer coincide with each one's own cutoff).
8.  Subtract the resulting correction from the background field.

Weight computation, symbolically (per POI p, station s)
---------------------------------------------------------------------------
Step 1 — raw distance (km):     d_euc[p,s]
Step 2 — elevation-aware (km):  d_eff[p,s]     = lookup from precomputed cache (d_eff_file) → "ned_sta_poi"
Step 3 — descriptor importance: importance[d]  = |corr_s(residual[s], descriptor_d[s])| / Σ_d |corr_s(...)|
                                 (one weight per descriptor d, computed across stations s; recomputed per variable)
Step 4 — topo similarity:       w_topo[p,s]    = Σ_d importance[d] * (1 - |sta_topo[d,s] - poi_topo[d,p]|), floored at min_topo_w
Step 5 — combine:               raw_w[p,s]     = w_topo[p,s] * (1 / d_eff[p,s]^weight_power)   [NaN if d_eff ≥ max_dist[s]]
Step 6 — normalize per POI:     w_ned[p,s]     = raw_w[p,s] / (Σ_s raw_w[p,s] + lim_effective)
Step 7 — per-station taper:     taper[p,s]     = 1 - clip(d_euc[p,s] / max_dist[s], 0, 1)   (v4.1; applied
                                 AFTER Step 6's normalisation — see ned_interp's "v4.1" docstring
                                 note for why applying it before normalisation would not work)
Step 8 — interpolate:           correction[p] = Σ_s w_ned[p,s] * taper[p,s] * residual[s]
Step 9 — apply:                 background[p] -= correction[p]

Note taper (step 7) uses raw d_euc (km), not d_eff — the geographic fade-out is
deliberately independent of the barrier logic (see barrier_distances docstring).
max_dist[s] may be a single global value shared by every station (use_reliability_check=False)
or per-station (use_reliability_check=True; see "Station reliability" below) — the taper always
uses whichever max_dist[s] applies to that particular station.

Unit history: earlier versions expressed max_dist/elev_scale/elev_diff_scale in
projected degrees and m/° respectively (1 projected degree ≈ 111.32 km near
Swiss latitudes). max_dist's default below is the degree-tuned value converted
to km, so default behaviour is preserved rather than re-tuned; deployment
configs must supply a km value directly (see the YAML config for this filter).
elev_scale/elev_diff_scale are no longer parameters of this filter — they are
baked into the offline-built d_eff cache (see *d_eff_file*).

Station reliability (v4) — optional, ported unchanged from
notebooks/nudging_analysis_v10.ipynb
---------------------------------------------------------------------------
Before step 6 (spreading residuals via ned_interp), each station's own
influence radius may be scaled by a leave-one-out spatial-consistency check
(NudgeTowardObservation._compute_reliability), independent of any particular
POI:
  1. r_hat[s] = ned_interp(residuals of all OTHER stations j != s,
                            station<->station d_eff(s, j), sta_topo, sta_topo)
     i.e. the same barrier-aware distance + topographic-similarity weighting
     used for POIs, evaluated at station s's own location with s excluded
     from its own neighbour set (self-distance forced to +inf beforehand).
  2. e[s]      = r_at_st[s] - r_hat[s]         (self-reported minus neighbour consensus)
  3. u[s]      = (e[s] - median(e)) / (1.4826 * MAD(e))   (robust z-score;
     median/MAD instead of mean/std for BOTH the centre and the spread, so a
     few bad stations don't drag either one toward themselves)
  4. reliability[s] = clip(1 - (u[s]/number_of_std)^2, 0, None)^2   (Tukey
     biweight: 1 when u[s]=0, smoothly falling to 0 once |u[s]| >= number_of_std)
     A station with no neighbour at all within max_dist gets e[s]=NaN (step 1's
     ned_interp returns NaN, not 0, for a fully-masked POI) and is excluded from
     the median(e)/MAD(e) in step 3 — otherwise a single NaN would silently
     poison every OTHER station's u/reliability too (np.median is not NaN-safe).
     Such a station is left at reliability[s]=1 (full radius) since there is no
     neighbour evidence to judge it against.
  5. min_dist         = reliability_min_dist_frac * max_dist
     station_max_dist[s] = min_dist + (max_dist - min_dist) * reliability[s]
     used in place of the single global max_dist in ned_interp's Step 1 mask,
     for THIS station's column only (max_dist there may be a per-station array;
     ned_interp needs no change to support this — xarray broadcasts a (sta,)
     array against the (poi, sta) distance matrix automatically).

reliability[s]=1 (fully consistent with its neighbours) keeps station s's full
max_dist reach; reliability[s]=0 (>= number_of_std robust sigmas off) shrinks
it down to the configurable floor (reliability_min_dist_frac * max_dist),
never all the way to zero — every station keeps at least some very local
influence. Within whatever radius a station keeps, its weight is otherwise
undiminished (same IDW/topo formula as any other station); it is excluded
entirely beyond that radius. use_reliability_check=False reproduces the pre-v4
(v3) behaviour exactly (a single global max_dist shared by every station).

The leave-one-out neighbour search in step 1, and the domain buffer computed
in _nudge_field, always use the GLOBAL max_dist — never a per-station radius
— since reliability must be computed before any per-station radius can be
derived from it, and the domain buffer only needs to be large enough to
contain every station's largest possible reach. The final linear taper, by
contrast, is reliability-DEPENDENT as of v4.1 (see "Per-station linear
taper" in _nudge_field): each station's own taper fades out at its own
(possibly reliability-shrunk) max_dist, not a single radius shared by every
station.
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
# Precomputed barrier- and elevation-aware effective distances (d_eff_poi, d_eff_sta)
# for the full station catalog — see NudgeTowardObservation's d_eff_file parameter
# and notebooks/nudging_analysis_v11.ipynb, "Precomputed d_eff cache".
_DEFAULT_D_EFF_FILE = "/scratch/mch/llanzila/sruc/aux_files/d_eff_cache_v11.nc"
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

# _plot_reliability_diagnostic's colour-scale percentile for the residual and RMSE
# panels: the true max is dominated by a handful of outlier stations/cells, which
# stretches the colour scale so far that the actual spatial pattern (the whole
# point of these panels) becomes invisible. A high percentile instead saturates
# those outliers at the colour scale's edge rather than letting them set its
# range. Purely cosmetic — never affects the nudging correction itself.
_DIAG_COLORBAR_PERCENTILE = 90


# ── ned_interp (adapted from data4web_pipelines/utils.py) ─────────────────────
# Direct import is not possible because data4web_pipelines depends on
# internal MeteoSwiss libraries not available in the production environment.


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
    max_dist: Optional[float] = None,
    weight_power: float = 1,
    min_topo_w: float = 0.2,
    lim_effective: float = 0,
    taper: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Spread station residuals to POIs via IDW with optional topographic similarity.

    Source: data4web_pipelines/utils.py.

    Steps when topo descriptors are provided
    ----------------------------------------
    1.  Mask stations beyond max_dist: set their distance to NaN so their
        inverse-distance weight becomes NaN (effectively 0 after normalisation).
    2.  Separate normalisation: scale sta_topo and poi_topo to [0, 1] EACH using its
        own min/max (data4web_pipelines/utils.py::normalize) — not a combined min/max
        over both sets. The ~150 station values and the ~400 k POI values therefore
        each get stretched to fill their own [0, 1] scale independently; this is
        data4web's actual reference behaviour, kept here for exact equivalence.
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
    7.  Multiply by `taper` (if given) — see the v4.1 note below.

    v4.1: `taper`, if given, is a per-(poi, sta) DataArray in [0, 1] (e.g. a linear
    fade based on each station's own distance and its own max_dist — see
    NudgeTowardObservation._nudge_field's "Per-station linear taper" section)
    multiplied into the weights AFTER Step 6's normalisation, not merged into the
    raw weights beforehand. This ordering matters: if a POI has only one
    contributing station, that station's normalised weight is always exactly 1
    regardless of its raw (pre-normalisation) weight's magnitude — so a taper
    applied before Step 6 would be exactly cancelled out by the division and have
    no effect at all in that (common, e.g. an isolated low-reliability station)
    case. Applying it after Step 6 avoids this: it independently dampens each
    station's own contribution without disturbing the relative blend between
    multiple contributing stations at the same POI.
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
        # Step 2 — separate normalisation: each dataset scaled to [0, 1] using its
        # OWN min/max, exactly as data4web_pipelines/utils.py::ned_interp.
        sta_topo = _normalize(sta_topo)
        poi_topo = _normalize(poi_topo)

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

    # Step 7 (v4.1) — per-(poi, sta) taper, applied AFTER normalisation. See the
    # docstring note above for why applying it before Step 6 would not work.
    if taper is not None:
        w_ned = w_ned * taper

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
        and ``y`` in LV95 metres). Barrier-aware distances (d_eff) are no
        longer computed from this DEM at run time — see *d_eff_file* — so
        this is now used only as a fallback to sample station elevation when
        it is missing from the observations Parquet.
    d_eff_file : str
        Path to a precomputed NetCDF holding barrier- and elevation-aware
        effective distances (d_eff) for the full station catalog: variables
        ``d_eff_poi`` (dims ``poi``, ``sta`` — every ICON cell within reach
        of any station, to every station) and ``d_eff_sta`` (dims ``sta_i``,
        ``sta`` — station-to-station, for the reliability check's
        leave-one-out prediction; self-distances are +inf). Built offline
        once via ``barrier_distances()`` for the full station catalog and a
        domain buffered around all of them (see
        notebooks/nudging_analysis_v11.ipynb, "Precomputed d_eff cache") and
        reused across ref_times/deployments as long as the station catalog,
        DEM, ICON grid, and barrier hyperparameters it was built with are
        unchanged. This filter never calls ``barrier_distances()`` itself:
        each call's (POI, station) or (station, station) subset — which
        varies per variable since the set of stations with a non-NaN
        observation differs — is sliced directly from this cache. A POI or
        station needed by a call but not covered by the cache raises a
        ``ValueError`` rather than triggering a live recompute; rebuild the
        cache (with the current station catalog) if that happens.
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
        Default station influence radius in km, used for every nudged
        variable unless overridden per-variable via *variable_overrides*: the
        barrier-distance cutoff, the base for the domain buffer, and (when
        *use_reliability_check* is ``False``) the linear taper radius shared
        by every station. When *use_reliability_check* is ``True``, each
        station's own taper radius is instead its own reliability-scaled
        *max_dist* (see "Station reliability" and "Per-station linear taper"
        below) — this parameter remains the upper bound every station's
        radius scales down from (notebook v8: ``MAX_DIST_KM``; default here
        is the historical degree-tuned value (0.35°) converted to km, ≈
        38.96).
    variable_overrides : dict, optional
        Per-variable override of *d_eff_file* and/or *max_dist*, e.g.
        ``{"U_10M": {"d_eff_file": "...", "max_dist": 20.0}, "V_10M": {...}}``.
        Keys must be GRIB shortNames present in the active nudge set (see
        *nudge_variables*); each value is a dict that may set either or both
        of ``d_eff_file``/``max_dist`` — whichever is omitted falls back to
        this filter's top-level *d_eff_file*/*max_dist*. Motivation: the
        barrier-aware effective-distance model (see module docstring) assumes
        a spatially smooth, terrain-correlated bias, which holds for
        temperature-like variables but not for near-surface wind (U_10M/
        V_10M), whose bias is dominated by hyper-local siting/channeling
        effects with a much shorter decorrelation length — a wind-specific
        cache built with a smaller ``MAX_DIST_KM``/``ELEV_SCALE_KM`` (see
        notebooks/d_eff_generator.ipynb) can be supplied here without
        affecting the variables that use the default cache. This only
        changes *which* precomputed d_eff cache and radius feed into the
        (unchanged) IDW/topo-similarity/taper/reliability computation for
        that variable — every other variable keeps using the top-level
        defaults. Defaults to ``None`` (no overrides; every variable uses the
        top-level *d_eff_file*/*max_dist*, reproducing pre-v5.1 behaviour
        exactly).
    min_topo_w : float
        Minimum topographic similarity weight floor so nearby stations always
        contribute (notebook: ``MIN_TOPO_W = 0.2``).
    lim_effective : float
        Virtual zero-residual station weight added to the normalisation
        denominator; 0 = pure IDW (notebook: ``LIM_EFFECTIVE = 0.0``).
    use_reliability_check : bool
        If ``True``, scale each station's own influence radius by a
        leave-one-out spatial-consistency check (see module docstring,
        "Station reliability"), so non-representative stations (bad sensors,
        siting issues, local micro-effects the model can't resolve) reach a
        smaller area. Defaults to ``False``, reproducing the pre-v4 (v3)
        behaviour exactly (a single global *max_dist* shared by every
        station) — existing deployment configs are unaffected unless they
        explicitly opt in (notebook: ``USE_RELIABILITY_CHECK``, which
        defaults to ``True`` there since the notebook is the exploratory
        context this was validated in).
    number_of_std : float
        Tukey biweight rejection threshold, in robust (median/MAD-based)
        standard deviations: a station whose leave-one-out residual
        discrepancy is at or beyond this many robust sigmas from the
        network's typical discrepancy gets reliability=0 (its influence
        radius shrinks to the configured floor). Only used when
        *use_reliability_check* is ``True`` (notebook: ``RELIABILITY_REJECT_C``,
        default 4.0).
    reliability_min_dist_frac : float
        Minimum fraction of *max_dist* every station keeps as its influence
        radius, even at reliability=0 — the radius never shrinks all the way
        to zero, so every station retains at least some very local influence.
        Must be in [0, 1]. Only used when *use_reliability_check* is ``True``
        (notebook: ``RELIABILITY_MIN_DIST_FRAC``, default 0.1, i.e. 10%).
    reliability_eps : float
        Numerical floor on the robust (MAD-based) scale estimate used by the
        reliability check, avoiding division by zero when stations agree with
        their neighbours almost exactly (notebook: ``RELIABILITY_EPS``,
        default 1e-6). Only used when *use_reliability_check* is ``True``.
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
    plot_dir : str, optional
        If given, save a 5-panel reliability diagnostic PNG (station residual
        — holdin AND holdout stations, pre-nudging; station reliability
        — holdin only, since reliability is only defined for stations that
        participated in the nudging; gridded correction; pre-nudging station
        RMSE; post-nudging station RMSE, the last two split by holdin/holdout
        — the station-residual, reliability, and gridded-correction panels
        are the same figure produced interactively in
        notebooks/nudging_analysis_v10.ipynb; the two RMSE panels are
        production-only) for every nudged variable at every call, one file
        per (variable, ref_time):
        ``{plot_dir}/reliability_diag_{shortname}_{ref_time:%Y%m%d%H%M}.png``.
        Only has an effect when *use_reliability_check* is ``True`` — there is
        no reliability to plot otherwise. Defaults to ``None`` (no plotting).
        Plotting failures (including matplotlib/cartopy not being installed)
        are logged and skipped; they never affect the nudging correction
        itself, which is computed and returned identically either way.
    plot_extent : list of float, optional
        ``[lon_min, lon_max, lat_min, lat_max]`` map extent for the diagnostic
        plot. Defaults to ``[5.8, 10.8, 45.7, 47.9]`` (Switzerland). Only used
        when *plot_dir* is set.
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
        d_eff_file: str = _DEFAULT_D_EFF_FILE,
        icon_orog_file: str = _DEFAULT_ICON_OROG_FILE,
        weight_power: float = 4.0,
        max_dist: float = 38.962,
        variable_overrides: Optional[dict] = None,
        min_topo_w: float = 0.2,
        lim_effective: float = 0.0,
        use_reliability_check: bool = False,
        number_of_std: float = 4.0,
        reliability_min_dist_frac: float = 0.1,
        reliability_eps: float = 1e-6,
        plot_dir: Optional[str] = None,
        plot_extent: Optional[list] = None,
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
        if not (0.0 <= reliability_min_dist_frac <= 1.0):
            raise ValueError(
                f"reliability_min_dist_frac must be in [0, 1], got {reliability_min_dist_frac!r}"
            )
        if number_of_std <= 0:
            raise ValueError(f"number_of_std must be > 0, got {number_of_std!r}")

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
        self.d_eff_file = Path(d_eff_file)
        self.icon_orog_file = Path(icon_orog_file)
        self.weight_power = weight_power
        self.max_dist = max_dist
        self.min_topo_w = min_topo_w
        self.lim_effective = lim_effective
        self.use_reliability_check = use_reliability_check
        self.number_of_std = number_of_std
        self.reliability_min_dist_frac = reliability_min_dist_frac
        self.reliability_eps = reliability_eps
        self.plot_dir = Path(plot_dir) if plot_dir is not None else None
        self.plot_extent = list(plot_extent) if plot_extent is not None else [5.8, 10.8, 45.7, 47.9]
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
        # param -> dict of diagnostic arrays from the last _compute_reliability() call,
        # consumed by _plot_reliability_diagnostic(). Mirrors RELIABILITY_DIAG in
        # notebooks/nudging_analysis_v10.ipynb.
        self._reliability_diag = {}

        if self.plot_dir is not None:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            if not self.use_reliability_check:
                LOG.warning(
                    "plot_dir=%s is set but use_reliability_check=False — there is no "
                    "reliability to plot, so no diagnostic plots will be produced.",
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
        # Every variable defaults to the top-level d_eff_file/max_dist; an
        # entry in variable_overrides replaces one or both for that variable
        # only. This only selects *which* precomputed cache and radius feed
        # into _nudge_field()/_compute_reliability() for a given variable —
        # the IDW/topo-similarity/taper/reliability computation itself is
        # unchanged and unaware that other variables might use a different
        # cache/radius.
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

        self._max_dist_by_var = {
            var: self.variable_overrides.get(var, {}).get("max_dist", self.max_dist)
            for var in self.param_map
        }
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
            "weight_power=%.1f, d_eff_file=%s, "
            "lapse_rate=%.5f K/m (vars=%s), use_reliability_check=%s, number_of_std=%.2f, "
            "reliability_min_dist_frac=%.3f, min radius=%s km",
            list(self.param_map.keys()),
            {v: self._max_dist_by_var[v] for v in self.param_map},
            self.weight_power,
            {v: str(self._d_eff_file_by_var[v]) for v in self.param_map},
            self.lapse_rate,
            sorted(self.lapse_rate_vars),
            self.use_reliability_check,
            self.number_of_std,
            self.reliability_min_dist_frac,
            {v: round(self.reliability_min_dist_frac * self._max_dist_by_var[v], 2) for v in self.param_map},
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

    def _load_d_eff_cache(self, path: Path) -> dict:
        """Load one precomputed barrier- and elevation-aware effective-distance
        (d_eff) cache file for the full station catalog.

        This filter never calls ``barrier_distances()`` itself — see
        _get_d_eff_poi()/_get_d_eff_sta() below, used by _nudge_field() and
        _compute_reliability() instead. Each cache is built offline once (see
        notebooks/nudging_analysis_v11.ipynb, "Precomputed d_eff cache") for
        the full station catalog and a domain buffered around all of them;
        every call's own (usually smaller, since stations with a NaN
        observation for that variable are excluded) POI/station subset is
        sliced from it.

        Called once per distinct path in *_d_eff_file_by_var* (see __init__):
        several variables may share the same cache file (e.g. T_2M/TD_2M),
        in which case it is only loaded once and reused.
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
        return cache["d_eff_poi_full"].sel(poi=list(dom_idx), sta=list(sta_ids))

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

        # Reference time: the time step to which nudging is applied.
        # devt: first field's valid_time (useful for single-step development runs).
        # depl: earliest valid_time across the fieldlist (the true initial condition).
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
        # Stations removed by _apply_holdout, kept around only for the optional
        # post-nudging error panel (_plot_reliability_diagnostic) — they never
        # participate in the nudging correction itself.
        held_out_stations = all_stations.drop(index=stations.index)

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
        ref_time=None,
        held_out_stations: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Apply v3 IoR nudging to a single field; return the corrected 1-D array.

        ``ref_time`` and ``held_out_stations`` are only used for the optional
        reliability diagnostic plot (see *plot_dir*); neither plays any role
        in the nudging computation itself and both may be omitted when not
        plotting.
        """
        # Background field values on the full ICON grid, shape (n_cells,).
        B_flat = np.asarray(field.values, dtype=float).ravel()

        # This variable's own max_dist (see *variable_overrides*): defaults to
        # self.max_dist, overridden per-variable when configured. Used below
        # exactly where a single shared self.max_dist used to be — the domain
        # buffer, the barrier-distance/ned_interp cutoff, and (together with
        # its own d_eff cache, see _get_d_eff_poi) the station-reliability
        # radius scaling.
        max_dist = self._max_dist_by_var[shortname]

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
        sta_x_min = sta_xy[:, 0].min() - max_dist
        sta_x_max = sta_xy[:, 0].max() + max_dist
        sta_y_min = sta_xy[:, 1].min() - max_dist
        sta_y_max = sta_xy[:, 1].max() + max_dist
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
        # Read from the precomputed cache instead of computing them live —
        # see _load_d_eff/_get_d_eff_poi. dom_idx/sta_ids for THIS call are
        # always a subset of the full catalog/domain the cache was built for
        # (see _get_d_eff_poi's coverage check).
        ned_sta_poi = self._get_d_eff_poi(shortname, dom_idx, sta_ids)

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

        # ── Station reliability → per-station influence radius (v4) ────────
        # A fully reliable station (reliability=1) keeps the full max_dist
        # (this variable's own — see *variable_overrides* at the top of this
        # method) reach; a station near reliability=0 shrinks down to a
        # configurable floor (self.reliability_min_dist_frac * max_dist)
        # rather than all the way to zero — every station keeps at least some
        # very local influence. Reliability scales the radius LINEARLY
        # between that floor and the full max_dist. Within whatever radius a
        # station keeps, its weight is otherwise undiminished (same IDW/topo
        # formula as any other station); it is excluded entirely beyond that
        # radius. See the module docstring ("Station reliability") for the
        # full algorithm, ported unchanged from
        # notebooks/nudging_analysis_v10.ipynb.
        # use_reliability_check=False reproduces the pre-v4 (v3) behaviour
        # exactly (a single global max_dist shared by every station).
        if self.use_reliability_check:
            reliability = self._compute_reliability(
                shortname, sta_ids, st_lon, st_lat, r_at_st, sta_topo, max_dist,
            )
            min_dist = self.reliability_min_dist_frac * max_dist
            station_max_dist = min_dist + (max_dist - min_dist) * reliability
        else:
            station_max_dist = max_dist

        # ── Per-station linear taper (v4.1) ─────────────────────────────────
        # v3/early-v4 used ONE taper per POI (distance to the nearest station
        # overall, fading out at the single global max_dist) — that coincided
        # exactly with ned_interp's hard cutoff back when every station shared
        # the same max_dist. Once each station got its OWN (possibly much
        # smaller) max_dist above (station reliability), a single global-radius
        # taper no longer coincides with THAT station's own cutoff: its
        # contribution jumped straight from ~full strength to exactly zero right
        # at its own (small) radius — a sharp-edged "blob" instead of a smooth
        # fade. Fix: one taper factor per (POI, station) pair, using RAW
        # Euclidean distance (d_euc_mat — not barrier-inflated d_eff, same
        # "independent of barrier logic" principle as the original taper) and
        # THAT station's own station_max_dist. Passed into ned_interp, which
        # applies it AFTER weight normalisation (see ned_interp's "v4.1"
        # docstring note for why applying it before normalisation would not work).
        d_euc_da = xr.DataArray(
            d_euc_mat, dims=["poi", "sta"], coords={"poi": dom_idx, "sta": sta_ids}
        )
        pair_taper = (1.0 - (d_euc_da / station_max_dist).clip(min=0.0, max=1.0)).astype(np.float32)

        # ── ned_interp ─────────────────────────────────────────────────────
        # Spreads residuals to POIs using IDW weighted by topographic similarity.
        # The max_dist cutoff here acts on barrier-aware distances, so a POI that is
        # geographically close but behind a ridge may receive zero correction.
        result = ned_interp(
            sta_res, ned_sta_poi,
            sta_topo=sta_topo, poi_topo=poi_topo,
            max_dist=station_max_dist,
            weight_power=self.weight_power,
            min_topo_w=self.min_topo_w,
            lim_effective=self.lim_effective,
            taper=pair_taper,
        )

        # POIs with no station within their radius return NaN from ned_interp → no correction.
        correction = np.nan_to_num(result[shortname].values, nan=0.0)

        # ── Apply correction ───────────────────────────────────────────────
        # corrected ≈ background − (background − obs) = obs near stations.
        # Only POIs within the domain are modified; the rest of the field is unchanged.
        corrected_flat = B_flat.copy()
        corrected_flat[dom_idx] -= correction

        # ── Pre/post-nudging station error (holdin + holdout), for the
        # diagnostic plot's station-residual and RMSE panels only — never
        # feeds back into the correction above. Holdin stations (gi/
        # st_obs_lr/st_lat/st_lon already computed above) validate the fit at
        # stations that DID influence the correction; holdout stations
        # (excluded by _apply_holdout, passed in separately since `stations`
        # no longer contains them) give an independent check at locations the
        # correction never saw.
        if self.plot_dir is not None and self.use_reliability_check:
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

                    if "elevation" in held_out_stations.columns:
                        ho_elev = held_out_stations.loc[ho_valid, "elevation"].to_numpy(dtype=float)
                        nan_mask = np.isnan(ho_elev)
                        if nan_mask.any():
                            hx, hy = self._wgs84_to_lv95.transform(ho_lon[nan_mask], ho_lat[nan_mask])
                            ho_elev[nan_mask] = self._dem_rgi(np.c_[hy, hx])
                    else:
                        hx, hy = self._wgs84_to_lv95.transform(ho_lon, ho_lat)
                        ho_elev = self._dem_rgi(np.c_[hy, hx])

                    ho_x, ho_y = self._wgs84_to_lv95.transform(ho_lon, ho_lat)
                    ho_xy = np.c_[ho_x, ho_y] / 1000.0
                    _, ho_gi = cKDTree(grid_xy).query(ho_xy, k=1)

                    ho_obs_lr = ho_obs
                    if shortname in self.lapse_rate_vars:
                        elev_model_at_ho = self._ds_topo["ICON_OROG"].values[ho_gi]
                        ho_obs_lr = ho_obs - self.lapse_rate * (elev_model_at_ho - ho_elev)

                    err_lat.append(ho_lat)
                    err_lon.append(ho_lon)
                    err_val_pre.append(B_flat[ho_gi] - ho_obs_lr)
                    err_val_post.append(corrected_flat[ho_gi] - ho_obs_lr)
                    err_is_holdout.append(np.ones(len(ho_lat), dtype=bool))

            # Squared-error rooted per station: a single ref_time gives a single
            # sample per station, so this is algebraically just |error| — kept as
            # an explicit RMS formula in case this is ever extended to aggregate
            # multiple ref_times/variables per station.
            residual_pre_nudge = np.concatenate(err_val_pre)

            # Stations CleanObservation's QC dropped for this variable (see its
            # "{col}_qc_dropped" flag column — set for both the physical-bounds
            # and background-check tiers, never for a holdout/excluded station;
            # see clean_observation.py's "Holdout protection"). `stations` here
            # is post-_apply_holdout but QC only ever nulls a value, never a
            # whole row, so lat/lon for a QC-dropped station are still present.
            # Absent column (e.g. an older cleaned Parquet, or no field found
            # for the background check) => no QC-dropped stations to show.
            qc_flag_col = f"{col}_qc_dropped"
            if qc_flag_col in stations.columns:
                qc_mask = stations[qc_flag_col].fillna(False).astype(bool)
            else:
                qc_mask = pd.Series(False, index=stations.index)

            self._reliability_diag.setdefault(shortname, {}).update({
                "err_latitude": np.concatenate(err_lat),
                "err_longitude": np.concatenate(err_lon),
                # Signed pre-nudging residual (background − obs) at every station,
                # holdin and holdout alike — feeds the "all stations" residual panel.
                "residual_pre_nudge": residual_pre_nudge,
                "pre_nudge_rmse": np.sqrt(residual_pre_nudge ** 2),
                "post_nudge_rmse": np.sqrt(np.concatenate(err_val_post) ** 2),
                "err_is_holdout": np.concatenate(err_is_holdout),
                "qc_dropped_latitude": stations.loc[qc_mask, "latitude"].to_numpy(),
                "qc_dropped_longitude": stations.loc[qc_mask, "longitude"].to_numpy(),
                "qc_dropped_ids": stations.index[qc_mask].tolist(),
            })

        if self.use_reliability_check:
            n_shrunk = int((station_max_dist < max_dist).sum())
            LOG.info(
                "Nudged '%s': %d stations, %d POIs, max |correction| = %.4f "
                "(reliability check: number_of_std=%.2f, %d/%d station(s) with a shrunk radius)",
                shortname, len(st_lat), n_poi, float(np.abs(correction).max()),
                self.number_of_std, n_shrunk, len(st_lat),
            )
        else:
            LOG.info(
                "Nudged '%s': %d stations, %d POIs, max |correction| = %.4f "
                "(reliability check: disabled)",
                shortname, len(st_lat), n_poi, float(np.abs(correction).max()),
            )

        # Diagnostic plot (opt-in, never affects the correction above — see
        # _plot_reliability_diagnostic's own try/except).
        if self.plot_dir is not None and self.use_reliability_check:
            self._plot_reliability_diagnostic(shortname, ref_time, B_flat, corrected_flat)

        return corrected_flat

    def _compute_reliability(
        self,
        shortname: str,
        sta_ids: list,
        st_lon: np.ndarray,
        st_lat: np.ndarray,
        r_at_st: np.ndarray,
        sta_topo: xr.Dataset,
        max_dist: float,
    ) -> xr.DataArray:
        """Leave-one-out spatial-consistency check (v4; ported unchanged from
        notebooks/nudging_analysis_v10.ipynb — see module docstring, "Station
        reliability", for the full algorithm description).

        Down-weights (via a shrunk influence radius, applied by the caller)
        stations whose residual disagrees with what their own neighbours would
        predict for them, independent of distance/topo similarity to any
        particular POI.

        Reuses ``st_lon``/``st_lat``/``r_at_st``/``sta_topo``/``max_dist``
        already computed by ``_nudge_field`` for the exact same station set
        and variable, rather than recomputing them — a pure
        implementation-level deduplication; it does not change any numerical
        result, since these arrays are deterministic given the same station
        DataFrame and field. ``max_dist`` is this variable's own radius (see
        *variable_overrides*), used here as the single global radius the
        module docstring's "Station reliability" section refers to — still
        global across stations, just no longer necessarily the same value
        for every nudged variable.

        Returns an xr.DataArray (dims=["sta"]) of reliability in [0, 1],
        aligned with ``sta_ids``.
        """
        n_sta = len(sta_ids)

        sta_res = xr.Dataset(
            {shortname: xr.DataArray(
                r_at_st.astype(np.float32), dims=["sta"], coords={"sta": sta_ids}
            )}
        )

        # ── Station <-> station barrier-aware distances ────────────────────
        # Read from the precomputed cache instead of computing them live —
        # see _load_d_eff_cache/_get_d_eff_sta. The diagonal (station vs.
        # itself) is already +inf in the cache (see the offline
        # cache-building step), which ned_interp's max_dist cutoff then masks
        # out, guaranteeing station s can never be its own neighbour. Always
        # uses this variable's own GLOBAL max_dist (not any per-station
        # radius) below — reliability itself has to be computed before any
        # per-station radius can be derived from it.
        ned_ss = self._get_d_eff_sta(shortname, sta_ids)

        # Stations stand in as both "sta" and "poi" for the leave-one-out prediction.
        poi_topo = sta_topo.rename({"sta": "poi"}).assign_coords({"poi": sta_ids})

        # ── Leave-one-out neighbour prediction ──────────────────────────────
        # No reliability weighting here: the check itself must trust all OTHER
        # stations equally on this first pass, otherwise a bad station could
        # suppress the very signal that would flag it.
        r_hat = ned_interp(
            sta_res, ned_ss,
            sta_topo=sta_topo, poi_topo=poi_topo,
            max_dist=max_dist, weight_power=self.weight_power,
            min_topo_w=self.min_topo_w, lim_effective=self.lim_effective,
        )
        r_hat_at_st = r_hat[shortname].rename({"poi": "sta"}).sel(sta=sta_ids).values

        # ── Discrepancy → robust z-score → Tukey biweight reliability ──────
        # Robust z-score: recentre by median(e), not just rescale by MAD(e) —
        # otherwise a systematic offset in e (e.g. leave-one-out spatial
        # smoothing tends to under-predict local extremes even for perfectly
        # good stations) would shift every station's u by the same amount
        # instead of being absorbed into the "typical" reference point.
        e = r_at_st - r_hat_at_st

        # A station with no OTHER station within max_dist gets r_hat=NaN from
        # ned_interp's leave-one-out call (its min_count=1 makes a fully-masked
        # POI return NaN rather than 0) — so e is NaN for that station too. With
        # no neighbours to compare it against, there's no basis to judge it, and
        # np.median/np.abs(...).median() are NOT NaN-safe: a single NaN would
        # otherwise silently turn med_e/mad/scale into NaN, which would then
        # cascade into every OTHER station's u and reliability as well —
        # collapsing the whole station set to reliability=NaN → station_max_dist
        # =NaN → ned_interp masks every pair against it (`dist < NaN` is always
        # False) → zero correction everywhere. Guard against this explicitly:
        # exclude non-finite e from the median/MAD/scale computation, and leave
        # an isolated station's own reliability at 1.0 (full trust, full
        # radius) rather than penalising it for something it can't be judged on.
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

        # Stashed for _plot_reliability_diagnostic(); mirrors RELIABILITY_DIAG in
        # notebooks/nudging_analysis_v10.ipynb. Purely a side channel for the
        # optional plot — does not feed back into the reliability computation.
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
        station residual (holdin AND holdout stations, pre-nudging), station
        reliability (holdin only — reliability is only defined for stations
        that participated in the nudging), gridded correction (these three
        match notebooks/nudging_analysis_v10.ipynb's diagnostic cell), two
        production-only panels of pre- and post-nudging station RMSE (holdin
        circles vs. holdout triangles — see the "Pre/post-nudging station
        error" block in ``_nudge_field``), and a sixth panel marking which
        stations CleanObservation's QC removed for this variable (see
        ``qc_dropped_*`` in that same block) — always empty for a
        holdout/excluded station, by construction (see
        clean_observation.py's "Holdout protection").

        Opt-in via *plot_dir*; a no-op if it is ``None`` or if
        ``_compute_reliability`` wasn't called for ``shortname`` this call
        (i.e. *use_reliability_check* is ``False``). Never raises: any failure
        (missing matplotlib/cartopy, bad data, ...) is logged and swallowed so
        it can never affect the nudging correction, which has already been
        computed and returned by the time this runs.
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

            # Shared colour scale for the two residual panels, so they're directly
            # comparable — same convention as the notebook's diagnostic cell.
            # Uses the _DIAG_COLORBAR_PERCENTILE-th percentile of each source's
            # magnitude rather than its true max (see that constant's docstring):
            # a few outlier stations/cells no longer stretch the whole scale.
            # Percentile is taken separately per source (station vs. gridded),
            # then combined via max — mirroring the previous max-of-two-maxes
            # structure — so the (usually far more numerous) gridded points don't
            # dilute the station outliers' influence on the scale, or vice versa.
            res_abs = max(
                float(np.percentile(np.abs(station_res), _DIAG_COLORBAR_PERCENTILE)) if len(station_res) else 0.0,
                float(np.percentile(np.abs(res_ch), _DIAG_COLORBAR_PERCENTILE)) if len(res_ch) else 0.0,
            ) or 1.0  # guard against an all-zero degenerate case
            vmin, vmax = -res_abs, res_abs

            # Shared colour scale for the two RMSE panels, so pre- and post-nudging
            # station error are directly comparable. Same percentile-based
            # rationale as the residual panels above.
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
                # Colorbar exactly as tall as `ax` — plt.colorbar's fraction/pad
                # doesn't track a cartopy GeoAxes' actual rendered aspect
                # (distorted by set_extent).
                cax = make_axes_locatable(ax).append_axes(
                    "right", size="5%", pad=0.3, axes_class=plt.Axes
                )
                return fig.colorbar(mappable, cax=cax, label=label)

            def _holdin_holdout_scatter(ax, lon, lat, values, s=45, **kwargs):
                # Shared holdin (circle) / holdout (triangle, 2x the marker size)
                # scatter used by the station-residual and RMSE panels.
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
            # levels must be an explicit array spanning [vmin, vmax] — a bare
            # integer count would make tricontourf compute level *positions*
            # from res_ch's actual data range instead of from vmin/vmax.
            # extend="both": values outside [vmin, vmax] get clamped to the
            # colormap's over/under color instead of being left unfilled —
            # without it, tricontourf has no bin for out-of-range points and
            # silently skips painting them, leaving gaps that show the bare
            # map background through (easily mistaken for a genuine near-zero
            # residual when it's actually the opposite: the most extreme
            # values in the field).
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
            # "Kept" here means every station that passed QC and reported this
            # variable (diag["err_*"] — same population as ax0/ax3/ax4); a
            # QC-dropped station has NaN for this column by the time _nudge_field
            # builds err_latitude/err_longitude, so the two sets are disjoint.
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
