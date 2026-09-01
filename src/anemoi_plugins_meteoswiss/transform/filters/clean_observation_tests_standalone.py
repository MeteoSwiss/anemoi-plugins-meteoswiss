"""
Standalone Python implementations of titanlib QC algorithms.

Algorithms: isolation_check, buddy_check, fgt (first-guess test),
            sct_resistant (OI-based SCT), sct_dual (binary SCT)
Reference:  https://github.com/metno/titanlib/tree/master/src

All three are implemented from scratch in NumPy/SciPy, matching the C++
logic as closely as possible.  High-level wrappers share the same call
signature as the functions in clean_observation_tests.py, so they can be
swapped in for direct comparison.

Comparison routines (Python vs titanlib) are in clean_observation_compare_tests.py.

BackgroundType enum (mirrors titanlib):
    VERTICAL_PROFILE            = 0   (Nelder-Mead piecewise lapse rate)
    VERTICAL_PROFILE_THEIL_SEN  = 1   (Theil-Sen lapse rate)
    MEAN_OUTER_CIRCLE           = 2
    MEDIAN_OUTER_CIRCLE         = 3
    EXTERNAL                    = 4   (use provided background_values)

ConditionType enum (mirrors titanlib, used by sct_dual):
    CONDITION_EQ  = 0   (value == event_threshold)
    CONDITION_GT  = 1   (value >  event_threshold)
    CONDITION_GEQ = 2   (value >= event_threshold)
    CONDITION_LT  = 3   (value <  event_threshold)
    CONDITION_LEQ = 4   (value <= event_threshold)

Flag values returned by fgt_raw / sct_resistant_raw (mirrors titanlib):
    -999  not checked
       0  passed (good)
       1  failed (bad)
      11  isolated – too few inner neighbours
      12  isolated – too few outer neighbours
     100  matrix inversion failure (unused here, kept for compatibility)
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree  # haversine queries via ECEF unit-sphere projection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M: float = 6_371_000.0
NA: float = -999.0  # "not available" sentinel used by titanlib

# BackgroundType enum values
VERTICAL_PROFILE = 0
VERTICAL_PROFILE_THEIL_SEN = 1
MEAN_OUTER_CIRCLE = 2
MEDIAN_OUTER_CIRCLE = 3
EXTERNAL = 4

# ConditionType enum values (for sct_dual)
CONDITION_EQ  = 0
CONDITION_GT  = 1
CONDITION_GEQ = 2
CONDITION_LT  = 3
CONDITION_LEQ = 4

# ---------------------------------------------------------------------------
# Spatial helpers — KDTree on ECEF unit-sphere Cartesian coords
#
# Strategy: project (lat, lon) → (x, y, z) on the unit sphere.
# Euclidean chord distance d_3d and great-circle arc distance d_arc satisfy:
#   d_3d  = 2 * sin(d_arc / (2 * R))
#   d_arc = 2 * R * arcsin(d_3d / 2)
# For a query radius r (metres): search with d_3d_max = 2 * sin(r / (2*R)).
# ---------------------------------------------------------------------------

def _latlon_to_xyz(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat_r = np.deg2rad(lats)
    lon_r = np.deg2rad(lons)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def _build_tree(lats: np.ndarray, lons: np.ndarray) -> KDTree:
    return KDTree(_latlon_to_xyz(lats, lons))


def _query_radius(
    tree: KDTree,
    lat: float,
    lon: float,
    radius_m: float,
    include_self: bool = True,
) -> tuple[list[int], list[float]]:
    """Return (indices, haversine_distances_m) sorted by distance, within radius_m."""
    r_chord = 2.0 * np.sin(radius_m / (2.0 * EARTH_RADIUS_M))
    pt = _latlon_to_xyz(np.array([lat]), np.array([lon]))
    raw_idx = tree.query_ball_point(pt[0], r_chord)
    if not raw_idx:
        return [], []
    raw_idx = list(raw_idx)
    # Recover arc distances in metres
    chord = np.linalg.norm(tree.data[raw_idx] - pt[0], axis=1)
    chord = np.clip(chord, 0.0, 2.0)
    arc_m = (2.0 * EARTH_RADIUS_M * np.arcsin(chord / 2.0)).tolist()
    # Sort by distance
    order = np.argsort(arc_m)
    idx  = [raw_idx[k] for k in order]
    dist = [arc_m[k]   for k in order]
    if not include_self:
        pairs = [(i, d) for i, d in zip(idx, dist) if d > 1.0]
        idx  = [p[0] for p in pairs]
        dist = [p[1] for p in pairs]
    return idx, dist

# ---------------------------------------------------------------------------
# isolation_check
# ---------------------------------------------------------------------------

def isolation_check_raw(
    lats: np.ndarray,
    lons: np.ndarray,
    elevs: np.ndarray,
    num_min: int,
    radius: float,
    vertical_radius: float | None = None,
) -> np.ndarray:
    """
    Pure-Python isolation check (mirrors titanlib.isolation_check).

    Returns integer flag array: 0 = pass, 1 = isolated.
    vertical_radius: if given, neighbour must also be within this elevation
    difference.  Mirrors the per-station vector overload.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    n = len(lats)
    flags = np.zeros(n, dtype=int)
    tree = _build_tree(lats, lons)
    use_vert = vertical_radius is not None and np.isfinite(vertical_radius)

    for i in range(n):
        if not (np.isfinite(lats[i]) and np.isfinite(lons[i])):
            flags[i] = 1
            continue
        if use_vert and not np.isfinite(elevs[i]):
            flags[i] = 1
            continue

        idx, _ = _query_radius(tree, lats[i], lons[i], radius, include_self=False)

        if use_vert:
            count = sum(
                1 for j in idx
                if np.isfinite(elevs[j]) and abs(elevs[j] - elevs[i]) <= vertical_radius
            )
        else:
            count = len(idx)

        if count < num_min:
            flags[i] = 1

    return flags


def isolation_check_py(
    stations,
    lats,
    lons,
    elevs,
    para: str,
    date: str,
    num_min: int = 2,
    radius: float = 50_000,
    vertical_radius: float | None = None,
) -> dict:
    """
    Isolation check – same interface as clean_observation_tests.isolation_check.
    Returns dict {'Station':…, 'Time':…, 'Parameter':…} or {} if nothing flagged.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    flags = isolation_check_raw(lats, lons, elevs, num_min, radius, vertical_radius)
    my_dict: dict = {"Station": [], "Time": [], "Parameter": []}
    for idx in np.nonzero(flags)[0]:
        my_dict["Station"].append(stations[idx])
        my_dict["Time"].append(date)
        my_dict["Parameter"].append(para)
    return my_dict if my_dict["Station"] else {}

# ---------------------------------------------------------------------------
# buddy_check
# ---------------------------------------------------------------------------

def buddy_check_raw(
    lats: np.ndarray,
    lons: np.ndarray,
    elevs: np.ndarray,
    values: np.ndarray,
    radius_arr: np.ndarray,
    num_min_arr: np.ndarray,
    threshold: float,
    max_elev_diff: float = 200.0,
    elev_gradient: float = -0.0065,
    min_std: float = 1.0,
    num_iterations: int = 5,
    obs_to_check: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pure-Python buddy check (mirrors titanlib.buddy_check).

    radius_arr / num_min_arr: scalar or per-station arrays (broadcast to n).
    Returns integer flag array: 0 = pass, 1 = flagged.

    Note: variance is computed as population variance (ddof=0), matching
    boost::accumulators::variance.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(lats)

    radius_arr = np.broadcast_to(np.asarray(radius_arr, dtype=float), n).copy()
    num_min_arr = np.broadcast_to(np.asarray(num_min_arr, dtype=int), n).copy()

    if obs_to_check is None or len(obs_to_check) == 0:
        check_all = True
        _obs_to_check = np.ones(n, dtype=int)
    else:
        check_all = len(obs_to_check) != n
        _obs_to_check = np.asarray(obs_to_check, dtype=int)

    flags = np.zeros(n, dtype=int)
    for i in range(n):
        if not np.isfinite(values[i]):
            flags[i] = 1

    tree = _build_tree(lats, lons)
    num_removed_last = 0

    for _ in range(num_iterations):
        flags_prev = flags.copy()
        new_flags = flags_prev.copy()

        for i in range(n):
            if flags_prev[i] != 0:
                continue
            if not (check_all or _obs_to_check[i] == 1):
                continue

            idx, _ = _query_radius(
                tree, lats[i], lons[i], radius_arr[i], include_self=False
            )

            buddies: list[float] = []
            for j in idx:
                if flags_prev[j] != 0:
                    continue
                if max_elev_diff > 0:
                    if abs(elevs[j] - elevs[i]) <= max_elev_diff:
                        adjusted = values[j] + (elevs[i] - elevs[j]) * elev_gradient
                        buddies.append(adjusted)
                else:
                    buddies.append(float(values[j]))

            n_b = len(buddies)
            if n_b >= num_min_arr[i]:
                arr = np.array(buddies)
                mean = arr.mean()
                var = arr.var()  # population variance – matches C++ accumulator
                std_adj = np.sqrt(var + var / n_b)
                if std_adj < min_std:
                    std_adj = min_std
                pog = abs(values[i] - mean) / std_adj
                if pog > threshold:
                    new_flags[i] = 1

        flags = new_flags
        num_removed = int(np.sum(flags != 0))
        delta = num_removed - num_removed_last
        if delta == 0:
            break
        num_removed_last = delta  # titanlib resets to curr-iter delta, not cumulative

    return flags


def buddy_check_py(
    stations,
    lats,
    lons,
    elevs,
    values,
    para: str,
    date: str,
    threshold=4,
    max_elev_diff: float = 200,
    elev_gradient: float = -0.0065,
    min_std: float = 1,
    num_iterations: int = 5,
    num_min: int = 3,
    radius: float = 30_000,
) -> dict:
    """
    Buddy check – same interface as clean_observation_tests.buddy_check.
    threshold may be scalar or [smn_thr, other_thr].
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    snames = np.asarray(stations)
    n = len(lats)

    thresholds = threshold if isinstance(threshold, list) else [threshold, threshold]
    thr_short = float(thresholds[0])
    thr_long = float(thresholds[1] if len(thresholds) > 1 else thresholds[0])
    short_mask = np.array([len(s) == 3 for s in snames])

    radius_arr = np.full(n, radius, dtype=float)
    num_min_arr = np.full(n, num_min, dtype=int)
    flagged: set[int] = set()

    for thr, mask in [(thr_short, short_mask), (thr_long, ~short_mask)]:
        f = buddy_check_raw(
            lats, lons, elevs, values, radius_arr, num_min_arr, thr,
            max_elev_diff, elev_gradient, min_std, num_iterations,
        )
        for idx in np.nonzero(f)[0]:
            if mask[idx]:
                flagged.add(int(idx))

    my_dict: dict = {"Station": [], "Time": [], "Parameter": []}
    for idx in sorted(flagged):
        if np.isfinite(values[idx]):
            my_dict["Station"].append(snames[idx])
            my_dict["Time"].append(date)
            my_dict["Parameter"].append(para)
    return my_dict if my_dict["Station"] else {}

# ---------------------------------------------------------------------------
# Background computation
# ---------------------------------------------------------------------------


def _basic_vp(elevs: np.ndarray, t0: float, gamma: float) -> np.ndarray:
    return t0 + gamma * elevs


def _full_vp(
    elevs: np.ndarray, t0: float, gamma: float, a: float, h0: float, h1i: float
) -> np.ndarray:
    h1 = h0 + abs(h1i)
    e = np.asarray(elevs, dtype=float)
    out = np.empty_like(e)
    lo = e <= h0
    hi = e >= h1
    mi = ~lo & ~hi
    out[lo] = t0 + gamma * e[lo] - a
    out[hi] = t0 + gamma * e[hi]
    out[mi] = (
        t0 + gamma * e[mi]
        - (a / 2) * (1 + np.cos(np.pi * (e[mi] - h0) / (h1 - h0)))
    )
    return out


def _compute_vp(
    elevs: np.ndarray,
    oelevs: np.ndarray,
    values: np.ndarray,
    num_min_prof: int,
    min_elev_diff: float,
) -> np.ndarray:
    """
    Vertical profile via Nelder-Mead optimisation (titanlib VerticalProfile=0).
    Basic (2-param) model when n < num_min_prof or elevation spread too small.
    Full (5-param piecewise) model otherwise.
    """
    elevs = np.asarray(elevs, dtype=float)
    oelevs = np.asarray(oelevs, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(elevs)
    mean_v = float(values.mean())

    if elevs.min() == elevs.max():
        return np.full(len(oelevs), mean_v)

    z05, z95 = np.percentile(elevs, [5, 95])
    if (z95 - z05) < min_elev_diff:
        return _basic_vp(oelevs, mean_v, 0.0)

    gamma0 = -0.0065
    use_basic = n < num_min_prof

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_basic:
            def obj2(p):
                pred = _basic_vp(elevs, p[0], p[1])
                return float(np.log(max(np.sqrt(np.mean((pred - values) ** 2)), 1e-6)))
            res = minimize(obj2, [mean_v, gamma0], method="Nelder-Mead",
                           options={"xatol": 1e-2, "fatol": 1e-2, "maxiter": 1000})
            return _basic_vp(oelevs, res.x[0], res.x[1])
        else:
            p10, p90 = np.percentile(elevs, [10, 90])
            def obj5(p):
                pred = _full_vp(elevs, p[0], p[1], p[2], p[3], p[4])
                return float(np.log(max(np.sqrt(np.mean((pred - values) ** 2)), 1e-6)))
            res = minimize(obj5, [mean_v, gamma0, 5.0, p10, p90 - p10],
                           method="Nelder-Mead",
                           options={"xatol": 1e-2, "fatol": 1e-2, "maxiter": 2000})
            return _full_vp(oelevs, *res.x)


def _compute_vp_theil_sen(
    elevs: np.ndarray,
    oelevs: np.ndarray,
    values: np.ndarray,
    num_min_prof: int,
    min_elev_diff: float,
) -> np.ndarray:
    """
    Theil-Sen vertical profile (titanlib VerticalProfileTheilSen=1).
    Falls back to constant lapse rate gamma=-0.0065 when sample is too small
    or elevation spread too narrow.
    """
    elevs = np.asarray(elevs, dtype=float)
    oelevs = np.asarray(oelevs, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(elevs)
    gamma = -0.0065

    if elevs.min() == elevs.max():
        return np.full(len(oelevs), float(values.mean()))

    z05, z95 = np.percentile(elevs, [5, 95])
    use_basic = n < num_min_prof or (z95 - z05) < min_elev_diff

    if use_basic:
        m = gamma
    else:
        slopes: list[float] = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                dz = abs(elevs[i] - elevs[j])
                slopes.append(0.0 if dz < 1.0 else (values[i] - values[j]) / (elevs[i] - elevs[j]))
        m = float(np.median(slopes)) if slopes else gamma

    q = values - m * elevs
    q_med = float(np.median(q))
    return q_med + m * oelevs


def _compute_background(
    elevs: np.ndarray,
    values: np.ndarray,
    background_type: int,
    external_bg: np.ndarray,
    indices_global_outer: list[int],
    num_min_prof: int,
    min_elev_diff: float,
    value_minp: float,
    value_maxp: float,
) -> np.ndarray:
    """Compute background for the outer-circle subset."""
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)

    if background_type == VERTICAL_PROFILE:
        bg = _compute_vp(elevs, elevs, values, num_min_prof, min_elev_diff)
    elif background_type == VERTICAL_PROFILE_THEIL_SEN:
        bg = _compute_vp_theil_sen(elevs, elevs, values, num_min_prof, min_elev_diff)
    elif background_type == MEAN_OUTER_CIRCLE:
        bg = np.full(n, float(values.mean()))
    elif background_type == MEDIAN_OUTER_CIRCLE:
        bg = np.full(n, float(np.median(values)))
    elif background_type == EXTERNAL:
        bg = np.asarray(external_bg, dtype=float)[indices_global_outer]
    else:
        raise ValueError(f"Unknown background_type: {background_type}")

    if np.isfinite(value_minp):
        bg = np.where(bg < value_minp, value_minp, bg)
    if np.isfinite(value_maxp):
        bg = np.where(bg > value_maxp, value_maxp, bg)
    return bg

# ---------------------------------------------------------------------------
# Index management (mirrors titanlib::set_indices from sct.cpp)
# ---------------------------------------------------------------------------

def _set_indices(
    indices_global_outer_guess: list[int],
    obs_test: np.ndarray,
    flags: np.ndarray,
    distances: list[float],
    inner_radius: float,
    curr: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """
    Partition a guess neighbour list into the outer / inner / test hierarchy.

    curr < 0  → test every unchecked (flags == NA) inner observation
    curr >= 0 → test only `curr` (forced into test; handles re-check of bad obs)

    Returns
    -------
    indices_global_outer : global indices of non-bad outer observations
    indices_global_test  : global indices of observations to test
    indices_outer_inner  : for each inner obs, its position in the outer list
    indices_outer_test   : for each test obs, its position in the outer list
    indices_inner_test   : for each test obs, its position in the inner list
    """
    # Outer: non-bad neighbours (flags != 1). When re-checking a bad obs (curr
    # >= 0, flags[curr] == 1), we force curr into outer so fgt_core can access
    # its observed value and apply a background built from clean neighbours.
    indices_global_outer: list[int] = []
    distances_outer: list[float] = []

    curr_in_guess = False
    for g, d in zip(indices_global_outer_guess, distances):
        g = int(g)
        if flags[g] != 1:
            indices_global_outer.append(g)
            distances_outer.append(float(d))
        if g == curr:
            curr_in_guess = True

    # Force curr into outer for the re-check phase
    if curr >= 0 and flags[curr] == 1 and curr not in indices_global_outer:
        for g, d in zip(indices_global_outer_guess, distances):
            if int(g) == curr:
                indices_global_outer.append(curr)
                distances_outer.append(float(d))
                break

    # Inner: outer members within inner_radius
    indices_outer_inner: list[int] = []
    for i, (g, d) in enumerate(zip(indices_global_outer, distances_outer)):
        if d <= inner_radius:
            indices_outer_inner.append(i)

    # Position maps for fast lookup
    inner_pos: dict[int, int] = {
        indices_global_outer[i]: l
        for l, i in enumerate(indices_outer_inner)
    }

    # Test set
    indices_outer_test: list[int] = []
    indices_inner_test: list[int] = []
    indices_global_test: list[int] = []

    if curr < 0:
        for l, i in enumerate(indices_outer_inner):
            g = indices_global_outer[i]
            if obs_test[g] == 1 and flags[g] == NA:
                indices_outer_test.append(i)
                indices_inner_test.append(l)
                indices_global_test.append(g)
    else:
        if curr in inner_pos:
            l = inner_pos[curr]
            i = indices_outer_inner[l]
            indices_outer_test.append(i)
            indices_inner_test.append(l)
            indices_global_test.append(curr)

    return (
        indices_global_outer,
        indices_global_test,
        indices_outer_inner,
        indices_outer_test,
        indices_inner_test,
    )

# ---------------------------------------------------------------------------
# FGT core (mirrors titanlib::fgt_core)
# ---------------------------------------------------------------------------

def _fgt_core(
    yo: np.ndarray,
    yb: np.ndarray,
    sigma_b: np.ndarray,
    minp: float,
    maxp: float,
    mina: np.ndarray,
    maxa: np.ndarray,
    minv: np.ndarray,
    maxv: np.ndarray,
    tpos: np.ndarray,
    tneg: np.ndarray,
    indices_global_test: list[int],
    indices_outer_inner: list[int],
    indices_outer_test: list[int],
    indices_inner_test: list[int],
    basic: bool,
    set_flag0: bool,
    scores: np.ndarray,
    flags: np.ndarray,
) -> int:
    """
    First-guess test for one inner circle.

    All array arguments are outer-circle-indexed (length = p_outer).
    Mutates `scores` and `flags` (global-indexed) in-place.
    Returns number of new bad flags set.
    """
    p_inner = len(indices_outer_inner)
    p_test = len(indices_global_test)
    if p_inner == 0 or p_test == 0:
        return 0

    # chi = |obs - background| / sigma_b, for every inner obs
    chi_inner = np.empty(p_inner, dtype=float)
    chi_inner_alt = np.empty(p_inner, dtype=float)
    chi_stat: list[float] = []
    chi_stat_alt: list[float] = []

    for l, i in enumerate(indices_outer_inner):
        # sigma_b == 0 with obs == bg → 0/0 = NaN, treat as chi=0 (no flag).
        # sigma_b == 0 with obs != bg → inf, matching C++ behaviour (always flags).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = abs(yo[i] - yb[i]) / sigma_b[i] if sigma_b[i] != 0.0 else (
                0.0 if yo[i] == yb[i] else float("inf")
            )
        chi_inner[l] = raw
        chi_inner_alt[l] = float(maxv[i] - minv[i])
        if mina[i] <= yb[i] <= maxa[i]:
            chi_stat.append(chi_inner[l])
            chi_stat_alt.append(chi_inner_alt[l])

    # All backgrounds outside admissible range → flag everything bad
    if not chi_stat:
        for g in indices_global_test:
            flags[g] = 1
        return len(indices_global_test)

    # Non-basic mode: standardise chi by median/IQR
    mu = sigma = sigma_mu = 0.0
    if not basic:
        ca = np.array(chi_stat)
        ca_alt = np.array(chi_stat_alt)
        mu = float(np.percentile(ca, 50))
        sigma = float(np.percentile(ca, 75) - np.percentile(ca, 25))
        sigma_alt = float(np.percentile(ca_alt, 75) - np.percentile(ca_alt, 25))
        if sigma_alt > sigma:
            sigma = sigma_alt
        if sigma == 0.0:
            return 0
        sigma_mu = sigma / np.sqrt(len(chi_stat))

    # Find worst test observation (largest z with background outside valid range)
    zmx = -1e9
    mmx = -1
    for m in range(p_test):
        i = indices_outer_test[m]
        l = indices_inner_test[m]
        z = chi_inner[l] if basic else (chi_inner[l] - mu) / (sigma + sigma_mu)
        if z > zmx and (yb[i] < minv[i] or yb[i] > maxv[i]):
            zmx = z
            mmx = m

    if mmx >= 0:
        i = indices_outer_test[mmx]
        thr = float(tpos[i]) if (yo[i] - yb[i]) >= 0 else float(tneg[i])
        if zmx > thr:
            g = indices_global_test[mmx]
            scores[g] = zmx
            flags[g] = 1
            return 1

    if set_flag0:
        for g in indices_global_test:
            flags[g] = 0

    return 0

# ---------------------------------------------------------------------------
# FGT main (mirrors titanlib::fgt)
# ---------------------------------------------------------------------------

def fgt_raw(
    lats,
    lons,
    elevs,
    values,
    obs_to_check,
    background_values,
    background_uncertainties,
    background_elab_type: int,
    num_min_outer: int,
    num_max_outer: int,
    inner_radius: float,
    outer_radius: float,
    num_iterations: int,
    num_min_prof: int,
    min_elev_diff: float,
    values_mina,
    values_maxa,
    values_minv,
    values_maxv,
    tpos,
    tneg,
    debug: bool = False,
    basic: bool = True,
) -> tuple[list[int], list[float]]:
    """
    Python reimplementation of titanlib.fgt.

    Returns (flags, scores) – same tuple as the titanlib Python binding.
    Three phases:
      1. Main iterative loop (until convergence or num_iterations exhausted)
      2. Sweep over observations still missing a QC flag
      3. Re-check of bad observations using only good neighbours
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    values_mina = np.asarray(values_mina, dtype=float)
    values_maxa = np.asarray(values_maxa, dtype=float)
    values_minv = np.asarray(values_minv, dtype=float)
    values_maxv = np.asarray(values_maxv, dtype=float)
    tpos = np.asarray(tpos, dtype=float)
    tneg = np.asarray(tneg, dtype=float)
    background_values = np.asarray(background_values, dtype=float)

    p = len(lats)
    flags = np.full(p, NA, dtype=float)
    scores = np.full(p, NA, dtype=float)
    obs_test = np.ones(p, dtype=int)
    sigma_b = np.ones(p, dtype=float)

    if len(obs_to_check) == p:
        obs_test[:] = np.asarray(obs_to_check, dtype=int)
    if len(background_uncertainties) == p:
        sigma_b[:] = np.asarray(background_uncertainties, dtype=float)

    value_minp = float(np.min(values_mina))
    value_maxp = float(np.max(values_maxa))

    tree = _build_tree(lats, lons)

    # ------------------------------------------------------------------
    # Helper: run fgt_core for one centroid observation
    # ------------------------------------------------------------------
    def _process_centroid(curr_arg: int) -> int:
        """Build outer/inner/test, compute background, call fgt_core.
        curr_arg < 0 means 'test all unchecked inner obs' (main loop).
        Returns thrown_out increment."""

        # Centroid for tree query: when re-checking a bad obs, curr_arg is its
        # global index; use it as the spatial centroid.
        centre = curr_arg if curr_arg >= 0 else _curr
        idx_guess, dists_guess = _query_radius(
            tree, lats[centre], lons[centre], outer_radius
        )

        # Cap to num_max_outer nearest neighbours
        if len(idx_guess) > num_max_outer:
            order = np.argsort(dists_guess)[:num_max_outer]
            idx_guess = [idx_guess[k] for k in order]
            dists_guess = [dists_guess[k] for k in order]

        (ig_outer, ig_test, io_inner, io_test, ii_test) = _set_indices(
            idx_guess, obs_test, flags, dists_guess, inner_radius, curr_arg
        )

        p_outer = len(ig_outer)
        p_inner = len(io_inner)
        p_test = len(ig_test)

        if p_outer < num_min_outer:
            flags[centre] = 12
            return 0
        if p_inner < 2:
            flags[centre] = 11
            return 0
        if p_test == 0:
            return 0

        # Subset to outer circle
        g_o = np.asarray(ig_outer, dtype=int)
        elev_o = elevs[g_o]
        val_o = values[g_o]
        sig_o = sigma_b[g_o]
        tpos_o = tpos[g_o]
        tneg_o = tneg[g_o]
        mina_o = values_mina[g_o]
        maxa_o = values_maxa[g_o]
        minv_o = values_minv[g_o]
        maxv_o = values_maxv[g_o]

        bval_o = _compute_background(
            elev_o, val_o, background_elab_type, background_values, ig_outer,
            num_min_prof, min_elev_diff, value_minp, value_maxp,
        )

        # Small-innovation shortcut: background within valid range for all test obs
        if curr_arg < 0:
            small_innov = all(
                minv_o[io_test[m]] <= bval_o[io_test[m]] <= maxv_o[io_test[m]]
                for m in range(p_test)
            )
        else:
            j = io_test[0]
            small_innov = minv_o[j] <= bval_o[j] <= maxv_o[j]

        if small_innov:
            for g in ig_test:
                flags[g] = 0
            return 0

        return _fgt_core(
            val_o, bval_o, sig_o, value_minp, value_maxp,
            mina_o, maxa_o, minv_o, maxv_o, tpos_o, tneg_o,
            ig_test, io_inner, io_test, ii_test,
            basic, _set_flag0, scores, flags,
        )

    # ------------------------------------------------------------------
    # Phase 1: main iterative loop
    # ------------------------------------------------------------------
    set_all_good = False
    _curr = -1  # used by _process_centroid as spatial centroid in main loop

    for iteration in range(num_iterations):
        thrown_out = 0
        _set_flag0 = iteration > 0

        for curr in range(p):
            _curr = curr
            if obs_test[curr] != 1 or flags[curr] >= 0:
                continue
            thrown_out += _process_centroid(-1)

        if debug:
            print(f"Phase-1 iter {iteration}: thrown_out={thrown_out}")

        if thrown_out == 0:
            if iteration == 0:
                set_all_good = True
            break

    if set_all_good:
        mask = (flags == NA) & (obs_test == 1)
        flags[mask] = 0

    # ------------------------------------------------------------------
    # Phase 2: sweep over still-unchecked observations
    # ------------------------------------------------------------------
    _set_flag0 = True
    for curr in range(p):
        if obs_test[curr] != 1 or flags[curr] >= 0:
            continue
        _curr = curr
        _process_centroid(curr)

    # ------------------------------------------------------------------
    # Phase 3: re-check bad observations with clean-neighbour background
    # ------------------------------------------------------------------
    for curr in range(p):
        if obs_test[curr] != 1 or flags[curr] != 1:
            continue
        if values[curr] < value_minp or values[curr] > value_maxp:
            continue
        _curr = curr
        _process_centroid(curr)

    return flags.astype(int).tolist(), scores.tolist()


# ---------------------------------------------------------------------------
# first_guess_test_py  (same interface as clean_observation_tests.first_guess_test)
# ---------------------------------------------------------------------------

def first_guess_test_py(
    stations,
    lats,
    lons,
    elevs,
    values,
    background_values,
    para: str,
    date: str,
    background_elab_type: int = 1,
    num_min_outer: int = 3,
    num_max_outer: int = 10,
    inner_radius: float = 50_000,
    outer_radius: float = 100_000,
    num_iterations: int = 10,
    num_min_prof: int = 1,
    min_elev_diff: float = 250,
    min_horizontal_scale: float = 250,     # unused – kept for API parity
    max_horizontal_scale: float = 100_000,  # unused – kept for API parity
    kth_closest_obs_horizontal_scale: int = 2,  # unused – kept for API parity
    bdebug: bool = False,
    bbasic: bool = True,
    tpostneg=5,
) -> dict:
    """
    FGT wrapper – same interface as clean_observation_tests.first_guess_test.
    tpostneg may be scalar or [smn_thr, other_thr].
    """
    thresholds = tpostneg if isinstance(tpostneg, list) else [tpostneg, tpostneg]
    thr_short = float(thresholds[0])
    thr_long = float(thresholds[1] if len(thresholds) > 1 else thresholds[0])

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    background_values = np.asarray(background_values, dtype=float)
    snames = np.asarray(stations)
    short_mask = np.array([len(s) == 3 for s in snames])
    N = len(lats)

    obs_to_check = np.ones(N, dtype=int)
    bg_unc = np.full(N, float(np.nanstd(background_values)))
    values_mina = values - 25
    values_maxa = values + 25
    values_minv = values - 1
    values_maxv = values + 1

    flagged: set[int] = set()

    for thr, mask in [(thr_short, short_mask), (thr_long, ~short_mask)]:
        tv = np.full(N, thr)
        try:
            flg, scr = fgt_raw(
                lats, lons, elevs, values, obs_to_check, background_values, bg_unc,
                background_elab_type, num_min_outer, num_max_outer,
                inner_radius, outer_radius, num_iterations, num_min_prof,
                min_elev_diff, values_mina, values_maxa, values_minv, values_maxv,
                tv, tv, debug=bdebug, basic=bbasic,
            )
        except Exception:
            flg, scr = [], []
        if flg:
            fa = np.asarray(flg)
            sa = np.asarray(scr)
            for idx in np.where((fa > 0) & (sa > 0))[0]:
                if mask[idx]:
                    flagged.add(int(idx))

    my_dict: dict = {"Station": [], "Time": [], "Parameter": []}
    for idx in sorted(flagged):
        if np.isfinite(values[idx]):
            my_dict["Station"].append(snames[idx])
            my_dict["Time"].append(date)
            my_dict["Parameter"].append(para)
    return my_dict if my_dict["Station"] else {}

# ---------------------------------------------------------------------------
# SCT helpers (shared by sct_resistant and sct_dual)
# ---------------------------------------------------------------------------

def _compute_pairwise_distances(
    lats_b: np.ndarray, lons_b: np.ndarray, elevs_b: np.ndarray
):
    """Return (disth, distz): N×N great-circle and elevation difference matrices."""
    xyz = _latlon_to_xyz(lats_b, lons_b)  # (N, 3)
    chord = np.sqrt(np.maximum(0.0, np.sum((xyz[:, None] - xyz[None, :]) ** 2, axis=2)))
    arc = 2.0 * EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    dz = np.abs(elevs_b[:, None] - elevs_b[None, :])
    return arc, dz


def _build_sct_cov_matrix(
    disth: np.ndarray,
    distz: np.ndarray,
    Dh_mean: float,
    vertical_scale: float,
    eps2_b: np.ndarray,
) -> np.ndarray:
    """Gaussian covariance matrix S with eps2 added to the diagonal."""
    S = np.exp(
        -0.5 * (disth / Dh_mean) ** 2 - 0.5 * (distz / vertical_scale) ** 2
    )
    np.fill_diagonal(S, S.diagonal() + eps2_b)
    return S


def _adaptive_dh_mean(
    disth: np.ndarray,
    kth_closest: int,
    min_scale: float,
    max_scale: float,
) -> float:
    """
    Compute adaptive Dh_mean used by sct_resistant.

    For each station i, Dh(i) = kth-closest actual-neighbour distance.
    The C++ vector includes a spurious 0 at slot N-1, so kth=2 effectively
    picks the 1st-closest actual distance.  We replicate this by appending
    a 0 before sorting.
    """
    N = disth.shape[0]
    Dh_vec = np.zeros(N, dtype=float)
    for i in range(N):
        row_actual = sorted(disth[i, j] for j in range(N) if j != i)
        row_with_zero = sorted(row_actual + [0.0])
        k = min(kth_closest, len(row_with_zero))
        Dh_vec[i] = row_with_zero[k - 1]
    Dh_mean = float(np.mean(Dh_vec))
    return float(np.clip(Dh_mean, min_scale, max_scale))


def _apply_condition(
    values: np.ndarray, thresholds: np.ndarray, condition: int
) -> np.ndarray:
    """Convert continuous values to binary {0, 1} via condition vs threshold."""
    if condition == CONDITION_EQ:
        return (values == thresholds).astype(float)
    elif condition == CONDITION_GT:
        return (values > thresholds).astype(float)
    elif condition == CONDITION_GEQ:
        return (values >= thresholds).astype(float)
    elif condition == CONDITION_LT:
        return (values < thresholds).astype(float)
    elif condition == CONDITION_LEQ:
        return (values <= thresholds).astype(float)
    else:
        raise ValueError(f"Unknown condition: {condition}")

# ---------------------------------------------------------------------------
# sct_resistant helpers
# ---------------------------------------------------------------------------

def _sct_set_indices(outer_guess, obs_test_arr, flags_arr, distances, inner_radius, forced_curr):
    """Partition outer-guess stations into outer/inner/test groups.

    forced_curr >= 0: only this station is the test station (forced into
                      outer+inner regardless of its flag value).
    forced_curr == -1: all untested (flags < 0) obs_test=1 inner stations
                       are test stations.
    """
    outer_pairs = []  # (global_idx, distance)
    for g, d in zip(outer_guess, distances):
        if flags_arr[g] != 1 or (forced_curr >= 0 and g == forced_curr):
            outer_pairs.append((g, d))

    if not outer_pairs:
        return [], [], [], [], []

    indices_global_outer = [g for g, _ in outer_pairs]
    indices_outer_inner = []    # [l] = i (outer index of l-th inner station)
    indices_global_test = []
    indices_outer_test = []     # [m] = i (outer index of m-th test station)
    indices_inner_test = []     # [m] = l (inner index of m-th test station)

    for i, (g, d) in enumerate(outer_pairs):
        if d <= inner_radius:
            l = len(indices_outer_inner)
            indices_outer_inner.append(i)
            is_test = (g == forced_curr) if forced_curr >= 0 \
                      else (obs_test_arr[g] == 1 and flags_arr[g] < 0)
            if is_test:
                indices_global_test.append(g)
                indices_outer_test.append(i)
                indices_inner_test.append(l)

    return (indices_global_outer, indices_global_test,
            indices_outer_inner, indices_outer_test, indices_inner_test)


def _sct_core(
    yo, yb, Sinv, S_uw, eps2_o, tpos_o, tneg_o,
    mina_o, maxa_o, minv_o, maxv_o, minp, maxp,
    ioi, iot, iit, igt,
    basic, set_flag0, flags_arr, scores_arr,
):
    """Chi score and flag the worst inner-circle test station.

    Implements titanlib/sct_resistant.cpp sct_core().

    yo/yb/… are outer-circle arrays (p_outer).
    Sinv = (S + eps2*I)^{-1}  [p_outer × p_outer]
    S_uw = S without eps2 on diagonal (Gaussian-only part)
    ioi  = indices_outer_inner  [l] -> outer index
    iot  = indices_outer_test   [m] -> outer index
    iit  = indices_inner_test   [m] -> inner index
    igt  = indices_global_test  [m] -> global index

    Returns n_thrown_out (0 or 1).
    """
    p_inner = len(ioi)
    p_test = len(igt)

    d = yo - yb
    Sinv_d = Sinv @ d

    yav = np.empty(p_inner)
    chi_inner = np.empty(p_inner)
    chi_inner_alt = np.empty(p_inner)
    chi_stat = []
    chi_stat_alt = []

    for l in range(p_inner):
        i = ioi[l]
        # OI analysis (S_uw = S without eps2 on diagonal)
        ya = float(np.clip(yb[i] + S_uw[i, :] @ Sinv_d, minp, maxp))
        # leave-one-out cross-validation analysis
        yav[l] = float(np.clip(yo[i] - Sinv_d[i] / Sinv[i, i], minp, maxp))

        prod = (yo[i] - ya) * (yo[i] - yav[l])
        chi_inner[l] = np.sqrt(max(float(prod), 0.0))
        chi_inner_alt[l] = (
            np.sqrt(eps2_o[i] / (1.0 + eps2_o[i])) * (maxv_o[i] - minv_o[i])
        )
        if mina_o[i] <= yav[l] <= maxa_o[i]:
            chi_stat.append(chi_inner[l])
            chi_stat_alt.append(chi_inner_alt[l])

    # All yav outside admissible range → flag all test stations as bad
    if not chi_stat:
        for g in igt:
            flags_arr[g] = 1
        return len(igt)

    # IQR statistics for non-basic mode
    mu = sigma = sigma_mu = 0.0
    if not basic:
        ca = np.asarray(chi_stat)
        caa = np.asarray(chi_stat_alt)
        mu = float(np.median(ca))
        q = np.quantile(ca, [0.25, 0.75])
        sigma = float(q[1] - q[0])
        qa = np.quantile(caa, [0.25, 0.75])
        sigma_alt = float(qa[1] - qa[0])
        if sigma_alt > sigma:
            sigma = sigma_alt
        if sigma == 0.0:
            return 0
        sigma_mu = sigma / np.sqrt(len(chi_stat))

    # Find worst test station (highest z, yav outside valid range)
    zmx = -1e9
    mmx = -1
    for m in range(p_test):
        i = iot[m]
        l = iit[m]
        z = chi_inner[l] if basic else (chi_inner[l] - mu) / (sigma + sigma_mu)
        if z > zmx and (yav[l] < minv_o[i] or yav[l] > maxv_o[i]):
            zmx = z
            mmx = m

    if mmx < 0:
        if set_flag0:
            for g in igt:
                flags_arr[g] = 0
        return 0

    i = iot[mmx]
    thr = float(tneg_o[i]) if (yo[i] - yb[i]) < 0.0 else float(tpos_o[i])
    if zmx > thr:
        g = igt[mmx]
        scores_arr[g] = zmx
        flags_arr[g] = 1
        return 1
    elif set_flag0:
        for g in igt:
            flags_arr[g] = 0
    return 0


# ---------------------------------------------------------------------------
# sct_resistant
# ---------------------------------------------------------------------------

def sct_resistant_raw(
    lats,
    lons,
    elevs,
    values,
    obs_to_check,
    background_values,
    background_elab_type: int,
    num_min_outer: int,
    num_max_outer: int,
    inner_radius: float,
    outer_radius: float,
    num_iterations: int,
    num_min_prof: int,
    min_elev_diff: float,
    min_horizontal_scale: float,
    max_horizontal_scale: float,
    kth_closest_obs_horizontal_scale: int,
    vertical_scale: float,
    values_mina,
    values_maxa,
    values_minv,
    values_maxv,
    eps2,
    tpos,
    tneg,
    debug: bool = False,
    basic: bool = True,
):
    """
    Python reimplementation of titanlib.sct_resistant.

    Three-loop structure from sct_resistant.cpp:
      1. Main iterations: flag worst bad obs per neighbourhood, repeat until stable.
      2. Missing-QC pass: test any station not reached in the main loop.
      3. Re-check bad obs: rescue wrongly-flagged stations using good neighbours only.

    Score: chi = sqrt((yo-ya)*(yo-yav)) where ya is the OI analysis and
    yav is the leave-one-out cross-validation analysis.

    Returns (flags, scores).
      flags : -999 not tested, 0 good, 1 bad, 100 OI failure
      scores: -999 for untested/good; chi or z-score at flag time for bad stations
    Flag condition in wrappers: flags > 0 AND scores > 15.
    """
    na = -999.0

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    values_mina = np.asarray(values_mina, dtype=float)
    values_maxa = np.asarray(values_maxa, dtype=float)
    values_minv = np.asarray(values_minv, dtype=float)
    values_maxv = np.asarray(values_maxv, dtype=float)
    eps2 = np.asarray(eps2, dtype=float)
    tpos = np.asarray(tpos, dtype=float)
    tneg = np.asarray(tneg, dtype=float)
    background_values = np.asarray(background_values, dtype=float)

    p = len(lats)
    obs_test = np.ones(p, dtype=int)
    if len(obs_to_check) == p:
        obs_test[:] = np.asarray(obs_to_check, dtype=int)

    flags = np.full(p, na)
    scores = np.full(p, na)

    value_minp = float(np.min(values_mina))
    value_maxp = float(np.max(values_maxa))

    tree = _build_tree(lats, lons)

    def _run_neighbourhood(curr, forced_curr, set_flag0):
        """Run one SCT neighbourhood. Returns (n_thrown, is_isolated)."""
        idx_all, dist_all = _query_radius(tree, lats[curr], lons[curr], outer_radius)
        if len(idx_all) > num_max_outer:
            order = np.argsort(dist_all)[:num_max_outer]
            idx_all = [idx_all[k] for k in order]
            dist_all = [dist_all[k] for k in order]

        igo, igt, ioi, iot, iit = _sct_set_indices(
            idx_all, obs_test, flags, dist_all, inner_radius, forced_curr
        )
        p_outer = len(igo)
        p_inner = len(ioi)
        p_test  = len(igt)

        if p_outer < num_min_outer:
            return 0, True   # isolated: too few outer neighbours
        if p_inner < 2:
            return 0, True   # isolated: too few inner neighbours
        if p_test == 0:
            return 0, False

        g_arr = np.asarray(igo, dtype=int)
        yo    = values[g_arr]
        eps2_o = eps2[g_arr]
        tpos_o = tpos[g_arr]
        tneg_o = tneg[g_arr]
        mina_o = values_mina[g_arr]
        maxa_o = values_maxa[g_arr]
        minv_o = values_minv[g_arr]
        maxv_o = values_maxv[g_arr]

        yb = _compute_background(
            elevs[g_arr], yo, background_elab_type,
            background_values, igo, num_min_prof, min_elev_diff,
            value_minp, value_maxp,
        )

        # small_innov: background within [minv, maxv] for all test stations
        if all(minv_o[iot[m]] <= yb[iot[m]] <= maxv_o[iot[m]] for m in range(p_test)):
            for g in igt:
                flags[g] = 0
            return 0, False

        disth, distz = _compute_pairwise_distances(
            lats[g_arr], lons[g_arr], elevs[g_arr]
        )
        Dh_mean = _adaptive_dh_mean(
            disth, kth_closest_obs_horizontal_scale,
            min_horizontal_scale, max_horizontal_scale,
        )
        S = _build_sct_cov_matrix(disth, distz, Dh_mean, vertical_scale, eps2_o)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            flags[curr] = 100
            return 0, False

        S_uw = S.copy()
        np.fill_diagonal(S_uw, S_uw.diagonal() - eps2_o)

        n = _sct_core(
            yo, yb, Sinv, S_uw, eps2_o, tpos_o, tneg_o,
            mina_o, maxa_o, minv_o, maxv_o, value_minp, value_maxp,
            ioi, iot, iit, igt, basic, set_flag0, flags, scores,
        )
        return n, False

    # ── Main iteration loop ──────────────────────────────────────────────────
    set_all_good = False
    for iteration in range(num_iterations):
        thrown_out = 0
        for curr in range(p):
            if obs_test[curr] != 1 or flags[curr] >= 0:
                continue
            n, isolated = _run_neighbourhood(curr, -1, iteration > 0)
            if isolated:
                flags[curr] = 0
            thrown_out += n
        if thrown_out == 0:
            if iteration == 0:
                set_all_good = True
            break

    if set_all_good:
        for curr in range(p):
            if flags[curr] == na and obs_test[curr] == 1:
                flags[curr] = 0

    # ── Missing-QC pass ──────────────────────────────────────────────────────
    for curr in range(p):
        if obs_test[curr] != 1 or flags[curr] >= 0:
            continue
        _, isolated = _run_neighbourhood(curr, curr, True)
        if isolated:
            flags[curr] = 0

    # ── Re-check bad observations ─────────────────────────────────────────────
    for curr in range(p):
        if obs_test[curr] != 1 or flags[curr] != 1:
            continue
        if values[curr] < value_minp or values[curr] > value_maxp:
            continue
        _run_neighbourhood(curr, curr, True)

    return flags.tolist(), scores.tolist()


def sct_resistant_py(
    stations,
    lats,
    lons,
    elevs,
    values,
    para: str,
    date: str,
    background_elab_type: int = 1,
    num_min_outer: int = 3,
    num_max_outer: int = 10,
    inner_radius: float = 50_000,
    outer_radius: float = 100_000,
    num_iterations: int = 10,
    num_min_prof: int = 1,
    min_elev_diff: float = 250,
    min_horizontal_scale: float = 250,
    max_horizontal_scale: float = 100_000,
    kth_closest_obs_horizontal_scale: int = 2,
    vertical_scale: float = 200,
    bdebug: bool = False,
    bbasic: bool = True,
) -> dict:
    """
    SCT resistant – same interface as clean_observation_tests.spacial_ct_resistant.

    Fixed constants matching titanlib wrapper:
      tpos = tneg = 16, eps2 = 0.5,
      values_mina/maxa = values ± 15, values_minv/maxv = values ± 1
    Flag condition: flags > 0 AND scores > 15.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    snames = np.asarray(stations)
    N = len(lats)

    obs_to_check = np.ones(N, dtype=int)
    background_values = np.zeros(N, dtype=float)
    tpos_arr = np.full(N, 16.0)
    tneg_arr = np.full(N, 16.0)
    eps2_arr = np.full(N, 0.5)
    values_mina = values - 15
    values_maxa = values + 15
    values_minv = values - 1
    values_maxv = values + 1

    try:
        flags_list, scores_list = sct_resistant_raw(
            lats, lons, elevs, values, obs_to_check, background_values,
            background_elab_type, num_min_outer, num_max_outer,
            inner_radius, outer_radius, num_iterations, num_min_prof,
            min_elev_diff, min_horizontal_scale, max_horizontal_scale,
            kth_closest_obs_horizontal_scale, vertical_scale,
            values_mina, values_maxa, values_minv, values_maxv,
            eps2_arr, tpos_arr, tneg_arr, bdebug, bbasic,
        )
    except Exception:
        return {}

    fa = np.asarray(flags_list)
    sa = np.asarray(scores_list)
    my_dict: dict = {"Station": [], "Time": [], "Parameter": []}
    for idx in np.where((fa > 0) & (sa > 15))[0]:
        if np.isfinite(values[idx]):
            my_dict["Station"].append(snames[idx])
            my_dict["Time"].append(date)
            my_dict["Parameter"].append(para)
    return my_dict if my_dict["Station"] else {}

# ---------------------------------------------------------------------------
# sct_dual
# ---------------------------------------------------------------------------

def sct_dual_raw(
    lats,
    lons,
    elevs,
    values,
    obs_to_check,
    event_thresholds,
    condition: int,
    num_min_outer: int,
    num_max_outer: int,
    inner_radius: float,
    outer_radius: float,
    num_iterations: int,
    min_horizontal_scale: float,
    max_horizontal_scale: float,
    kth_closest_obs_horizontal_scale: int,
    vertical_scale: float,
    test_thresholds,
    debug: bool = False,
) -> list:
    """
    Python reimplementation of titanlib.sct_dual.

    Binary / event-based spatial consistency test (titanlib/src/sct_dual.cpp).
    Values are binarised via (values OP event_thresholds) per condition.
    For each neighbourhood, the OI covariance is split into w=0 / w=1 sub-matrices
    and the leave-one-out IDI (Information Dissociation Index) is compared via a
    KL-divergence-like score.  At most one station per neighbourhood per iteration.

    Returns integer flag list: 0 = passed, 1 = flagged.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    event_thresholds = np.asarray(event_thresholds, dtype=float)
    test_thresholds = np.asarray(test_thresholds, dtype=float)

    s = len(lats)
    obs_test = np.ones(s, dtype=int)
    has_obs_to_check = len(obs_to_check) == s
    if has_obs_to_check:
        obs_test[:] = np.asarray(obs_to_check, dtype=int)

    eps2_global = np.full(s, 0.5, dtype=float)
    w_global = _apply_condition(values, event_thresholds, condition)
    flags = np.zeros(s, dtype=int)
    tree = _build_tree(lats, lons)

    for iteration in range(num_iterations):
        thrown_out = 0
        checked = np.zeros(s, dtype=int)

        for curr in range(s):
            if has_obs_to_check and obs_test[curr] != 1:
                checked[curr] = 1
                continue
            if flags[curr] != 0:
                checked[curr] = 1
                continue
            if checked[curr] > 0:
                continue

            idx_all, dist_all = _query_radius(tree, lats[curr], lons[curr], outer_radius)
            pairs = [(j, d) for j, d in zip(idx_all, dist_all) if flags[j] == 0]

            if len(pairs) > num_max_outer:
                pairs = sorted(pairs, key=lambda x: x[1])[:num_max_outer]

            N = len(pairs)
            if N < num_min_outer:
                checked[curr] = 1
                continue

            idx_b = [p[0] for p in pairs]
            dist_b = np.asarray([p[1] for p in pairs])
            g = np.asarray(idx_b, dtype=int)

            lats_b = lats[g]
            lons_b = lons[g]
            elevs_b = elevs[g]
            w_b = w_global[g]
            eps2_b = eps2_global[g]
            t_b = test_thresholds[g]

            disth, distz = _compute_pairwise_distances(lats_b, lons_b, elevs_b)
            Dh_mean = _adaptive_dh_mean(
                disth, kth_closest_obs_horizontal_scale,
                min_horizontal_scale, max_horizontal_scale,
            )

            S = _build_sct_cov_matrix(disth, distz, Dh_mean, vertical_scale, eps2_b)
            S_uw = S.copy()
            np.fill_diagonal(S_uw, S_uw.diagonal() - eps2_b)

            # Split outer circle into w=0 and w=1 sub-groups
            w0_pos = np.where(w_b == 0)[0]
            w1_pos = np.where(w_b == 1)[0]
            p_w0 = len(w0_pos)
            p_w1 = len(w1_pos)

            i0_map = np.full(N, -1, dtype=int)
            i1_map = np.full(N, -1, dtype=int)
            for k, j in enumerate(w0_pos):
                i0_map[j] = k
            for k, j in enumerate(w1_pos):
                i1_map[j] = k

            Sinv_w0 = Sinv_w1 = Sinv_d_w0 = Sinv_d_w1 = None

            if p_w0 > 0:
                try:
                    Sinv_w0 = np.linalg.inv(S[np.ix_(w0_pos, w0_pos)])
                    Sinv_d_w0 = Sinv_w0.sum(axis=1)
                except np.linalg.LinAlgError:
                    pass

            if p_w1 > 0:
                try:
                    Sinv_w1 = np.linalg.inv(S[np.ix_(w1_pos, w1_pos)])
                    Sinv_d_w1 = Sinv_w1.sum(axis=1)
                except np.linalg.LinAlgError:
                    pass

            if Sinv_w0 is None and Sinv_w1 is None:
                checked[curr] = 1
                continue

            zmx = float("nan")
            best_gi = -1
            inner_test_gis = []

            for li in range(N):
                gi = idx_b[li]
                if dist_b[li] > inner_radius:
                    continue
                if has_obs_to_check and obs_test[gi] != 1:
                    checked[gi] = 1
                    continue
                inner_test_gis.append(gi)

                i0 = i0_map[li]
                i1 = i1_map[li]

                # Cross-class IDI (sum over opposite-class neighbours)
                w0_idiv_cross = 0.0
                w1_idiv_cross = 0.0
                for lj in range(N):
                    j0 = i0_map[lj]
                    j1 = i1_map[lj]
                    if i1 >= 0 and j0 >= 0 and Sinv_d_w0 is not None:
                        w0_idiv_cross += S_uw[li, lj] * Sinv_d_w0[j0]
                    elif i0 >= 0 and j1 >= 0 and Sinv_d_w1 is not None:
                        w1_idiv_cross += S_uw[li, lj] * Sinv_d_w1[j1]

                # Leave-one-out IDI (same-class) and cross-class IDI
                if i1 >= 0:  # station is w=1
                    w1_idiv = (max(1.0 - Sinv_d_w1[i1] / Sinv_w1[i1, i1], 0.001)
                               if Sinv_w1 is not None and Sinv_w1[i1, i1] != 0.0
                               else 0.001)
                    w0_idiv = max(w0_idiv_cross, 0.0)
                else:  # station is w=0
                    w0_idiv = (max(1.0 - Sinv_d_w0[i0] / Sinv_w0[i0, i0], 0.001)
                               if Sinv_w0 is not None and Sinv_w0[i0, i0] != 0.0
                               else 0.001)
                    w1_idiv = max(w1_idiv_cross, 0.0)

                if w1_idiv <= 0.0 or w0_idiv <= 0.0:
                    continue

                z0wrt1 = w0_idiv * np.log(w0_idiv / w1_idiv)
                z1wrt0 = w1_idiv * np.log(w1_idiv / w0_idiv)
                t_i = float(t_b[li])

                z = float("nan")
                if w_b[li] == 1.0 and w0_idiv > w1_idiv and z0wrt1 > t_i:
                    z = z0wrt1
                elif w_b[li] == 0.0 and w1_idiv > w0_idiv and z1wrt0 > t_i:
                    z = z1wrt0

                if np.isfinite(z) and (not np.isfinite(zmx) or z > zmx):
                    zmx = z
                    best_gi = gi

            if best_gi >= 0:
                flags[best_gi] = 1
                thrown_out += 1
                checked[best_gi] = 1

            for gi in inner_test_gis:
                checked[gi] = 1

        if thrown_out == 0:
            break

    return flags.tolist()


def sct_dual_py(
    stations,
    lats,
    lons,
    elevs,
    values,
    para: str,
    date: str,
    num_min_outer: int = 3,
    num_max_outer: int = 10,
    inner_radius: float = 50_000,
    outer_radius: float = 100_000,
    num_iterations: int = 10,
    min_horizontal_scale: float = 250,
    max_horizontal_scale: float = 100_000,
    kth_closest_obs_horizontal_scale: int = 2,
    vertical_scale: float = 200,
    debug: bool = False,
    condition: int = 1,
    event_thresholds: float = 0.1,
    test_thresholds: float = 0.8,
) -> dict:
    """
    SCT dual – same interface as clean_observation_tests.spacial_ct_dual.
    Flag condition: flags > 0.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    elevs = np.asarray(elevs, dtype=float)
    values = np.asarray(values, dtype=float)
    snames = np.asarray(stations)
    N = len(lats)

    obs_to_check = np.ones(N, dtype=int)
    ev_thr = np.full(N, float(event_thresholds))
    tst_thr = np.full(N, float(test_thresholds))

    try:
        flags_list = sct_dual_raw(
            lats, lons, elevs, values, obs_to_check, ev_thr, condition,
            num_min_outer, num_max_outer, inner_radius, outer_radius,
            num_iterations, min_horizontal_scale, max_horizontal_scale,
            kth_closest_obs_horizontal_scale, vertical_scale, tst_thr, debug,
        )
    except Exception:
        return {}

    fa = np.asarray(flags_list)
    my_dict: dict = {"Station": [], "Time": [], "Parameter": []}
    for idx in np.where(fa > 0)[0]:
        if np.isfinite(values[idx]):
            my_dict["Station"].append(snames[idx])
            my_dict["Time"].append(date)
            my_dict["Parameter"].append(para)
    return my_dict if my_dict["Station"] else {}
