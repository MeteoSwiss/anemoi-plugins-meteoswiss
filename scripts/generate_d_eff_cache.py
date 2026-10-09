"""Generate the barrier- and elevation-aware effective-distance (``d_eff``)
cache consumed by ``NudgeTowardObservation``'s ``d_eff_file`` parameter
(``anemoi_plugins_meteoswiss/transform/filters/nudging.py``).

Steps
-----
1.  Retrieve the station catalog with ``jretrieve.fetch_meta()``. Only
    station metadata is needed: d_eff depends on station/POI geometry and the
    DEM, not on observed values or a reference time.
2.  Trim the stations to a bounding box or to the Swiss national border
    (``--station-filter-mode``).
3.  Compute ``d_eff_poi`` (POI <-> station) and ``d_eff_sta``
    (station <-> station) with ``barrier_distances()``.
4.  Write both to a NetCDF file, together with each station's
    longitude/latitude, LV95 x/y and elevation (``station_*`` variables along
    ``sta``). The barrier hyperparameters and the station count are stored as
    NetCDF attributes and in a JSON sidecar file.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

LOG = logging.getLogger(__name__)

# DWH parameters queried by RetrieveObservation (_ECMWF_NAME_TO_DWH_PARAMS).
# fetch_meta() only returns stations that report these, so keep them in sync
# with RetrieveObservation; otherwise this catalog may miss stations that
# NudgeTowardObservation sees at run time.
DEFAULT_DWH_PARAMS = [
    "tre200s0",
    "tde200s0",
    "pp0qffs0",
    "prestas0",
    "rre150h0",
    "fkl010z0",
    "dkl010z0",
    "fkl010z1",
]

# Stations per barrier_distances() call, so progress is visible in the logs.
PROGRESS_EVERY = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--jretrieve-src-path",
        required=True,
        help="Directory containing the 'data_input' package (data_input/jretrieve.py).",
    )
    p.add_argument("--dwh-params", nargs="+", default=DEFAULT_DWH_PARAMS)
    p.add_argument("--seq-type", default="surface")
    p.add_argument(
        "--stations-bbox",
        type=float,
        nargs=4,
        default=[40.5, 53.0, 0.0, 17.5],
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Bounding box passed to jretrieve.fetch_meta(). The stations are then "
        "trimmed with --station-filter-mode.",
    )
    p.add_argument(
        "--station-filter-mode",
        choices=["domain", "switzerland"],
        default="domain",
        help="'domain': keep stations inside --domain-bbox. 'switzerland': keep "
        "stations inside the Swiss national border (Natural Earth "
        "admin_0_countries, ADM0_A3 == 'CHE'; requires cartopy).",
    )
    p.add_argument(
        "--domain-bbox",
        type=float,
        nargs=4,
        default=[45.7, 48.0, 5.8, 10.8],
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Only used when --station-filter-mode=domain.",
    )
    p.add_argument(
        "--icon-grid-file",
        required=True,
        help="ICON grid NetCDF (clat/clon in radians); the same file as NudgeTowardObservation's icon_grid_file.",
    )
    p.add_argument(
        "--dem-barrier-file",
        required=True,
        help="DEM NetCDF on the LV95 grid (variable DEM_1000M, coordinates x/y) used for the barrier term.",
    )
    # Barrier-aware distance hyperparameters. They are fixed in the cache when
    # it is built; NudgeTowardObservation only reads the resulting d_eff values
    # and cannot change them.
    p.add_argument(
        "--n-barrier-samples",
        type=int,
        default=50,
        help="DEM sample points along the straight-line path (endpoints excluded).",
    )
    p.add_argument(
        "--n-barrier-width-samples",
        type=int,
        default=3,
        help="Perpendicular samples per step (odd = centred; 1 = straight line only).",
    )
    p.add_argument(
        "--barrier-width",
        type=float,
        default=1500.0,
        help="Half-width of the perpendicular corridor [m].",
    )
    p.add_argument(
        "--elev-scale",
        type=float,
        default=50.0,
        help="m/km: ridge height above both endpoints that adds 1 km to effective distance.",
    )
    p.add_argument(
        "--elev-diff-scale",
        type=float,
        default=100.0,
        help="m/km: endpoint elevation difference that adds 1 km.",
    )
    p.add_argument(
        "--max-dist",
        type=float,
        default=50000.0,
        help="Station influence radius [m]; pairs farther apart keep their Euclidean "
        "distance. Same unit as NudgeTowardObservation's max_dist.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the cache NetCDF and its .meta.json sidecar. The file "
        "name is built from the station filter, the barrier hyperparameters and "
        "the station count.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a matching cache file (same cache key) already exists.",
    )
    return p.parse_args(argv)


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
    elev_scale: float = 50,
    elev_diff_scale: float = 100,
    n_barrier_width_samples: int = 3,
    barrier_width: float = 1500.0,
) -> np.ndarray:
    """Replace Euclidean distances with elevation-aware distances.

    For each (POI, station) pair with d_euc < max_dist:
        d_eff = sqrt(d_euc² + (barrier / elev_scale)² + (elev_diff / elev_diff_scale)²)

    d_euc and max_dist are in km, elev_scale and elev_diff_scale in m/km,
    barrier_width in m (LV95).

    Barrier term: n_samples points along the straight path (endpoints
    excluded), each the Gaussian-weighted mean (sigma = barrier_width / 2) of
    n_barrier_width_samples DEM values across a ±barrier_width corridor
    perpendicular to the path. The barrier is the 95th percentile of these
    values above the higher endpoint:
        barrier = max(0, percentile_95(elev_path) − max(elev_poi, elev_sta))

    Elevation term: elev_diff = |elev_poi − elev_sta| penalises pairs at
    different altitudes even without a ridge in between.

    sta_elev is the station elevation from the station catalog.
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
    path_len = np.sqrt(dx**2 + dy**2)
    safe_len = np.where(path_len > 0, path_len, 1.0)  # avoid /0 for co-located pairs
    perp_x = -dy / safe_len
    perp_y = dx / safe_len

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
    n_perp = n_barrier_width_samples
    n_close = len(pi_idx)
    elev_slab = dem_rgi(np.c_[y_slab.ravel(), x_slab.ravel()]).reshape(n_samples, n_perp, n_close)

    elev_mean_cross = (elev_slab * gauss_w[None, :, None]).sum(axis=1)

    barrier = np.maximum(0.0, np.percentile(elev_mean_cross, 95, axis=0) - ref_elev).astype(np.float32)

    elev_diff = np.abs(elev_poi - elev_sta).astype(np.float32)

    d_eff = d_euc.copy()
    d_eff[pi_idx, si_idx] = np.sqrt(
        d_euc[pi_idx, si_idx] ** 2 + (barrier / elev_scale) ** 2 + (elev_diff / elev_diff_scale) ** 2
    ).astype(np.float32)

    n_blocked = int((d_eff[pi_idx, si_idx] >= max_dist).sum())
    LOG.debug(
        "barrier_distances: %d close pairs → %d newly blocked by barrier+elev_diff (%.1f%%)",
        len(pi_idx),
        n_blocked,
        100.0 * n_blocked / max(len(pi_idx), 1),
    )
    return d_eff


def load_icon_grid(icon_grid_file: str) -> tuple[np.ndarray, np.ndarray]:
    ds_grid = xr.open_dataset(icon_grid_file)
    lat_icon = np.degrees(ds_grid["clat"]).values.ravel()
    lon_icon = np.degrees(ds_grid["clon"]).values.ravel()
    LOG.info("ICON grid loaded: %d cells from %s", len(lat_icon), icon_grid_file)
    return lat_icon, lon_icon


def load_dem(dem_barrier_file: str) -> tuple[RegularGridInterpolator, Transformer]:
    dem_ds = xr.open_dataset(dem_barrier_file)
    # Replace NaN (ocean / no-data) with 0 m so out-of-domain path segments
    # don't produce NaN barriers.
    dem_z = np.where(np.isnan(dem_ds["DEM_1000M"].values), 0.0, dem_ds["DEM_1000M"].values)
    # RGI axes must match the DEM array layout: first axis = y (northing,
    # rows), second = x (easting, cols).
    dem_rgi = RegularGridInterpolator(
        (dem_ds["y"].values, dem_ds["x"].values),
        dem_z,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    # always_xy=True: input order is (longitude, latitude) -> output is (easting, northing).
    wgs84_to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    LOG.info("DEM loaded: shape=%s from %s", dem_ds["DEM_1000M"].shape, dem_barrier_file)
    return dem_rgi, wgs84_to_lv95


def fetch_station_catalog(
    jretrieve_src_path: str,
    stations_bbox: list[float],
    dwh_params: list[str],
    seq_type: str,
) -> pd.DataFrame:
    if jretrieve_src_path not in sys.path:
        sys.path.insert(0, jretrieve_src_path)
    from data_input import jretrieve as jr

    jr.check_prerequisites()

    stations_sel = {"bbox": list(stations_bbox)}
    # Station catalog (nat_abbr, lat, lon, elevation, ...) of the stations
    # reporting dwh_params inside stations_sel.
    meta = jr.fetch_meta(stations=stations_sel, params=dwh_params, seq_type=seq_type)
    catalog = jr.StationCatalog.from_meta(meta)

    stations = pd.DataFrame(
        {
            "latitude": catalog.latitude,
            "longitude": catalog.longitude,
            "elevation": catalog.elevation,
        },
        index=pd.Index(catalog.nat_abbr, name="station"),
    )
    LOG.info("Retrieved %d stations from jretrieve (domain=%s).", len(stations), stations_sel)
    return stations


def trim_stations(
    stations: pd.DataFrame,
    mode: str,
    domain_bbox: list[float],
) -> pd.DataFrame:
    requested_mode = mode
    if mode == "switzerland":
        try:
            import cartopy.io.shapereader  # noqa: F401
        except ImportError:
            LOG.warning(
                "cartopy is not installed — cannot filter to the Swiss national "
                "border; falling back to station_filter_mode='domain'."
            )
            mode = "domain"

    if mode == "domain":
        lat_min, lat_max, lon_min, lon_max = domain_bbox
        mask = (
            (stations["latitude"] >= lat_min)
            & (stations["latitude"] <= lat_max)
            & (stations["longitude"] >= lon_min)
            & (stations["longitude"] <= lon_max)
        )
        desc = (
            f"domain bbox {domain_bbox}"
            if requested_mode == "domain"
            else f"domain bbox {domain_bbox} (cartopy unavailable, fell back from 'switzerland')"
        )

    elif mode == "switzerland":
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point

        shp_path = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
        ch_country = next(r for r in shpreader.Reader(shp_path).records() if r.attributes["ADM0_A3"] == "CHE")
        swiss_geom = ch_country.geometry

        def _in_switzerland(lat, lon):
            if pd.isna(lat) or pd.isna(lon):
                return False
            return swiss_geom.contains(Point(lon, lat))

        mask = [_in_switzerland(lat, lon) for lat, lon in zip(stations["latitude"], stations["longitude"])]
        desc = "Swiss national border (Natural Earth)"

    else:
        raise ValueError(f"Unknown station_filter_mode: {mode!r} (expected 'domain' or 'switzerland')")

    n_before = len(stations)
    stations = stations[mask]
    LOG.info("Station filter [%s]: %d -> %d stations", desc, n_before, len(stations))
    return stations


def station_dataset(stations: pd.DataFrame, wgs84_to_lv95: Transformer) -> xr.Dataset:
    """Position and elevation of each cached station, along the ``sta`` dim of
    ``d_eff_poi``/``d_eff_sta``: longitude/latitude [deg], LV95 x/y [m] and
    elevation [m] from the station catalog."""
    lon = stations["longitude"].to_numpy()
    lat = stations["latitude"].to_numpy()
    x, y = wgs84_to_lv95.transform(lon, lat)
    return xr.Dataset(
        {
            "station_longitude": ("sta", lon, {"units": "degrees_east"}),
            "station_latitude": ("sta", lat, {"units": "degrees_north"}),
            "station_x": ("sta", x, {"units": "m", "long_name": "LV95 easting (EPSG:2056)"}),
            "station_y": ("sta", y, {"units": "m", "long_name": "LV95 northing (EPSG:2056)"}),
            "station_elevation": ("sta", stations["elevation"].to_numpy(dtype=float), {"units": "m"}),
        },
        coords={"sta": stations.index.tolist()},
    )


def cache_key(
    dem_barrier_file: str,
    icon_grid_file: str,
    stations: pd.DataFrame,
    n_barrier_samples: int,
    n_barrier_width_samples: int,
    barrier_width_m: float,
    elev_scale_km: float,
    elev_diff_scale_km: float,
    max_dist_km: float,
) -> str:
    """Hash of everything d_eff depends on: DEM/grid file identity, the
    trimmed station catalog's positions, and the barrier hyperparameters. Any
    change to one of these must invalidate the cache."""
    dem_stat = Path(dem_barrier_file).stat()
    grid_stat = Path(icon_grid_file).stat()
    sta_identity = stations[["latitude", "longitude", "elevation"]].sort_index().round(6).to_csv()

    payload = {
        "dem_file": str(dem_barrier_file),
        "dem_mtime": dem_stat.st_mtime,
        "dem_size": dem_stat.st_size,
        "grid_file": str(icon_grid_file),
        "grid_mtime": grid_stat.st_mtime,
        "grid_size": grid_stat.st_size,
        "stations": sta_identity,
        "N_BARRIER_SAMPLES": n_barrier_samples,
        "N_BARRIER_WIDTH_SAMPLES": n_barrier_width_samples,
        "BARRIER_WIDTH_M": barrier_width_m,
        "ELEV_SCALE_KM": elev_scale_km,
        "ELEV_DIFF_SCALE_KM": elev_diff_scale_km,
        "MAX_DIST_KM": max_dist_km,
    }
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()


def cache_file_path(
    output_dir: str,
    mode: str,
    max_dist_km: float,
    n_barrier_samples: int,
    n_barrier_width_samples: int,
    barrier_width_m: float,
    elev_scale_km: float,
    elev_diff_scale_km: float,
    n_stations: int,
) -> Path:
    """Cache file name built from the station filter, the barrier
    hyperparameters and the station count, so different configurations
    never share a file."""
    return Path(output_dir) / (
        f"d_eff_cache_{mode}"
        f"_maxdist{max_dist_km:g}km"
        f"_nbar{n_barrier_samples}x{n_barrier_width_samples}"
        f"_bw{barrier_width_m:g}m"
        f"_elev{elev_scale_km:g}"
        f"_elevdiff{elev_diff_scale_km:g}"
        f"_nsta{n_stations}.nc"
    )


def build_d_eff(
    stations: pd.DataFrame,
    lat_icon: np.ndarray,
    lon_icon: np.ndarray,
    dem_rgi: RegularGridInterpolator,
    wgs84_to_lv95: Transformer,
    max_dist_km: float,
    n_barrier_samples: int,
    n_barrier_width_samples: int,
    barrier_width_m: float,
    elev_scale_km: float,
    elev_diff_scale_km: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    icon_x_km, icon_y_km = wgs84_to_lv95.transform(lon_icon, lat_icon)
    icon_grid_xy_km = np.c_[icon_x_km, icon_y_km] / 1000.0  # (n_cells, 2), km

    sta_ids = stations.index.tolist()
    st_lat = stations["latitude"].to_numpy()
    st_lon = stations["longitude"].to_numpy()
    st_elev = stations["elevation"].to_numpy()
    sta_x, sta_y = wgs84_to_lv95.transform(st_lon, st_lat)
    sta_xy = np.c_[sta_x, sta_y] / 1000.0  # (n_sta, 2), km

    # POI domain: every ICON cell within max_dist_km of any (trimmed) station.
    x_min, x_max = sta_xy[:, 0].min() - max_dist_km, sta_xy[:, 0].max() + max_dist_km
    y_min, y_max = sta_xy[:, 1].min() - max_dist_km, sta_xy[:, 1].max() + max_dist_km
    dom_mask = (
        (icon_grid_xy_km[:, 0] >= x_min)
        & (icon_grid_xy_km[:, 0] <= x_max)
        & (icon_grid_xy_km[:, 1] >= y_min)
        & (icon_grid_xy_km[:, 1] <= y_max)
    )
    dom_idx = np.where(dom_mask)[0]
    poi_xy = icon_grid_xy_km[dom_idx]
    n_sta = len(sta_ids)

    # ── POI <-> station ──────────────────────────────────────────────────
    d_euc_poi = np.sqrt(((poi_xy[:, None, :] - sta_xy[None, :, :]) ** 2).sum(axis=-1)).astype(np.float32)
    d_eff_poi = np.empty_like(d_euc_poi)
    for start in range(0, n_sta, PROGRESS_EVERY):
        end = min(start + PROGRESS_EVERY, n_sta)
        d_eff_poi[:, start:end] = barrier_distances(
            lon_icon[dom_idx],
            lat_icon[dom_idx],
            st_lon[start:end],
            st_lat[start:end],
            d_euc_poi[:, start:end],
            max_dist_km,
            st_elev[start:end],
            dem_rgi,
            wgs84_to_lv95,
            n_samples=n_barrier_samples,
            elev_scale=elev_scale_km,
            elev_diff_scale=elev_diff_scale_km,
            n_barrier_width_samples=n_barrier_width_samples,
            barrier_width=barrier_width_m,
        )
        LOG.info("d_eff_poi: %d/%d stations processed", end, n_sta)
    d_eff_poi_full = xr.DataArray(
        d_eff_poi,
        dims=["poi", "sta"],
        coords={"poi": dom_idx, "sta": sta_ids},
        name="d_eff_poi",
    )

    # ── Station <-> station (for the leave-one-out reliability check) ───────
    d_euc_sta = np.sqrt(((sta_xy[:, None, :] - sta_xy[None, :, :]) ** 2).sum(axis=-1)).astype(np.float32)
    np.fill_diagonal(d_euc_sta, np.inf)  # a station is never its own neighbour
    d_eff_sta = np.empty_like(d_euc_sta)
    for start in range(0, n_sta, PROGRESS_EVERY):
        end = min(start + PROGRESS_EVERY, n_sta)
        # The batch of stations is the "poi" side (rows); the "sta" side
        # (columns, and sta_elev) is always the full station set.
        d_eff_sta[start:end, :] = barrier_distances(
            st_lon[start:end],
            st_lat[start:end],
            st_lon,
            st_lat,
            d_euc_sta[start:end, :],
            max_dist_km,
            st_elev,
            dem_rgi,
            wgs84_to_lv95,
            n_samples=n_barrier_samples,
            elev_scale=elev_scale_km,
            elev_diff_scale=elev_diff_scale_km,
            n_barrier_width_samples=n_barrier_width_samples,
            barrier_width=barrier_width_m,
        )
        LOG.info("d_eff_sta: %d/%d stations processed", end, n_sta)
    # Row dim "sta_i" rather than "poi": its coordinate (station IDs, str) must
    # not clash with d_eff_poi's "poi" coordinate (ICON cell indices, int) in
    # the same Dataset.
    d_eff_sta_full = xr.DataArray(
        d_eff_sta,
        dims=["sta_i", "sta"],
        coords={"sta_i": sta_ids, "sta": sta_ids},
        name="d_eff_sta",
    )

    return d_eff_poi_full, d_eff_sta_full


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    # --max-dist is in m; everything below works in km.
    max_dist_km = args.max_dist / 1000.0

    LOG.info(
        "station_filter_mode=%r max_dist_km=%g elev_scale_km=%g elev_diff_scale_km=%g "
        "n_barrier_samples=%d n_barrier_width_samples=%d barrier_width_m=%g",
        args.station_filter_mode,
        max_dist_km,
        args.elev_scale,
        args.elev_diff_scale,
        args.n_barrier_samples,
        args.n_barrier_width_samples,
        args.barrier_width,
    )
    LOG.info("Output directory: %s", args.output_dir)

    lat_icon, lon_icon = load_icon_grid(args.icon_grid_file)
    dem_rgi, wgs84_to_lv95 = load_dem(args.dem_barrier_file)

    stations = fetch_station_catalog(
        args.jretrieve_src_path,
        args.stations_bbox,
        args.dwh_params,
        args.seq_type,
    )
    stations = trim_stations(stations, args.station_filter_mode, args.domain_bbox)

    out_file = cache_file_path(
        args.output_dir,
        args.station_filter_mode,
        max_dist_km,
        args.n_barrier_samples,
        args.n_barrier_width_samples,
        args.barrier_width,
        args.elev_scale,
        args.elev_diff_scale,
        len(stations),
    )
    meta_file = out_file.with_suffix(".meta.json")
    LOG.info("Output: %s", out_file)

    key = cache_key(
        args.dem_barrier_file,
        args.icon_grid_file,
        stations,
        args.n_barrier_samples,
        args.n_barrier_width_samples,
        args.barrier_width,
        args.elev_scale,
        args.elev_diff_scale,
        max_dist_km,
    )
    existing = xr.open_dataset(out_file) if out_file.exists() else None
    cache_hit = not args.force and existing is not None and existing.attrs.get("cache_key") == key

    if cache_hit:
        d_eff_poi_full = existing["d_eff_poi"].load()
        d_eff_sta_full = existing["d_eff_sta"].load()
        meta = dict(existing.attrs)
        has_station_vars = "station_x" in existing
        existing.close()
        LOG.info(
            "d_eff cache HIT (%s): loaded POI x station %s and station x station %s — barrier_distances() skipped.",
            out_file,
            d_eff_poi_full.shape,
            d_eff_sta_full.shape,
        )
        if not has_station_vars:
            # Same cache key, hence the same stations: add the station
            # positions and elevations if the file does not contain them.
            station_dataset(stations, wgs84_to_lv95).to_netcdf(out_file, mode="a")
            LOG.info("Added station positions and elevations to %s", out_file)
    else:
        if existing is not None:
            existing.close()
        LOG.info(
            "d_eff cache MISS%s — computing full d_eff matrices...",
            " (--force)"
            if args.force
            else " (missing, or DEM/grid/station-catalog/hyperparameters changed since it was built)",
        )

        d_eff_poi_full, d_eff_sta_full = build_d_eff(
            stations,
            lat_icon,
            lon_icon,
            dem_rgi,
            wgs84_to_lv95,
            max_dist_km,
            args.n_barrier_samples,
            args.n_barrier_width_samples,
            args.barrier_width,
            args.elev_scale,
            args.elev_diff_scale,
        )

        out_ds = xr.merge(
            [
                xr.Dataset({"d_eff_poi": d_eff_poi_full, "d_eff_sta": d_eff_sta_full}),
                station_dataset(stations, wgs84_to_lv95),
            ]
        )
        out_ds.attrs["cache_key"] = key
        out_ds.attrs["station_filter_mode"] = args.station_filter_mode
        out_ds.attrs["N_BARRIER_SAMPLES"] = args.n_barrier_samples
        out_ds.attrs["N_BARRIER_WIDTH_SAMPLES"] = args.n_barrier_width_samples
        out_ds.attrs["BARRIER_WIDTH"] = args.barrier_width
        out_ds.attrs["ELEV_SCALE"] = args.elev_scale
        out_ds.attrs["ELEV_DIFF_SCALE"] = args.elev_diff_scale
        out_ds.attrs["MAX_DIST_KM"] = max_dist_km
        out_ds.attrs["n_stations"] = len(stations)
        meta = dict(out_ds.attrs)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(out_file)
        LOG.info(
            "d_eff cache written to %s: POI x station %s, station x station %s, n_stations=%d",
            out_file,
            d_eff_poi_full.shape,
            d_eff_sta_full.shape,
            len(stations),
        )

    # Sidecar JSON with the same metadata, for quick inspection without loading xarray.
    meta_file.write_text(json.dumps(meta, indent=2, default=str))
    LOG.info("Metadata written to %s", meta_file)
    for k, v in meta.items():
        LOG.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
