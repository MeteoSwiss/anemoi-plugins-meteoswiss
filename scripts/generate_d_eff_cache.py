"""Generate the barrier- and elevation-aware effective-distance (``d_eff``)
cache consumed by ``NudgeTowardObservation``'s ``d_eff_file`` parameter
(``anemoi_plugins_meteoswiss/transform/filters/nudging.py``).

Converted from ``notebooks/d_eff_generator.ipynb`` (see that repo's git
history for the original) so it can run headlessly, e.g. from
``ci/d_eff_generator.yaml``.

Steps
-----
1.  Retrieve the station catalog via ``jretrieve`` — metadata only, no
    ``ref_time`` and no actual observation values. ``barrier_distances()`` is
    a pure function of station/POI geometry and the DEM; it never depends on
    a specific timestamp or observed value, so ``jretrieve.fetch_meta()``
    alone (no ``fetch_data()``) is enough.
2.  Trim the retrieved stations to either a bounding box **or** the real
    Swiss national border (``--station-filter-mode``).
3.  Compute ``d_eff_poi`` (POI<->station) and ``d_eff_sta``
    (station<->station) via the production ``barrier_distances()`` (imported
    directly from ``nudging.py``, not re-implemented here, so this is
    guaranteed to match what ``NudgeTowardObservation`` itself uses offline).
4.  Write both to a NetCDF file, with the barrier hyperparameters and the
    station count attached as metadata — both as NetCDF attrs (so
    ``xr.open_dataset(...).attrs`` works) and as a JSON sidecar file (so they
    can be checked without loading xarray at all).

Skips everything specific to running the nudging algorithm itself (no GRIB
loading, no ``ned_interp``, no topographic descriptors, no reliability check,
no diagnostic plots beyond the optional station map) — this script's only job
is producing the cache file.
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

from anemoi_plugins_meteoswiss.transform.filters.nudging import barrier_distances

LOG = logging.getLogger(__name__)

# Same DWH params the production observation-retrieval filter queries —
# fetch_meta() only returns stations that report these, so keep this in sync
# with whatever RetrieveObservation is actually configured with, or the
# station catalog here may not match what nudging.py sees at run time.
DEFAULT_DWH_PARAMS = [
    "tre200s0", "tde200s0", "pp0qffs0", "prestas0", "rre150h0", "fkl010z0", "dkl010z0", "fkl010z1",
]
DEFAULT_JRETRIEVE_SRC_PATH = "/scratch/mch/llanzila/sruc/evalml/src"
DEFAULT_ICON_GRID_FILE = "/scratch/mch/llanzila/sruc/aux_files/icon_grid_0001_R19B08_mch.nc"
DEFAULT_DEM_BARRIER_FILE = (
    "/store_new/mch/msclim/appclim/data/grids/topodem/v2/topo/radar_100/topo_DEM_1000M.nc"
)
DEFAULT_OUTPUT_DIR = "/scratch/mch/llanzila/sruc/aux_files"

# Batch size for barrier_distances() calls: processes stations in chunks so
# progress is visible in the logs — otherwise a single call covering every
# station at once gives no feedback until it's entirely done.
PROGRESS_EVERY = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--jretrieve-src-path", default=DEFAULT_JRETRIEVE_SRC_PATH,
        help="Directory containing the 'data_input' package (data_input/jretrieve.py).",
    )
    p.add_argument("--dwh-params", nargs="+", default=DEFAULT_DWH_PARAMS)
    p.add_argument("--seq-type", default="surface")
    p.add_argument(
        "--stations-bbox", type=float, nargs=4, default=[40.5, 53.0, 0.0, 17.5],
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Broad domain passed to jretrieve.fetch_meta() — this is metadata "
        "only, so retrieving broadly here is cheap; trimmed down below.",
    )
    p.add_argument(
        "--station-filter-mode", choices=["domain", "switzerland"], default="domain",
        help="'domain': keep stations inside --domain-bbox. 'switzerland': keep "
        "stations inside the real Swiss national border (Natural Earth "
        "admin_0_countries, ADM0_A3 == 'CHE').",
    )
    p.add_argument(
        "--domain-bbox", type=float, nargs=4, default=[45.7, 48.0, 5.8, 10.8],
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Only used when --station-filter-mode=domain.",
    )
    p.add_argument("--icon-grid-file", default=DEFAULT_ICON_GRID_FILE)
    p.add_argument("--dem-barrier-file", default=DEFAULT_DEM_BARRIER_FILE)
    # Barrier-aware distance hyperparameters — baked into the cache at build
    # time, NOT read back by consumers, and no longer configurable on
    # NudgeTowardObservation itself (see its d_eff_file parameter). Defaults
    # below match the notebook's own "currently active" configuration; the
    # production configs' d_eff_file currently points at a different combo
    # (max-dist-km=50, elev-scale-km=50, elev-diff-scale-km=100 — the
    # 'd_eff_5' cache) — override below to reproduce that one instead.
    p.add_argument(
        "--n-barrier-samples", type=int, default=75,
        help="DEM sample points along the straight-line path (endpoints excluded).",
    )
    p.add_argument(
        "--n-barrier-width-samples", type=int, default=3,
        help="Perpendicular samples per step (odd = centred; 1 = straight line only).",
    )
    p.add_argument(
        "--barrier-width-m", type=float, default=1500.0,
        help="Half-width of the perpendicular corridor [m].",
    )
    p.add_argument(
        "--elev-scale-km", type=float, default=150.0,
        help="m/km: ridge height above both endpoints that adds 1 km to effective distance.",
    )
    p.add_argument(
        "--elev-diff-scale-km", type=float, default=300.0,
        help="m/km: endpoint elevation difference that adds 1 km.",
    )
    p.add_argument(
        "--max-dist-km", type=float, default=75.0,
        help="km: station influence radius — barrier-distance cutoff.",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory the cache NetCDF (and its .meta.json sidecar) is written to. "
        "Only the directory is fixed here — the filename is derived from the "
        "metadata (station filter, barrier hyperparameters, station count) once "
        "the station catalog is retrieved and trimmed, so the file name itself "
        "tells you what's inside without opening it.",
    )
    p.add_argument(
        "--plot-out", default=None,
        help="If given, save a PNG map of the stations considered for d_eff to this "
        "path (matplotlib Agg backend — safe in a headless/batch job). Skipped "
        "by default.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Recompute even if a matching cache file (same cache key) already exists.",
    )
    return p.parse_args(argv)


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
        dem_z, method="linear", bounds_error=False, fill_value=0.0,
    )
    # always_xy=True: input order is (longitude, latitude) -> output is (easting, northing).
    wgs84_to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    LOG.info("DEM loaded: shape=%s from %s", dem_ds["DEM_1000M"].shape, dem_barrier_file)
    return dem_rgi, wgs84_to_lv95


def fetch_station_catalog(
    jretrieve_src_path: str, stations_bbox: list[float], dwh_params: list[str], seq_type: str,
) -> pd.DataFrame:
    if jretrieve_src_path not in sys.path:
        sys.path.insert(0, jretrieve_src_path)
    from data_input import jretrieve as jr

    jr.check_prerequisites()

    stations_sel = {"bbox": list(stations_bbox)}
    # fetch_meta() returns the station catalog (nat_abbr, lat, lon, elevation,
    # ...) for whichever stations report dwh_params within stations_sel. No
    # fetch_data() call, and no ref_time: station geometry doesn't depend on
    # either.
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
    stations: pd.DataFrame, mode: str, domain_bbox: list[float],
) -> pd.DataFrame:
    if mode == "domain":
        lat_min, lat_max, lon_min, lon_max = domain_bbox
        mask = (
            (stations["latitude"] >= lat_min) & (stations["latitude"] <= lat_max) &
            (stations["longitude"] >= lon_min) & (stations["longitude"] <= lon_max)
        )
        desc = f"domain bbox {domain_bbox}"

    elif mode == "switzerland":
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point

        shp_path = shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"
        )
        ch_country = next(
            r for r in shpreader.Reader(shp_path).records()
            if r.attributes["ADM0_A3"] == "CHE"
        )
        swiss_geom = ch_country.geometry

        def _in_switzerland(lat, lon):
            if pd.isna(lat) or pd.isna(lon):
                return False
            return swiss_geom.contains(Point(lon, lat))

        mask = [
            _in_switzerland(lat, lon)
            for lat, lon in zip(stations["latitude"], stations["longitude"])
        ]
        desc = "Swiss national border (Natural Earth)"

    else:
        raise ValueError(f"Unknown station_filter_mode: {mode!r} (expected 'domain' or 'switzerland')")

    n_before = len(stations)
    stations = stations[mask]
    LOG.info("Station filter [%s]: %d -> %d stations", desc, n_before, len(stations))
    return stations


def save_station_plot(
    stations: pd.DataFrame, mode: str, domain_bbox: list[float], out_path: str,
) -> None:
    """Save a PNG map of the stations considered for d_eff. Never raises: a
    plotting failure (missing matplotlib/cartopy, bad data, ...) is logged
    and skipped, since it never affects the cache itself."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe in headless/batch jobs
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        pad = 0.4
        extent = [
            stations["longitude"].min() - pad, stations["longitude"].max() + pad,
            stations["latitude"].min() - pad, stations["latitude"].max() + pad,
        ]

        fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f0", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#c8dff0", alpha=0.6, zorder=0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="#444444", zorder=3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.9, zorder=3)
        ax.add_feature(cfeature.LAKES, facecolor="#a8cce0", alpha=0.7, zorder=2)

        ax.scatter(
            stations["longitude"], stations["latitude"],
            s=20, c="#d62728", marker="o", edgecolors="black", linewidths=0.3,
            transform=ccrs.PlateCarree(), zorder=4,
        )

        if mode == "domain":
            lat_min, lat_max, lon_min, lon_max = domain_bbox
            ax.add_patch(mpatches.Rectangle(
                (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                fill=False, edgecolor="#1565C0", linewidth=1.5, linestyle="--",
                transform=ccrs.PlateCarree(), zorder=5,
            ))

        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        ax.set_title(f"Stations considered for d_eff (mode={mode!r}, n={len(stations)})")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved station map to %s", out_path)
    except Exception:
        LOG.exception("Station map plotting failed; continuing without it.")


def cache_key(
    dem_barrier_file: str, icon_grid_file: str, stations: pd.DataFrame,
    n_barrier_samples: int, n_barrier_width_samples: int, barrier_width_m: float,
    elev_scale_km: float, elev_diff_scale_km: float, max_dist_km: float,
) -> str:
    """Hash of everything d_eff depends on: DEM/grid file identity, the
    trimmed station catalog's positions, and the barrier hyperparameters. Any
    change to one of these must invalidate the cache."""
    dem_stat = Path(dem_barrier_file).stat()
    grid_stat = Path(icon_grid_file).stat()
    sta_identity = stations[["latitude", "longitude", "elevation"]].sort_index().round(6).to_csv()

    payload = {
        "dem_file": str(dem_barrier_file), "dem_mtime": dem_stat.st_mtime, "dem_size": dem_stat.st_size,
        "grid_file": str(icon_grid_file), "grid_mtime": grid_stat.st_mtime, "grid_size": grid_stat.st_size,
        "stations": sta_identity,
        "N_BARRIER_SAMPLES": n_barrier_samples, "N_BARRIER_WIDTH_SAMPLES": n_barrier_width_samples,
        "BARRIER_WIDTH_M": barrier_width_m, "ELEV_SCALE_KM": elev_scale_km,
        "ELEV_DIFF_SCALE_KM": elev_diff_scale_km, "MAX_DIST_KM": max_dist_km,
    }
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()


def cache_file_path(
    output_dir: str, mode: str, max_dist_km: float, n_barrier_samples: int,
    n_barrier_width_samples: int, barrier_width_m: float, elev_scale_km: float,
    elev_diff_scale_km: float, n_stations: int,
) -> Path:
    """Filename encodes the metadata that actually distinguishes one cache
    from another (station filter/domain, barrier hyperparameters, station
    count) — so two different configurations never collide on the same file,
    and you can tell what a given file contains without opening it."""
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
    lat_icon: np.ndarray, lon_icon: np.ndarray,
    dem_rgi: RegularGridInterpolator, wgs84_to_lv95: Transformer,
    max_dist_km: float, n_barrier_samples: int, n_barrier_width_samples: int,
    barrier_width_m: float, elev_scale_km: float, elev_diff_scale_km: float,
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
        (icon_grid_xy_km[:, 0] >= x_min) & (icon_grid_xy_km[:, 0] <= x_max) &
        (icon_grid_xy_km[:, 1] >= y_min) & (icon_grid_xy_km[:, 1] <= y_max)
    )
    dom_idx = np.where(dom_mask)[0]
    poi_xy = icon_grid_xy_km[dom_idx]
    n_sta = len(sta_ids)

    # ── POI <-> station ──────────────────────────────────────────────────
    d_euc_poi = np.sqrt(
        ((poi_xy[:, None, :] - sta_xy[None, :, :]) ** 2).sum(axis=-1)
    ).astype(np.float32)
    d_eff_poi = np.empty_like(d_euc_poi)
    for start in range(0, n_sta, PROGRESS_EVERY):
        end = min(start + PROGRESS_EVERY, n_sta)
        d_eff_poi[:, start:end] = barrier_distances(
            lon_icon[dom_idx], lat_icon[dom_idx],
            st_lon[start:end], st_lat[start:end],
            d_euc_poi[:, start:end], max_dist_km, st_elev[start:end],
            dem_rgi, wgs84_to_lv95,
            n_samples=n_barrier_samples,
            elev_scale=elev_scale_km,
            elev_diff_scale=elev_diff_scale_km,
            n_barrier_width_samples=n_barrier_width_samples,
            barrier_width_m=barrier_width_m,
        )
        LOG.info("d_eff_poi: %d/%d stations processed", end, n_sta)
    d_eff_poi_full = xr.DataArray(
        d_eff_poi, dims=["poi", "sta"],
        coords={"poi": dom_idx, "sta": sta_ids}, name="d_eff_poi",
    )

    # ── Station <-> station (for compute_reliability's leave-one-out check) ──
    d_euc_sta = np.sqrt(
        ((sta_xy[:, None, :] - sta_xy[None, :, :]) ** 2).sum(axis=-1)
    ).astype(np.float32)
    np.fill_diagonal(d_euc_sta, np.inf)  # a station is never its own neighbour
    d_eff_sta = np.empty_like(d_euc_sta)
    for start in range(0, n_sta, PROGRESS_EVERY):
        end = min(start + PROGRESS_EVERY, n_sta)
        # This batch of stations plays the "poi" role (rows); the "sta" role
        # (columns, and therefore sta_elev) stays the FULL station set — only
        # the row side is chunked.
        d_eff_sta[start:end, :] = barrier_distances(
            st_lon[start:end], st_lat[start:end],
            st_lon, st_lat,
            d_euc_sta[start:end, :], max_dist_km, st_elev,
            dem_rgi, wgs84_to_lv95,
            n_samples=n_barrier_samples,
            elev_scale=elev_scale_km,
            elev_diff_scale=elev_diff_scale_km,
            n_barrier_width_samples=n_barrier_width_samples,
            barrier_width_m=barrier_width_m,
        )
        LOG.info("d_eff_sta: %d/%d stations processed", end, n_sta)
    # dim named "sta_i" (not "poi"): keeps this matrix's coordinate (station
    # IDs, str) from colliding with d_eff_poi_full's "poi" coordinate (ICON
    # cell indices, int) when both are saved into the same on-disk Dataset.
    d_eff_sta_full = xr.DataArray(
        d_eff_sta, dims=["sta_i", "sta"],
        coords={"sta_i": sta_ids, "sta": sta_ids}, name="d_eff_sta",
    )

    return d_eff_poi_full, d_eff_sta_full


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    LOG.info(
        "station_filter_mode=%r max_dist_km=%g elev_scale_km=%g elev_diff_scale_km=%g "
        "n_barrier_samples=%d n_barrier_width_samples=%d barrier_width_m=%g",
        args.station_filter_mode, args.max_dist_km, args.elev_scale_km, args.elev_diff_scale_km,
        args.n_barrier_samples, args.n_barrier_width_samples, args.barrier_width_m,
    )
    LOG.info("Output directory: %s", args.output_dir)

    lat_icon, lon_icon = load_icon_grid(args.icon_grid_file)
    dem_rgi, wgs84_to_lv95 = load_dem(args.dem_barrier_file)

    stations = fetch_station_catalog(
        args.jretrieve_src_path, args.stations_bbox, args.dwh_params, args.seq_type,
    )
    stations = trim_stations(stations, args.station_filter_mode, args.domain_bbox)

    if args.plot_out:
        save_station_plot(stations, args.station_filter_mode, args.domain_bbox, args.plot_out)

    out_file = cache_file_path(
        args.output_dir, args.station_filter_mode, args.max_dist_km, args.n_barrier_samples,
        args.n_barrier_width_samples, args.barrier_width_m, args.elev_scale_km,
        args.elev_diff_scale_km, len(stations),
    )
    meta_file = out_file.with_suffix(".meta.json")
    LOG.info("Output: %s", out_file)

    key = cache_key(
        args.dem_barrier_file, args.icon_grid_file, stations,
        args.n_barrier_samples, args.n_barrier_width_samples, args.barrier_width_m,
        args.elev_scale_km, args.elev_diff_scale_km, args.max_dist_km,
    )
    existing = xr.open_dataset(out_file) if out_file.exists() else None
    cache_hit = not args.force and existing is not None and existing.attrs.get("cache_key") == key

    if cache_hit:
        d_eff_poi_full = existing["d_eff_poi"].load()
        d_eff_sta_full = existing["d_eff_sta"].load()
        meta = dict(existing.attrs)
        existing.close()
        LOG.info(
            "d_eff cache HIT (%s): loaded POI x station %s and station x station %s "
            "— barrier_distances() skipped.",
            out_file, d_eff_poi_full.shape, d_eff_sta_full.shape,
        )
    else:
        if existing is not None:
            existing.close()
        LOG.info(
            "d_eff cache MISS%s — computing full d_eff matrices...",
            " (--force)" if args.force else " (missing, or DEM/grid/station-catalog/"
            "hyperparameters changed since it was built)",
        )

        d_eff_poi_full, d_eff_sta_full = build_d_eff(
            stations, lat_icon, lon_icon, dem_rgi, wgs84_to_lv95,
            args.max_dist_km, args.n_barrier_samples, args.n_barrier_width_samples,
            args.barrier_width_m, args.elev_scale_km, args.elev_diff_scale_km,
        )

        out_ds = xr.Dataset({"d_eff_poi": d_eff_poi_full, "d_eff_sta": d_eff_sta_full})
        out_ds.attrs["cache_key"] = key
        out_ds.attrs["station_filter_mode"] = args.station_filter_mode
        out_ds.attrs["N_BARRIER_SAMPLES"] = args.n_barrier_samples
        out_ds.attrs["N_BARRIER_WIDTH_SAMPLES"] = args.n_barrier_width_samples
        out_ds.attrs["BARRIER_WIDTH_M"] = args.barrier_width_m
        out_ds.attrs["ELEV_SCALE_KM"] = args.elev_scale_km
        out_ds.attrs["ELEV_DIFF_SCALE_KM"] = args.elev_diff_scale_km
        out_ds.attrs["MAX_DIST_KM"] = args.max_dist_km
        out_ds.attrs["n_stations"] = len(stations)
        meta = dict(out_ds.attrs)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(out_file)
        LOG.info(
            "d_eff cache written to %s: POI x station %s, station x station %s, n_stations=%d",
            out_file, d_eff_poi_full.shape, d_eff_sta_full.shape, len(stations),
        )

    # Sidecar JSON with the same metadata, for quick inspection without loading xarray.
    meta_file.write_text(json.dumps(meta, indent=2, default=str))
    LOG.info("Metadata written to %s", meta_file)
    for k, v in meta.items():
        LOG.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
