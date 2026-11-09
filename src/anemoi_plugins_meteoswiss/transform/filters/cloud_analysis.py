"""
CloudAnalysis — SAFNWC CT + ICON cloud cover analysis plugin.

Reads SAFNWC CMA/CT/CTTH satellite data, extracts ICON cloud cover fields
(CLCT, CLCH, CLCM, CLCL) from the input FieldList or from an external GRIB
file, runs the INCA cloud analysis pipeline, and writes CT/CL/CM/CH on the
INCA 1 km grid to a NetCDF output file.

Pipeline (mirrors cloud_extrapolation.py / INCA cloud module):
  1. preprocess_saf  — handle fractional/cirrus/below-topo/lakes
  2. compute_cloud_analysis — classify into total/low/mid/high layers
  3. apply_icon_combination — fill from ICON where SAF is saturated or missing
  4. reproject SAF+ICON → INCA 1 km grid (LV03/CH1903)
  5. write NetCDF

The input FieldList is returned unchanged.
"""

import os

os.environ.setdefault("PROJ_IGNORE_CELESTIAL_BODY", "YES")

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import eccodes
import netCDF4
import numpy as np
import pyproj
from scipy.ndimage import binary_opening, gaussian_filter
from scipy.spatial import cKDTree

import earthkit.data as ekd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

# ── Fixed INCA/CCS4 1 km grid constants ──────────────────────────────────────
# Shape: (640 rows, 710 cols), row 0 = southern edge.
NY_INCA, NX_INCA = 640, 710
X0_LV03, X1_LV03 = 255_000.0, 965_000.0   # west / east  [m, LV03]
Y0_LV03, Y1_LV03 = -160_000.0, 480_000.0  # south / north [m, LV03]
DX = 1_000.0                               # pixel size [m]

_LV03_PROJ4 = (
    "+proj=somerc +lat_0=46.9524055556 +lon_0=7.43958333333 "
    "+k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +units=m"
)

# ECMWF shortName for each ICON cloud variable (overridable in __init__)
_DEFAULT_ICON_SHORTNAMES = {
    "CLCT": "tcc",   # total cloud cover
    "CLCL": "lcc",   # low cloud cover
    "CLCM": "mcc",   # medium cloud cover
    "CLCH": "hcc",   # high cloud cover
}


# ── SAF file discovery ─────────────────────────────────────────────────────────

def _find_saf_file(
    saf_dir: Path,
    timestamp: datetime,
    product: str,
    satellites: list,
    domains: list,
    use_plax: bool,
):
    """Return (path, satellite, domain) for the first matching SAFNWC file."""
    ts_str = f"{timestamp:%Y%m%dT%H%M%SZ}"
    suffixes = ["_PLAX.nc", ".nc"] if use_plax else [".nc"]
    for suffix in suffixes:
        for sat in satellites:
            for dom in domains:
                fn = saf_dir / f"S_NWC_{product}_{sat}_{dom}_{ts_str}{suffix}"
                if fn.exists():
                    return fn, sat, dom
    raise FileNotFoundError(
        f"No {product} SAF file for {timestamp:%Y%m%dT%H%M%SZ} in {saf_dir}"
    )


# ── NetCDF reading with CF scale/offset decoding ──────────────────────────────

def _read_var(ds, varname: str) -> np.ndarray:
    """Read a CF-encoded variable; apply scale/offset; fill → NaN."""
    v = ds[varname]
    v.set_auto_maskandscale(False)
    raw = np.array(v[:], dtype=float)
    fill = getattr(v, "_FillValue", None)
    scale = float(getattr(v, "scale_factor", 1.0))
    offset = float(getattr(v, "add_offset", 0.0))
    fill_mask = (raw == float(fill)) if fill is not None else np.zeros(raw.shape, bool)
    data = raw * scale + offset
    data[fill_mask] = np.nan
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    return data


# ── Reprojection helpers ───────────────────────────────────────────────────────

def _inca_pixel_centers():
    """Return (XX, YY) meshgrid of INCA pixel centres in LV03 [m], row 0 = south."""
    x = np.linspace(X0_LV03, X1_LV03 - DX, NX_INCA) + DX / 2.0
    y = np.linspace(Y0_LV03, Y1_LV03 - DX, NY_INCA) + DX / 2.0
    return np.meshgrid(x, y)


def _saf_pixel_centers(nx_saf, ny_saf, xgeo_ul, ygeo_ul, xgeo_lr, ygeo_lr):
    """Return (X, Y) meshgrid of SAF pixel centres in geostationary CRS, row 0 = north."""
    dx_saf = (xgeo_lr - xgeo_ul) / nx_saf
    dy_saf = (ygeo_lr - ygeo_ul) / ny_saf
    COL, ROW = np.meshgrid(np.arange(nx_saf), np.arange(ny_saf))
    return xgeo_ul + (COL + 0.5) * dx_saf, ygeo_ul + (ROW + 0.5) * dy_saf


def _make_transformers(gdal_proj: str):
    geos_crs = pyproj.CRS.from_proj4(gdal_proj)
    lv03_crs = pyproj.CRS.from_proj4(_LV03_PROJ4)
    return (
        pyproj.Transformer.from_crs(lv03_crs, geos_crs, always_xy=True),
        pyproj.Transformer.from_crs(geos_crs, lv03_crs, always_xy=True),
    )


def _reproject_saf_to_inca(
    saf_array: np.ndarray,
    gdal_proj: str,
    xgeo_ul: float,
    ygeo_ul: float,
    xgeo_lr: float,
    ygeo_lr: float,
    fill=np.nan,
) -> np.ndarray:
    """Nearest-neighbour reproject a SAF field → INCA 1 km grid.

    Input row 0 = north (SAF native).
    Output row 0 = south (INCA convention, row 0 ↔ Y_min).
    """
    dx_saf = (xgeo_lr - xgeo_ul) / saf_array.shape[1]
    dy_saf = (ygeo_lr - ygeo_ul) / saf_array.shape[0]
    XX, YY = _inca_pixel_centers()
    lv03_to_geos, _ = _make_transformers(gdal_proj)
    flat = lv03_to_geos.transform(XX.ravel(), YY.ravel())
    x_g = flat[0].reshape(NY_INCA, NX_INCA)
    y_g = flat[1].reshape(NY_INCA, NX_INCA)
    col_idx = np.round((x_g - xgeo_ul) / dx_saf).astype(int)
    row_idx = np.round((y_g - ygeo_ul) / dy_saf).astype(int)
    valid = (
        np.isfinite(x_g)
        & np.isfinite(y_g)
        & (col_idx >= 0)
        & (col_idx < saf_array.shape[1])
        & (row_idx >= 0)
        & (row_idx < saf_array.shape[0])
    )
    out = np.full((NY_INCA, NX_INCA), fill)
    out[valid] = saf_array[row_idx[valid], col_idx[valid]]
    return out


def _reproject_inca_to_saf(
    inca_array: np.ndarray,
    gdal_proj: str,
    xgeo_ul: float,
    ygeo_ul: float,
    xgeo_lr: float,
    ygeo_lr: float,
    ny_saf: int,
    nx_saf: int,
    fill=np.nan,
) -> np.ndarray:
    """Nearest-neighbour reproject an INCA field → SAF domain.

    Input row 0 = south (INCA convention).
    Output row 0 = north (SAF native).
    """
    x_geos, y_geos = _saf_pixel_centers(nx_saf, ny_saf, xgeo_ul, ygeo_ul, xgeo_lr, ygeo_lr)
    _, geos_to_lv03 = _make_transformers(gdal_proj)
    x_lv03, y_lv03 = geos_to_lv03.transform(x_geos.ravel(), y_geos.ravel())
    x_lv03 = x_lv03.reshape(ny_saf, nx_saf)
    y_lv03 = y_lv03.reshape(ny_saf, nx_saf)
    col_inca = np.round((x_lv03 - X0_LV03) / DX).astype(int)
    row_inca = np.round((y_lv03 - Y0_LV03) / DX).astype(int)
    valid = (
        np.isfinite(x_lv03)
        & (col_inca >= 0)
        & (col_inca < NX_INCA)
        & (row_inca >= 0)
        & (row_inca < NY_INCA)
    )
    out = np.full((ny_saf, nx_saf), fill)
    out[valid] = inca_array[row_inca[valid], col_inca[valid]]
    return out


def _reproject_icon_to_inca(
    icon_array: np.ndarray,
    lat_icon: np.ndarray,
    lon_icon: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour reproject a 1-D ICON unstructured field → INCA 1 km grid."""
    lat_r = np.deg2rad(lat_icon.ravel())
    lon_r = np.deg2rad(lon_icon.ravel())
    xyz_icon = np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )
    tree = cKDTree(xyz_icon)
    lv03_to_wgs84 = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4(_LV03_PROJ4), pyproj.CRS.from_epsg(4326), always_xy=True
    )
    XX, YY = _inca_pixel_centers()
    lon_inca, lat_inca = lv03_to_wgs84.transform(XX.ravel(), YY.ravel())
    lat_r2 = np.deg2rad(lat_inca)
    lon_r2 = np.deg2rad(lon_inca)
    xyz_inca = np.column_stack(
        [np.cos(lat_r2) * np.cos(lon_r2), np.cos(lat_r2) * np.sin(lon_r2), np.sin(lat_r2)]
    )
    max_dist = 3_000.0 / 6_371_000.0   # 3 km on unit sphere
    _, idx = tree.query(xyz_inca, distance_upper_bound=max_dist)
    n_icon = icon_array.ravel().shape[0]
    out = np.full(NY_INCA * NX_INCA, np.nan)
    valid = idx < n_icon
    out[valid] = icon_array.ravel()[idx[valid]]
    return out.reshape(NY_INCA, NX_INCA)


# ── Gaussian filter that respects NaN (mirrors cloud_utils.gaussian_filter_nans) ─

def _gaussian_filter_nans(U: np.ndarray, sigma=5, truncate=4, mode="nearest") -> np.ndarray:
    V = U.copy()
    V[np.isnan(U)] = 0.0
    VV = gaussian_filter(V, sigma=sigma, truncate=truncate, mode=mode)
    W = np.ones_like(U)
    W[np.isnan(U)] = 0.0
    WW = gaussian_filter(W, sigma=sigma, truncate=truncate, mode=mode)
    with np.errstate(invalid="ignore"):
        return VV / WW


# ── SAF preprocessing (mirrors cloud_funcs.preprocess_saf) ───────────────────

def _preprocess_saf(
    cloud_mask: np.ndarray,
    cloud_eff: np.ndarray,
    cloud_top: np.ndarray,
    cloud_type: np.ndarray,
    topo_saf,
    lakes_saf,
    height_tolerance_m: float,
    remove_below_topo: bool,
):
    """In-place preprocessing of raw SAF fields; all arrays on the SAF domain."""
    # Suppress cloud fraction for non-cloud surface types (CT 1-4)
    non_cloud_surface = np.isin(cloud_type, [1, 2, 3, 4])
    n_suppressed = int((non_cloud_surface & (cloud_mask > 0)).sum())
    cloud_mask[non_cloud_surface] = 0.0
    cloud_eff[non_cloud_surface] = 0.0
    cloud_top[non_cloud_surface] = np.nan
    LOG.info(
        "CT 1-4 suppression: cleared %d cloudy pixels (%.2f%% of domain)",
        n_suppressed,
        100.0 * n_suppressed / cloud_type.size,
    )

    fractional_mask = (np.isnan(cloud_eff) | np.isnan(cloud_top)) & (cloud_type == 10)
    thinhighcirrus_mask = (np.isnan(cloud_eff) | np.isnan(cloud_top)) & (cloud_type == 11)
    cirrus_above_mask = (np.isnan(cloud_eff) | np.isnan(cloud_top)) & (cloud_type == 14)
    noclouds_mask = cloud_mask == 0.0

    # Fractional clouds (CT=10): gaussian-fill missing heights, override with 2000 m
    cloud_top[fractional_mask] = _gaussian_filter_nans(cloud_top, sigma=10, mode="nearest")[
        fractional_mask
    ]
    cloud_top[fractional_mask] = 2_000.0

    # Thin cirrus (CT=11) and cirrus above (CT=14): fixed or gaussian-filled heights
    cloud_top[thinhighcirrus_mask] = 10_000.0
    cloud_top[cirrus_above_mask] = 10_000.0

    cloud_eff[noclouds_mask] = 0.0
    cloud_eff[thinhighcirrus_mask] = 0.4
    cloud_eff[fractional_mask] = _gaussian_filter_nans(cloud_eff, sigma=10, mode="nearest")[
        fractional_mask
    ]
    # Cap fractional cloud coverage at 0.5
    cloud_eff[cloud_type == 10] = np.minimum(cloud_eff[cloud_type == 10], 0.5)
    cloud_eff[cirrus_above_mask] = _gaussian_filter_nans(cloud_eff, sigma=10, mode="nearest")[
        cirrus_above_mask
    ]

    # Remove cloudy pixels whose cloud-top is below the terrain surface
    below_topo = None
    if topo_saf is not None:
        below_topo = cloud_top + height_tolerance_m < topo_saf
        n_below = int(below_topo.sum())
        if remove_below_topo:
            cloud_mask[below_topo] = 0.0
            cloud_eff[below_topo] = 0.0
            LOG.info(
                "Below-topo filter: removed %d cloudy pixels (%.2f%% of domain)",
                n_below,
                100.0 * n_below / cloud_top.size,
            )
        else:
            LOG.info(
                "Below-topo filter: zeroed cloud_top for %d pixels (%.2f%% of domain)",
                n_below,
                100.0 * n_below / cloud_top.size,
            )
        cloud_top[below_topo] = 0.0

    # Remove misclassified fractional clouds over lakes (morphological opening)
    if lakes_saf is not None:
        fractional_lakes = (cloud_type == 10.0) & lakes_saf
        opened = binary_opening(
            cloud_mask > 0, structure=np.ones((5, 5), dtype=bool)
        ).astype(float)
        cloud_mask[fractional_lakes] = opened[fractional_lakes]


# ── Cloud analysis (mirrors cloud_extrapolation height='top') ─────────────────

def _compute_cloud_analysis(
    cloud_mask: np.ndarray,
    cloud_eff: np.ndarray,
    cloud_top: np.ndarray,
    cloud_type: np.ndarray,
    lim_low: float,
    lim_medium: float,
) -> dict:
    """Derive CT/CL/CM/CH fractions from preprocessed SAF fields."""
    okcloud = cloud_mask > 0.1
    oklow = okcloud & (cloud_top <= lim_low)
    okmid = okcloud & (cloud_top > lim_low) & (cloud_top <= lim_medium)
    okhigh = okcloud & (cloud_top > lim_medium)
    ct_nan = np.isnan(cloud_type)

    def _layer(ok_mask):
        arr = np.zeros_like(cloud_eff)
        arr[ok_mask] = cloud_eff[ok_mask]
        arr[ct_nan] = np.nan
        return arr

    return {
        "CT": _layer(okcloud),
        "CL": _layer(oklow),
        "CM": _layer(okmid),
        "CH": _layer(okhigh),
    }


# ── ICON combination (mirrors cloud_extrapolation.py lines 219-244) ──────────

def _apply_icon_combination(
    saf_fields: dict,
    icon_saf: dict,
    cloud_eff: np.ndarray,
    cloud_top: np.ndarray,
    cloud_mask: np.ndarray,
    cloud_type: np.ndarray,
    lim_low: float,
    lim_medium: float,
    partial_high_thr: float,
):
    """Replace saturated SAF pixels and fill lower layers from ICON."""
    ct_nan = np.isnan(cloud_type)
    okcloud = cloud_mask > 0.1
    oklow = okcloud & (cloud_top <= lim_low)
    okmid = okcloud & (cloud_top > lim_low) & (cloud_top <= lim_medium)
    okhigh = okcloud & (cloud_top > lim_medium)

    clct = icon_saf.get("CLCT")
    clcl = icon_saf.get("CLCL")
    clcm = icon_saf.get("CLCM")
    clch = icon_saf.get("CLCH")

    # 1. Replace saturated SAF pixels (cloud_eff == 1) where ICON also shows heavy cloud
    if clcl is not None:
        saf_fields["CL"][oklow & (cloud_eff == 1) & (clcl > 0.8)] = clcl[
            oklow & (cloud_eff == 1) & (clcl > 0.8)
        ]
        saf_fields["CL"][ct_nan] = np.nan
    if clcm is not None:
        saf_fields["CM"][okmid & (cloud_eff == 1) & (clcm > 0.8)] = clcm[
            okmid & (cloud_eff == 1) & (clcm > 0.8)
        ]
        saf_fields["CM"][ct_nan] = np.nan
    if clch is not None:
        saf_fields["CH"][okhigh & (cloud_eff == 1) & (clch > 0.8)] = clch[
            okhigh & (cloud_eff == 1) & (clch > 0.8)
        ]
        saf_fields["CH"][ct_nan] = np.nan
    if clct is not None:
        saf_fields["CT"][okcloud & (cloud_eff == 1) & (clct > 0.8)] = clct[
            okcloud & (cloud_eff == 1) & (clct > 0.8)
        ]
        saf_fields["CT"][ct_nan] = np.nan

    # 2. Fill lower layers where SAF detects high/mid cloud above partial_high_thr
    if clcl is not None:
        mask = (okhigh | okmid) & (cloud_eff > partial_high_thr)
        saf_fields["CL"][mask] = np.fmax(saf_fields["CL"][mask], clcl[mask])
    if clcm is not None:
        mask = okhigh & (cloud_eff > partial_high_thr)
        saf_fields["CM"][mask] = np.fmax(saf_fields["CM"][mask], clcm[mask])

    # 3. Keep CT = max(CT, CL, CM, CH) consistent
    saf_fields["CT"][:] = np.fmax(
        saf_fields["CT"],
        np.fmax(saf_fields["CL"], np.fmax(saf_fields["CM"], saf_fields["CH"])),
    )
    saf_fields["CT"][ct_nan] = np.nan


# ── External ICON GRIB reading ─────────────────────────────────────────────────

def _find_icon_run(
    timestamp: datetime,
    icon_base_dir: Path,
    icon_offset_h: int,
    icon_member: str,
):
    """Return (run_time, lead_h, grib_path) for the nearest available ICON-CH1-EPS run."""
    for day_delta in [0, -1]:
        base_date = timestamp.date() + timedelta(days=day_delta)
        for rh in reversed([0, 3, 6, 9, 12, 15, 18, 21]):
            run_time = datetime(base_date.year, base_date.month, base_date.day, rh)
            lag = (timestamp - run_time).total_seconds() / 3600.0
            if lag < icon_offset_h:
                continue
            lead_h = round(lag)
            matches = sorted(icon_base_dir.glob(f"{run_time:%y%m%d%H}*"))
            if not matches:
                continue
            grib_fn = matches[-1] / "grib" / f"i1eff{lead_h:04d}0000_{icon_member}"
            if grib_fn.exists():
                return run_time, lead_h, grib_fn
    raise FileNotFoundError(
        f"No ICON-CH1-EPS GRIB found for {timestamp:%Y-%m-%dT%H:%M} in {icon_base_dir}"
    )


def _read_icon_grib(grib_path: Path, grid_file: Path):
    """Read CLCT/CLCL/CLCM/CLCH from an ICON-CH1-EPS GRIB2 file.

    Returns (fields_dict, lat_icon_1d, lon_icon_1d).
    Fields are normalised to [0, 1] (GRIB values are in [0, 100]).
    """
    with netCDF4.Dataset(grid_file) as ds:
        lat_icon = np.degrees(np.array(ds["clat"]))
        lon_icon = np.degrees(np.array(ds["clon"]))

    fields = {}
    with open(grib_path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                disc = eccodes.codes_get(h, "discipline")
                cat = eccodes.codes_get(h, "parameterCategory")
                num = eccodes.codes_get(h, "parameterNumber")
                if disc == 0 and cat == 6:
                    lv = eccodes.codes_get(h, "typeOfLevel")
                    top = eccodes.codes_get(h, "topLevel")
                    bot = eccodes.codes_get(h, "bottomLevel")
                    key = None
                    if num == 1 and lv == "surface":
                        key = "CLCT"
                    elif num == 22:
                        if top == 0 and bot == 400:
                            key = "CLCH"
                        elif top == 400 and bot == 800:
                            key = "CLCM"
                        elif top == 800:
                            key = "CLCL"
                    if key and key not in fields:
                        fields[key] = eccodes.codes_get_values(h) / 100.0
            except Exception:
                pass
            eccodes.codes_release(h)

    return fields, lat_icon, lon_icon


# ── Auxiliary static data ─────────────────────────────────────────────────────

def _load_topo(dem_file: Path):
    if not dem_file.exists():
        warnings.warn(f"DEM file not found: {dem_file}. Topography filter skipped.")
        return None
    with netCDF4.Dataset(dem_file) as ds:
        return np.array(ds["DEM"]).astype(float)


def _load_lakes(lakes_file: Path):
    if not lakes_file.exists():
        warnings.warn(f"Lakes file not found: {lakes_file}. Lakes filter skipped.")
        return None
    return np.load(lakes_file).astype(bool)


# ── NetCDF output ──────────────────────────────────────────────────────────────

def _write_netcdf(
    fields: dict,
    timestamp: datetime,
    sat: str,
    domain: str,
    out_path: Path,
    lim_low: float,
    lim_medium: float,
):
    x = np.linspace(X0_LV03, X1_LV03 - DX, NX_INCA) + DX / 2.0
    y = np.linspace(Y0_LV03, Y1_LV03 - DX, NY_INCA) + DX / 2.0
    with netCDF4.Dataset(out_path, "w") as ds:
        ds.title = f"SAFNWC CT analysis on INCA/CCS4 — {timestamp:%Y-%m-%dT%H:%M:%SZ}"
        ds.satellite = sat
        ds.domain = domain
        ds.timestamp = f"{timestamp:%Y-%m-%dT%H:%M:%SZ}"
        ds.projection = _LV03_PROJ4
        ds.createDimension("y", NY_INCA)
        ds.createDimension("x", NX_INCA)
        vx = ds.createVariable("x", "f4", ("x",))
        vx.units = "m"
        vx.long_name = "LV03 easting (pixel centre)"
        vx[:] = x
        vy = ds.createVariable("y", "f4", ("y",))
        vy.units = "m"
        vy.long_name = "LV03 northing (pixel centre, row 0 = south)"
        vy[:] = y
        long_names = {
            "CT": "Total cloud cover fraction",
            "CL": f"Low cloud fraction (ctth_alti <= {lim_low:.0f} m)",
            "CM": f"Medium cloud fraction ({lim_low:.0f} < ctth_alti <= {lim_medium:.0f} m)",
            "CH": f"High cloud fraction (ctth_alti > {lim_medium:.0f} m)",
        }
        for name, arr in fields.items():
            v = ds.createVariable(name, "f4", ("y", "x"), fill_value=np.nan)
            v.long_name = long_names[name]
            v.units = "1"
            v.valid_range = np.array([0.0, 1.0], dtype="f4")
            v[:] = arr
    LOG.info("Wrote %s", out_path)


# ── Plugin class ───────────────────────────────────────────────────────────────

class CloudAnalysis(Filter):
    """Cloud analysis combining SAFNWC CT satellite data with ICON cloud fields.

    Reads SAFNWC CMA/CT/CTTH satellite files, reads ICON cloud cover fields
    (CLCT, CLCH, CLCM, CLCL) either from the input FieldList or from an
    external GRIB file, runs the INCA cloud analysis pipeline, and writes the
    resulting CT/CL/CM/CH fields to a NetCDF file.

    The forecast FieldList is returned unchanged.

    Parameters
    ----------
    saf_dir : str
        Directory containing SAFNWC NetCDF files.
    output_dir : str
        Directory where CT_analysis_{timestamp}.nc files are written.
    icon_grid_file : str
        ICON-CH1 unstructured grid NetCDF (cell centres clat/clon).
        Required for ICON → INCA reprojection in both icon_source modes.
    dem_file : str, optional
        INCA-domain DEM NetCDF (variable "DEM", shape 640×710).
        When omitted the topography filter is skipped.
    lakes_file : str, optional
        INCA-domain lake boolean mask (.npy, shape 640×710).
        When omitted the lake morphological filter is skipped.
    satellites : list of str, optional
        SAF satellite IDs to try in order (default: MTI1, MTI2, MSG4, MSG3).
    domains : list of str, optional
        SAF domain names to try in order (default: alps-NR, alps-VISIR).
    use_plax : bool
        Prefer parallax-corrected (_PLAX.nc) SAF files. Default True.
    lim_low_m : float
        Upper limit [m] for low clouds. Default 2400.
    lim_medium_m : float
        Upper limit [m] for medium clouds. Default 7200.
    height_tolerance_m : float
        Pixels with cloud_top < topo - height_tolerance_m are removed. Default 200.
    remove_below_topo_clouds : bool
        When True: clear cloud_mask and cloud_eff for below-topo pixels.
        When False: only zero cloud_top. Default True.
    partial_high_thr : float
        Minimum SAF cloud_eff for ICON to fill lower layers beneath high cloud.
        Default 0.2.
    icon_source : str
        "fieldlist" — extract CLCT/CLCL/CLCM/CLCH from the input FieldList.
        "grib"      — find the nearest ICON-CH1-EPS GRIB file on disk.
        Default "fieldlist".
    icon_base_dir : str, optional
        Root directory for ICON-CH1-EPS GRIB runs. Required when
        icon_source == "grib".
    icon_member : str
        EPS member suffix used in GRIB filenames (default "000").
    icon_offset_h : int
        Minimum lag [h] after run time before GRIB output is considered
        available. Default 1.
    icon_shortnames : dict, optional
        Override ECMWF shortName for FieldList extraction.
        Default: {"CLCT": "tcc", "CLCL": "lcc", "CLCM": "mcc", "CLCH": "hcc"}.
    """

    def __init__(
        self,
        saf_dir: str,
        output_dir: str,
        icon_grid_file: str,
        dem_file: str = None,
        lakes_file: str = None,
        satellites: list = None,
        domains: list = None,
        use_plax: bool = True,
        lim_low_m: float = 2400.0,
        lim_medium_m: float = 7200.0,
        height_tolerance_m: float = 200.0,
        remove_below_topo_clouds: bool = True,
        partial_high_thr: float = 0.2,
        icon_source: str = "fieldlist",
        icon_base_dir: str = None,
        icon_member: str = "000",
        icon_offset_h: int = 1,
        icon_shortnames: dict = None,
    ):
        super().__init__()
        self._saf_dir = Path(saf_dir)
        self._output_dir = Path(output_dir)
        self._icon_grid_file = Path(icon_grid_file)
        self._dem_file = Path(dem_file) if dem_file else None
        self._lakes_file = Path(lakes_file) if lakes_file else None
        self._satellites = satellites or ["MTI1", "MTI2", "MTI3", "MSG4", "MSG3"]
        self._domains = domains or ["alps-NR", "alps-VISIR"]
        self._use_plax = use_plax
        self._lim_low = float(lim_low_m)
        self._lim_medium = float(lim_medium_m)
        self._height_tolerance_m = float(height_tolerance_m)
        self._remove_below_topo = bool(remove_below_topo_clouds)
        self._partial_high_thr = float(partial_high_thr)
        self._icon_source = icon_source
        self._icon_base_dir = Path(icon_base_dir) if icon_base_dir else None
        self._icon_member = str(icon_member)
        self._icon_offset_h = int(icon_offset_h)
        self._icon_shortnames = {**_DEFAULT_ICON_SHORTNAMES, **(icon_shortnames or {})}

        if icon_source == "grib" and self._icon_base_dir is None:
            raise ValueError("icon_base_dir is required when icon_source='grib'")

        # Lazily loaded static data (cached after first use)
        self._topo_inca = None
        self._lakes_inca = None
        self._topo_loaded = False
        self._lakes_loaded = False

        # Read ICON cell-centre lat/lon once from the grid file
        with netCDF4.Dataset(self._icon_grid_file) as ds:
            self._lat_icon = np.degrees(np.array(ds["clat"]))
            self._lon_icon = np.degrees(np.array(ds["clon"]))
        LOG.info("ICON grid loaded: %d cells from %s", len(self._lat_icon), self._icon_grid_file)

        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy auxiliary data ────────────────────────────────────────────────────

    def _get_topo(self):
        if not self._topo_loaded:
            self._topo_inca = _load_topo(self._dem_file) if self._dem_file else None
            self._topo_loaded = True
        return self._topo_inca

    def _get_lakes(self):
        if not self._lakes_loaded:
            self._lakes_inca = _load_lakes(self._lakes_file) if self._lakes_file else None
            self._lakes_loaded = True
        return self._lakes_inca

    # ── ICON field extraction from FieldList ───────────────────────────────────

    def _extract_icon_from_fieldlist(self, data: ekd.FieldList) -> dict:
        """Extract CLCT/CLCL/CLCM/CLCH from the earthkit FieldList by shortName."""
        icon_geo = {}
        for icon_name, short_name in self._icon_shortnames.items():
            candidates = [
                f for f in data if f.metadata("shortName", default=None) == short_name
            ]
            if not candidates:
                LOG.warning(
                    "Field '%s' (shortName='%s') not found in FieldList", icon_name, short_name
                )
                continue
            vals = candidates[0].to_numpy()
            # Normalise to [0, 1] if the values are in percent
            icon_geo[icon_name] = vals / 100.0 if np.nanmax(vals) > 1.5 else vals
        if not icon_geo:
            raise ValueError(
                "No ICON cloud fields found in the FieldList. "
                f"Expected shortNames: {list(self._icon_shortnames.values())}"
            )
        return icon_geo

    # ── Filter entry point ─────────────────────────────────────────────────────

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Run cloud analysis for the current timestep; return the FieldList unchanged."""
        ref_time = data[0].datetime()["valid_time"]
        LOG.info("CloudAnalysis: processing %s", ref_time.strftime("%Y-%m-%dT%H:%M UTC"))

        # ── Locate and read SAF files ──────────────────────────────────────────
        ct_fn, sat, domain = _find_saf_file(
            self._saf_dir, ref_time, "CT", self._satellites, self._domains, self._use_plax
        )
        cma_fn, *_ = _find_saf_file(
            self._saf_dir, ref_time, "CMA", self._satellites, self._domains, self._use_plax
        )
        ctth_fn, *_ = _find_saf_file(
            self._saf_dir, ref_time, "CTTH", self._satellites, self._domains, self._use_plax
        )
        LOG.info("SAF CT   : %s", ct_fn.name)
        LOG.info("SAF CMA  : %s", cma_fn.name)
        LOG.info("SAF CTTH : %s", ctth_fn.name)

        with netCDF4.Dataset(ct_fn) as ds:
            gdal_proj = str(ds.gdal_projection)
            xgeo_ul = float(ds.gdal_xgeo_up_left)
            ygeo_ul = float(ds.gdal_ygeo_up_left)
            xgeo_lr = float(ds.gdal_xgeo_low_right)
            ygeo_lr = float(ds.gdal_ygeo_low_right)
            cloud_type = _read_var(ds, "ct")

        with netCDF4.Dataset(cma_fn) as ds:
            cloud_mask = _read_var(ds, "cma")

        with netCDF4.Dataset(ctth_fn) as ds:
            cloud_top = _read_var(ds, "ctth_alti")
            cloud_eff = _read_var(ds, "ctth_effectiv")

        ny_saf, nx_saf = cloud_type.shape
        LOG.info(
            "SAF domain: %s / %s  shape %dx%d  cloudy=%.1f%%",
            sat,
            domain,
            nx_saf,
            ny_saf,
            float(np.nanmean(cloud_mask == 1.0)) * 100,
        )

        # ── Reproject auxiliary INCA data to SAF domain ────────────────────────
        kw_repr = dict(
            gdal_proj=gdal_proj,
            xgeo_ul=xgeo_ul,
            ygeo_ul=ygeo_ul,
            xgeo_lr=xgeo_lr,
            ygeo_lr=ygeo_lr,
            ny_saf=ny_saf,
            nx_saf=nx_saf,
        )
        topo_inca = self._get_topo()
        lakes_inca = self._get_lakes()
        topo_saf = (
            _reproject_inca_to_saf(topo_inca, fill=np.nan, **kw_repr)
            if topo_inca is not None
            else None
        )
        lakes_saf = (
            _reproject_inca_to_saf(lakes_inca.astype(float), fill=0.0, **kw_repr) > 0.5
            if lakes_inca is not None
            else None
        )

        # ── Preprocess SAF fields in-place ─────────────────────────────────────
        LOG.info("Applying preprocess_saf …")
        _preprocess_saf(
            cloud_mask,
            cloud_eff,
            cloud_top,
            cloud_type,
            topo_saf,
            lakes_saf,
            self._height_tolerance_m,
            self._remove_below_topo,
        )

        # ── Load ICON cloud fields ─────────────────────────────────────────────
        if self._icon_source == "fieldlist":
            LOG.info("Reading ICON cloud fields from FieldList …")
            icon_geo = self._extract_icon_from_fieldlist(data)
            lat_icon, lon_icon = self._lat_icon, self._lon_icon
        else:
            run_time, lead_h, grib_path = _find_icon_run(
                ref_time, self._icon_base_dir, self._icon_offset_h, self._icon_member
            )
            LOG.info(
                "ICON run %s +%dh → %s",
                run_time.strftime("%Y-%m-%dT%H:%M UTC"),
                lead_h,
                grib_path.name,
            )
            icon_geo, lat_icon, lon_icon = _read_icon_grib(grib_path, self._icon_grid_file)

        LOG.info("ICON fields available: %s", list(icon_geo))

        # ── Reproject ICON → INCA → SAF domain ────────────────────────────────
        LOG.info("Reprojecting ICON → INCA → SAF …")
        icon_inca = {
            k: _reproject_icon_to_inca(v, lat_icon, lon_icon) for k, v in icon_geo.items()
        }
        icon_saf = {
            k: _reproject_inca_to_saf(icon_inca[k], fill=np.nan, **kw_repr)
            for k in icon_inca
        }

        # ── Classify SAF into cloud layers ─────────────────────────────────────
        LOG.info("Classifying SAF cloud layers …")
        saf_fields = _compute_cloud_analysis(
            cloud_mask, cloud_eff, cloud_top, cloud_type, self._lim_low, self._lim_medium
        )

        # ── Apply ICON combination ─────────────────────────────────────────────
        LOG.info("Applying ICON combination …")
        _apply_icon_combination(
            saf_fields,
            icon_saf,
            cloud_eff,
            cloud_top,
            cloud_mask,
            cloud_type,
            self._lim_low,
            self._lim_medium,
            self._partial_high_thr,
        )

        # ── Reproject SAF+ICON → INCA grid ─────────────────────────────────────
        LOG.info("Reprojecting SAF+ICON → INCA grid …")
        kw_saf_to_inca = dict(
            gdal_proj=gdal_proj,
            xgeo_ul=xgeo_ul,
            ygeo_ul=ygeo_ul,
            xgeo_lr=xgeo_lr,
            ygeo_lr=ygeo_lr,
        )
        inca_fields = {
            name: _reproject_saf_to_inca(arr, fill=np.nan, **kw_saf_to_inca)
            for name, arr in saf_fields.items()
        }

        for name, arr in inca_fields.items():
            LOG.info(
                "INCA %-4s  mean=%.3f  valid=%.1f%%",
                name,
                np.nanmean(arr),
                np.isfinite(arr).mean() * 100,
            )

        # ── Write NetCDF ───────────────────────────────────────────────────────
        tag = ref_time.strftime("%Y%m%d%H%M")
        out_path = self._output_dir / f"CT_analysis_{tag}.nc"
        _write_netcdf(inca_fields, ref_time, sat, domain, out_path, self._lim_low, self._lim_medium)

        return data
