"""
CloudNudging — SAFNWC CT + ICON/VARDA cloud cover analysis plugin.

Reads SAFNWC CMA/CT/CTTH satellite data, extracts ICON/VARDA cloud cover fields
(CLCT, CLCH, CLCM, CLCL) from the input FieldList or from an external GRIB
file, runs the cloud analysis pipeline, and writes CT/CL/CM/CH on the
ICON/VARDA unstructured grid to a NetCDF output file.

Pipeline (mirrors cloud_extrapolation.py / INCA cloud module):
  1. preprocess_saf        — handle fractional/cirrus pixels (SAF domain)
  2. compute_cloud_analysis — classify into total/low/mid/high layers (SAF domain)
  3. reproject SAF fields  → ICON/VARDA unstructured grid
  4. apply_topo_lakes_filter — zero cloud below topography / over lakes (ICON/VARDA grid)
  5. apply_icon_combination — fill from ICON/VARDA where SAF is saturated or absent
  6. write NetCDF on the ICON/VARDA unstructured grid

The SAF alps domains do NOT cover the full ICON/VARDA domain:
  SAF alps-NR covers roughly 43–50°N, 2–16°E
  ICON/VARDA covers roughly 42–50.5°N, -1–18°E
ICON/VARDA cells outside the SAF footprint (~1–2° margin on all sides) receive
NaN from the SAF reprojection and are filled directly from ICON/VARDA model
fields in step 5.

Topography (topography_c) and lake fraction (FR_LAKE) are read from the
ICON/VARDA extpar file (same file used by the nudging plugin).

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
from scipy.ndimage import gaussian_filter

import earthkit.data as ekd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

# ECMWF shortName for each ICON/VARDA cloud variable (overridable in __init__)
_DEFAULT_ICON_SHORTNAMES = {
    "CLCT": "tcc",   # total cloud cover
    "CLCL": "lcc",   # low cloud cover
    "CLCM": "mcc",   # medium cloud cover
    "CLCH": "hcc",   # high cloud cover
}


# ── SAF file discovery ─────────────────────────────────────────────────────────

def _find_saf_file(saf_dir, timestamp, product, satellites, domains, use_plax):
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


# ── SAF → ICON/VARDA reprojection ───────────────────────────────────────────────────

def _reproject_saf_to_icon(
    saf_array: np.ndarray,
    gdal_proj: str,
    xgeo_ul: float,
    ygeo_ul: float,
    xgeo_lr: float,
    ygeo_lr: float,
    lat_icon: np.ndarray,
    lon_icon: np.ndarray,
    fill=np.nan,
) -> np.ndarray:
    """Nearest-neighbour reproject a 2-D SAF field → ICON/VARDA unstructured grid.

    For each ICON/VARDA cell centre (lat/lon in WGS84) the geostationary pixel index
    is computed directly.  Returns a 1-D array of length ncells.
    """
    geos_crs = pyproj.CRS.from_proj4(gdal_proj)
    wgs84_crs = pyproj.CRS.from_epsg(4326)
    wgs84_to_geos = pyproj.Transformer.from_crs(wgs84_crs, geos_crs, always_xy=True)

    ny_saf, nx_saf = saf_array.shape
    dx_saf = (xgeo_lr - xgeo_ul) / nx_saf
    dy_saf = (ygeo_lr - ygeo_ul) / ny_saf

    x_geos, y_geos = wgs84_to_geos.transform(lon_icon.ravel(), lat_icon.ravel())
    col_idx = np.round((x_geos - xgeo_ul) / dx_saf).astype(int)
    row_idx = np.round((y_geos - ygeo_ul) / dy_saf).astype(int)

    valid = (
        np.isfinite(x_geos) & np.isfinite(y_geos)
        & (col_idx >= 0) & (col_idx < nx_saf)
        & (row_idx >= 0) & (row_idx < ny_saf)
    )
    out = np.full(len(lat_icon.ravel()), fill)
    out[valid] = saf_array[row_idx[valid], col_idx[valid]]
    return out


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


# ── SAF preprocessing (SAF domain, no topo/lakes) ────────────────────────────

def _preprocess_saf(cloud_mask, cloud_eff, cloud_top, cloud_type):
    """In-place preprocessing of raw SAF fields (all arrays on the SAF domain).

    Mirrors cloud_funcs.preprocess_saf():
    - CT 1-4  (non-cloud surface):  cloud_mask=0, cloud_eff=0, cloud_top=NaN
    - CT 10   (fractional):         gaussian-fill cloud_top → override 2000 m;
                                    gaussian-fill cloud_eff; cap cloud_eff at 0.5
    - CT 11   (thin high cirrus):   cloud_top=10000 m, cloud_eff=0.4; floor at 0.05
    - CT 14   (cirrus above):       cloud_top=10000 m, cloud_eff gaussian-filled; floor at 0.05
    - clear pixels:                 cloud_eff=0

    Topography and lake filters are applied later on the ICON/VARDA grid
    by _apply_topo_lakes_filter().
    """
    non_cloud_surface = np.isin(cloud_type, [1, 2, 3, 4])
    n_suppressed = int((non_cloud_surface & (cloud_mask > 0)).sum())
    cloud_mask[non_cloud_surface] = 0.0
    cloud_eff[non_cloud_surface] = 0.0
    cloud_top[non_cloud_surface] = np.nan
    LOG.info(
        "CT 1-4 suppression: cleared %d cloudy pixels (%.2f%% of domain)",
        n_suppressed, 100.0 * n_suppressed / cloud_type.size,
    )

    fractional_mask = (np.isnan(cloud_eff) | np.isnan(cloud_top)) & (cloud_type == 10)
    thinhighcirrus_mask = (np.isnan(cloud_eff) | np.isnan(cloud_top)) & (cloud_type == 11)
    cirrus_above_mask = (np.isnan(cloud_eff) | np.isnan(cloud_top)) & (cloud_type == 14)
    noclouds_mask = cloud_mask == 0.0

    cloud_top[fractional_mask] = _gaussian_filter_nans(
        cloud_top, sigma=10, mode="nearest"
    )[fractional_mask]
    cloud_top[fractional_mask] = 2_000.0
    cloud_top[thinhighcirrus_mask] = 10_000.0
    cloud_top[cirrus_above_mask] = 10_000.0

    cloud_eff[noclouds_mask] = 0.0
    cloud_eff[thinhighcirrus_mask] = 0.4
    cloud_eff[fractional_mask] = _gaussian_filter_nans(
        cloud_eff, sigma=10, mode="nearest"
    )[fractional_mask]
    cloud_eff[cloud_type == 10] = np.minimum(cloud_eff[cloud_type == 10], 0.5)
    cloud_eff[cirrus_above_mask] = _gaussian_filter_nans(
        cloud_eff, sigma=10, mode="nearest"
    )[cirrus_above_mask]

    # enforce minimum cloud_eff for thin cirrus (CT=11) and cirrus-above (CT=14):
    # gaussian fill targets only NaN pixels; non-NaN pixels can still be near 0
    cloud_eff[cloud_type == 11] = np.fmax(cloud_eff[cloud_type == 11], 0.05)
    cloud_eff[cloud_type == 14] = np.fmax(cloud_eff[cloud_type == 14], 0.05)


# ── Topo / lakes filter on ICON/VARDA grid ─────────────────────────────────────────

def _apply_topo_lakes_filter(
    saf_icon: dict,
    cloud_mask_icon: np.ndarray,
    cloud_eff_icon: np.ndarray,
    cloud_top_icon: np.ndarray,
    cloud_type_icon: np.ndarray,
    topo_icon: np.ndarray,
    fr_lake_icon: np.ndarray,
    height_tolerance_m: float,
    remove_below_topo: bool,
):
    """Apply topography and lake filters to cloud fields on the ICON/VARDA grid.

    Mirrors the SAF-domain topo/lakes logic from preprocess_saf, applied
    directly on the 1-D ICON/VARDA arrays after reprojection.

    saf_icon and the auxiliary arrays are modified in-place.
    """
    # Topography: clear clouds whose top is below the terrain surface
    valid_top = np.isfinite(cloud_top_icon) & np.isfinite(topo_icon)
    below_topo = valid_top & (cloud_top_icon + height_tolerance_m < topo_icon)
    n_below = int(below_topo.sum())
    if remove_below_topo:
        cloud_mask_icon[below_topo] = 0.0
        cloud_eff_icon[below_topo] = 0.0
        for arr in saf_icon.values():
            arr[below_topo] = 0.0
        LOG.info(
            "Below-topo filter: cleared %d ICON/VARDA cells (%.2f%%)",
            n_below, 100.0 * n_below / len(topo_icon),
        )
    else:
        cloud_top_icon[below_topo] = 0.0
        LOG.info(
            "Below-topo filter: zeroed cloud_top for %d ICON/VARDA cells (%.2f%%)",
            n_below, 100.0 * n_below / len(topo_icon),
        )

    # Lakes: clear fractional clouds (CT type 10) over lake cells
    lake_cells = fr_lake_icon > 0.5
    fractional_lakes = lake_cells & (cloud_type_icon == 10.0)
    n_lakes = int(fractional_lakes.sum())
    cloud_mask_icon[fractional_lakes] = 0.0
    cloud_eff_icon[fractional_lakes] = 0.0
    for arr in saf_icon.values():
        arr[fractional_lakes] = 0.0
    LOG.info(
        "Lakes filter: cleared fractional cloud at %d ICON/VARDA cells (%.2f%%)",
        n_lakes, 100.0 * n_lakes / len(fr_lake_icon),
    )


# ── Cloud analysis (mirrors cloud_extrapolation height='top') ─────────────────

def _compute_cloud_analysis(
    cloud_mask, cloud_eff, cloud_top, cloud_type, lim_low: float, lim_medium: float,
) -> dict:
    """Derive CT/CL/CM/CH fractions from preprocessed SAF fields (SAF domain)."""
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


# ── ICON/VARDA combination on ICON/VARDA grid ─────────────────────────────────────────────

def _apply_icon_combination(
    saf_icon: dict,
    icon_geo: dict,
    cloud_eff_icon: np.ndarray,
    cloud_top_icon: np.ndarray,
    cloud_mask_icon: np.ndarray,
    cloud_type_icon: np.ndarray,
    lim_low: float,
    lim_medium: float,
    partial_high_thr: float,
):
    """Replace saturated SAF pixels and fill lower layers from ICON/VARDA.

    All arrays are 1-D (ncells) on the ICON/VARDA unstructured grid.
    saf_icon is modified in-place.

    Cells outside the SAF domain have cloud_type_icon=NaN (set by the
    reprojection step).  These are filled directly from ICON/VARDA so the output
    has no gaps at the domain boundary (~1–2° margin on all sides).
    """
    # Cells outside the SAF footprint (NaN from reprojection)
    no_saf = np.isnan(cloud_type_icon)
    n_no_saf = int(no_saf.sum())
    if n_no_saf:
        LOG.info(
            "Outside-SAF cells: %d (%.1f%%) — filled from ICON/VARDA",
            n_no_saf, 100.0 * n_no_saf / len(cloud_type_icon),
        )

    # Within the SAF domain: cells where cloud_type is genuinely undefined
    ct_nan_in_saf = np.isnan(cloud_type_icon) & ~no_saf
    okcloud = cloud_mask_icon > 0.1
    oklow   = okcloud & (cloud_top_icon <= lim_low)
    okmid   = okcloud & (cloud_top_icon > lim_low) & (cloud_top_icon <= lim_medium)
    okhigh  = okcloud & (cloud_top_icon > lim_medium)

    clct = icon_geo.get("CLCT")
    clcl = icon_geo.get("CLCL")
    clcm = icon_geo.get("CLCM")
    clch = icon_geo.get("CLCH")

    # 1. Replace saturated SAF pixels (cloud_eff == 1) where ICON/VARDA also shows heavy cloud
    if clcl is not None:
        mask = oklow & (cloud_eff_icon == 1) & (clcl > 0.8)
        saf_icon["CL"][mask] = clcl[mask]
        saf_icon["CL"][ct_nan_in_saf] = np.nan
    if clcm is not None:
        mask = okmid & (cloud_eff_icon == 1) & (clcm > 0.8)
        saf_icon["CM"][mask] = clcm[mask]
        saf_icon["CM"][ct_nan_in_saf] = np.nan
    if clch is not None:
        mask = okhigh & (cloud_eff_icon == 1) & (clch > 0.8)
        saf_icon["CH"][mask] = clch[mask]
        saf_icon["CH"][ct_nan_in_saf] = np.nan
    if clct is not None:
        mask = okcloud & (cloud_eff_icon == 1) & (clct > 0.8)
        saf_icon["CT"][mask] = clct[mask]
        saf_icon["CT"][ct_nan_in_saf] = np.nan

    # 2. Fill lower layers where SAF detects high/mid cloud above partial_high_thr
    if clcl is not None:
        mask = (okhigh | okmid) & (cloud_eff_icon > partial_high_thr)
        saf_icon["CL"][mask] = np.fmax(saf_icon["CL"][mask], clcl[mask])
    if clcm is not None:
        mask = okhigh & (cloud_eff_icon > partial_high_thr)
        saf_icon["CM"][mask] = np.fmax(saf_icon["CM"][mask], clcm[mask])

    # 3. Keep CT = max(CT, CL, CM, CH) consistent within SAF domain
    saf_icon["CT"][:] = np.fmax(
        saf_icon["CT"],
        np.fmax(saf_icon["CL"], np.fmax(saf_icon["CM"], saf_icon["CH"])),
    )
    saf_icon["CT"][ct_nan_in_saf] = np.nan

    # 4. Outside-SAF domain: fill directly from ICON/VARDA (no SAF data available)
    if n_no_saf:
        for icon_name, saf_key in [
            ("CLCT", "CT"), ("CLCL", "CL"), ("CLCM", "CM"), ("CLCH", "CH")
        ]:
            src = icon_geo.get(icon_name)
            if src is not None:
                saf_icon[saf_key][no_saf] = src[no_saf]


# ── External ICON/VARDA GRIB reading ─────────────────────────────────────────────────

def _find_icon_run(timestamp, icon_base_dir, icon_offset_h, icon_member):
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


def _read_icon_grib(grib_path: Path):
    """Read CLCT/CLCL/CLCM/CLCH from an ICON-CH1-EPS GRIB2 file.

    Returns a dict of 1-D field arrays normalised to [0, 1].
    """
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

    return fields


# ── NetCDF output on ICON/VARDA unstructured grid ───────────────────────────────────

def _write_netcdf(
    fields: dict,
    timestamp: datetime,
    sat: str,
    domain: str,
    lat_icon: np.ndarray,
    lon_icon: np.ndarray,
    out_path: Path,
    lim_low: float,
    lim_medium: float,
):
    """Write CT/CL/CM/CH as 1-D arrays on the ICON/VARDA unstructured grid."""
    ncells = len(lat_icon.ravel())
    with netCDF4.Dataset(out_path, "w") as ds:
        ds.title = f"SAFNWC CT analysis on ICON/VARDA grid — {timestamp:%Y-%m-%dT%H:%M:%SZ}"
        ds.satellite = sat
        ds.domain = domain
        ds.timestamp = f"{timestamp:%Y-%m-%dT%H:%M:%SZ}"
        ds.grid = "ICON/VARDA unstructured (cell centres)"

        ds.createDimension("ncells", ncells)

        vlat = ds.createVariable("lat", "f4", ("ncells",))
        vlat.units = "degrees_north"
        vlat.long_name = "ICON/VARDA cell centre latitude"
        vlat[:] = lat_icon.ravel()

        vlon = ds.createVariable("lon", "f4", ("ncells",))
        vlon.units = "degrees_east"
        vlon.long_name = "ICON/VARDA cell centre longitude"
        vlon[:] = lon_icon.ravel()

        long_names = {
            "CT": "Total cloud cover fraction",
            "CL": f"Low cloud fraction (ctth_alti <= {lim_low:.0f} m)",
            "CM": f"Medium cloud fraction ({lim_low:.0f} < ctth_alti <= {lim_medium:.0f} m)",
            "CH": f"High cloud fraction (ctth_alti > {lim_medium:.0f} m)",
        }
        for name, arr in fields.items():
            v = ds.createVariable(name, "f4", ("ncells",), fill_value=np.nan)
            v.long_name = long_names[name]
            v.units = "1"
            v.valid_range = np.array([0.0, 1.0], dtype="f4")
            v[:] = arr.ravel()

    LOG.info("Wrote %s", out_path)


# ── Plugin class ───────────────────────────────────────────────────────────────

class CloudNudging(Filter):
    """Cloud analysis combining SAFNWC CT satellite data with ICON/VARDA cloud fields.

    Reads SAFNWC CMA/CT/CTTH satellite files, reads ICON/VARDA cloud cover fields
    (CLCT, CLCH, CLCM, CLCL) either from the input FieldList or from an
    external GRIB file, runs the cloud analysis pipeline, and writes the
    resulting CT/CL/CM/CH fields on the ICON/VARDA unstructured grid to a NetCDF file.

    Topography (topography_c) and lake fraction (FR_LAKE) are read from the
    ICON/VARDA extpar file (same file used by the nudging plugin's
    icon_extpar_file / icon_orog_file parameter).

    ICON/VARDA cells outside the SAF alps domain boundary (~1–2° on all sides)
    are filled directly from ICON/VARDA model fields.

    The forecast FieldList is returned unchanged.

    Parameters
    ----------
    saf_dir : str
        Directory containing SAFNWC NetCDF files.
    output_dir : str
        Directory where CT_analysis_{timestamp}.nc files are written.
    icon_extpar_file : str, optional
        ICON/VARDA extpar NetCDF (clat/clon in radians, topography_c in metres,
        FR_LAKE as fraction 0-1).  Alias: icon_orog_file (same as nudging plugin).
        Example: external_parameter_icon_grid_0001_R19B08_mch.nc
    icon_orog_file : str, optional
        Alias for icon_extpar_file; accepted for consistency with the nudging plugin.
    satellites : list of str, optional
        SAF satellite IDs to try in order.  Default: MTI1, MTI2, MTI3, MSG4, MSG3.
    domains : list of str, optional
        SAF domain names to try in order.  Default: alps-NR, alps-VISIR.
    use_plax : bool
        Prefer parallax-corrected (_PLAX.nc) SAF files.  Default True.
    lim_low_m : float
        Upper limit [m] for low clouds.  Default 2400.
    lim_medium_m : float
        Upper limit [m] for medium clouds.  Default 7200.
    height_tolerance_m : float
        Pixels with cloud_top < topo - height_tolerance_m are removed.  Default 200.
    remove_below_topo_clouds : bool
        When True: clear cloud_mask and cloud_eff for below-topo pixels.
        When False: only zero cloud_top.  Default True.
    partial_high_thr : float
        Minimum SAF cloud_eff for ICON/VARDA to fill lower layers beneath high cloud.
        Default 0.2.
    icon_source : str
        "fieldlist" — extract CLCT/CLCL/CLCM/CLCH from the input FieldList.
        "grib"      — find the nearest ICON/VARDA GRIB file on disk.
        Default "fieldlist".
    icon_base_dir : str, optional
        Root directory for ICON/VARDA GRIB runs.  Required when icon_source == "grib".
    icon_member : str
        EPS member suffix used in GRIB filenames.  Default "000".
    icon_offset_h : int
        Minimum lag [h] after run time before GRIB output is considered
        available.  Default 1.
    icon_shortnames : dict, optional
        Override ECMWF shortName for FieldList extraction.
        Default: {"CLCT": "tcc", "CLCL": "lcc", "CLCM": "mcc", "CLCH": "hcc"}.
    """

    def __init__(
        self,
        saf_dir: str,
        output_dir: str,
        icon_extpar_file: str = None,
        satellites: list = None,
        icon_orog_file: str = None,
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
        extpar = icon_extpar_file or icon_orog_file
        if extpar is None:
            raise ValueError("icon_extpar_file (or icon_orog_file) is required")
        self._icon_extpar_file = Path(extpar)
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

        # Read ICON/VARDA static fields from extpar (same file as nudging plugin)
        with netCDF4.Dataset(self._icon_extpar_file) as ds:
            self._lat_icon = np.degrees(np.array(ds["clat"], dtype=np.float64)).ravel()
            self._lon_icon = np.degrees(np.array(ds["clon"], dtype=np.float64)).ravel()
            self._topo_icon = np.array(ds["topography_c"], dtype=np.float32).ravel()
            self._fr_lake_icon = np.array(ds["FR_LAKE"], dtype=np.float32).ravel()

        LOG.info(
            "ICON/VARDA extpar loaded: %d cells, topo %.0f-%.0f m, lake fraction %.1f%%  (%s)",
            len(self._lat_icon),
            float(np.nanmin(self._topo_icon)), float(np.nanmax(self._topo_icon)),
            float(np.nanmean(self._fr_lake_icon > 0.5)) * 100,
            self._icon_extpar_file,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── ICON/VARDA field extraction from FieldList ───────────────────────────────────

    def _extract_icon_from_fieldlist(self, data: ekd.FieldList) -> dict:
        """Extract CLCT/CLCL/CLCM/CLCH from the earthkit FieldList by shortName.

        Uses data.sel(shortName=...) like the nudging plugin so that earthkit
        can optimise the lookup (index scan rather than iterating all fields).
        """
        short_to_icon = {v: k for k, v in self._icon_shortnames.items()}
        icon_geo = {}
        for field in data.sel(shortName=list(self._icon_shortnames.values())):
            short_name = field.metadata("shortName")
            icon_name = short_to_icon[short_name]
            vals = field.to_numpy()
            icon_geo[icon_name] = vals / 100.0 if np.nanmax(vals) > 1.5 else vals
        missing = [k for k in self._icon_shortnames if k not in icon_geo]
        if missing:
            LOG.warning(
                "ICON/VARDA cloud fields not found in FieldList: %s (shortNames: %s)",
                missing, [self._icon_shortnames[k] for k in missing],
            )
        if not icon_geo:
            raise ValueError(
                "No ICON/VARDA cloud fields found in the FieldList. "
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
            self._saf_dir, ref_time, "CT",
            self._satellites, self._domains, self._use_plax,
        )
        cma_fn, *_ = _find_saf_file(
            self._saf_dir, ref_time, "CMA",
            self._satellites, self._domains, self._use_plax,
        )
        ctth_fn, *_ = _find_saf_file(
            self._saf_dir, ref_time, "CTTH",
            self._satellites, self._domains, self._use_plax,
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
            sat, domain, nx_saf, ny_saf,
            float(np.nanmean(cloud_mask == 1.0)) * 100,
        )

        # ── Preprocess SAF fields in-place (CT suppression + fill NaNs) ───────
        LOG.info("Applying preprocess_saf ...")
        _preprocess_saf(cloud_mask, cloud_eff, cloud_top, cloud_type)

        # ── Classify SAF into cloud layers (SAF domain) ────────────────────────
        LOG.info("Classifying SAF cloud layers ...")
        saf_fields = _compute_cloud_analysis(
            cloud_mask, cloud_eff, cloud_top, cloud_type,
            self._lim_low, self._lim_medium,
        )

        # ── Load ICON/VARDA cloud fields ─────────────────────────────────────────────
        # Default: read CLCT/CLCL/CLCM/CLCH directly from the input FieldList.
        # Option:  icon_source="grib" reads from an external ICON-CH1-EPS GRIB.
        if self._icon_source == "grib":
            run_time, lead_h, grib_path = _find_icon_run(
                ref_time, self._icon_base_dir, self._icon_offset_h, self._icon_member
            )
            LOG.info(
                "ICON/VARDA GRIB: run %s +%dh -> %s",
                run_time.strftime("%Y-%m-%dT%H:%M UTC"), lead_h, grib_path.name,
            )
            icon_geo = _read_icon_grib(grib_path)
        else:
            LOG.info("Reading ICON/VARDA cloud fields from FieldList ...")
            icon_geo = self._extract_icon_from_fieldlist(data)

        LOG.info("ICON/VARDA fields available: %s", list(icon_geo))

        # ── Reproject SAF fields + auxiliary SAF fields -> ICON/VARDA grid ──────────
        LOG.info("Reprojecting SAF -> ICON/VARDA grid (%d cells) ...", len(self._lat_icon))
        kw = dict(
            gdal_proj=gdal_proj, xgeo_ul=xgeo_ul, ygeo_ul=ygeo_ul,
            xgeo_lr=xgeo_lr, ygeo_lr=ygeo_lr,
            lat_icon=self._lat_icon, lon_icon=self._lon_icon,
        )
        saf_icon = {
            name: _reproject_saf_to_icon(arr, fill=np.nan, **kw)
            for name, arr in saf_fields.items()
        }
        cloud_eff_icon = _reproject_saf_to_icon(cloud_eff, fill=np.nan, **kw)
        cloud_top_icon = _reproject_saf_to_icon(cloud_top, fill=np.nan, **kw)
        cloud_mask_icon = _reproject_saf_to_icon(cloud_mask, fill=np.nan, **kw)
        cloud_type_icon = _reproject_saf_to_icon(cloud_type, fill=np.nan, **kw)

        # ── Topography and lakes filter (ICON/VARDA grid, using extpar data) ─────────
        LOG.info("Applying topo/lakes filter ...")
        _apply_topo_lakes_filter(
            saf_icon, cloud_mask_icon, cloud_eff_icon, cloud_top_icon, cloud_type_icon,
            self._topo_icon, self._fr_lake_icon,
            self._height_tolerance_m, self._remove_below_topo,
        )

        # ── Apply ICON/VARDA combination (ICON/VARDA grid) ─────────────────────────────────
        LOG.info("Applying ICON/VARDA combination ...")
        _apply_icon_combination(
            saf_icon, icon_geo,
            cloud_eff_icon, cloud_top_icon, cloud_mask_icon, cloud_type_icon,
            self._lim_low, self._lim_medium, self._partial_high_thr,
        )

        for name, arr in saf_icon.items():
            LOG.info(
                "ICON/VARDA %-4s  mean=%.3f  valid=%.1f%%",
                name, np.nanmean(arr), np.isfinite(arr).mean() * 100,
            )

        # ── Write NetCDF on ICON/VARDA grid ──────────────────────────────────────────
        tag = ref_time.strftime("%Y%m%d%H%M")
        out_path = self._output_dir / f"CT_analysis_{tag}.nc"
        _write_netcdf(
            saf_icon, ref_time, sat, domain,
            self._lat_icon, self._lon_icon,
            out_path, self._lim_low, self._lim_medium,
        )

        return data
