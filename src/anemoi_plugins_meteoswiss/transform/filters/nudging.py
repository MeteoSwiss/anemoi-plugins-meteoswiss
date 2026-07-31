import logging
import sys
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from scipy.spatial import cKDTree

LOG = logging.getLogger(__name__)

# Maps GRIB shortName -> (station DataFrame column, unit offset applied to obs before nudging).
# Uses COSMO/ICON shortNames as output by the LAM forecaster.
PARAM_MAP = {
    "T_2M": ("2t", 0.0),  # 2 m temperature        [K]
    "TD_2M": ("2d", 0.0),  # 2 m dewpoint           [K]
    "U_10M": ("10u", 0.0),  # 10 m wind U component  [m/s]
    "V_10M": ("10v", 0.0),  # 10 m wind V component  [m/s]
    "PMSL": ("msl", 0.0),  # mean sea-level pressure [Pa]
    "TOT_PREC": ("tp", 0.0),  # hourly precipitation   [kg m-2]
    "VMAX_10M": ("vmax", 0.0),  # 10 m wind gust         [m/s]
}

# Maps station column -> DWH (jretrieve) parameter names required to produce it.
_STATION_COL_TO_JR_PARAMS = {
    "2t": ["tre200s0"],  # 2 m temperature    [°C -> K]
    "2d": ["tde200s0"],  # 2 m dewpoint       [°C -> K]
    "10u": ["fkl010z0", "dkl010z0"],  # wind speed + dir   -> U/V [m/s]
    "10v": ["fkl010z0", "dkl010z0"],
    "msl": ["pp0qffs0"],  # MSLP               [hPa -> Pa]
    "tp": ["rre150h0"],  # hourly precip      [mm = kg m-2]
    "vmax": ["fkl010z1"],  # 10 m wind gust     [m/s]
}


def idw_points(xy_obs, v_obs, xy_tgt, k=5, power=2):
    """Interpolate scattered observations to target locations via Inverse Distance Weighting.

    Parameters
    ----------
    xy_obs : array-like, shape (n_obs, 2)
        2-D coordinates of observation points.
    v_obs : array-like, shape (n_obs,)
        Scalar values at observation points.
    xy_tgt : array-like, shape (n_tgt, 2)
        2-D coordinates of target points.
    k : int
        Number of nearest neighbours to use.
    power : float
        Distance-decay exponent; higher values give more weight to nearby obs.

    Returns
    -------
    numpy.ndarray, shape (n_tgt,)
        IDW-interpolated values at each target point.
    """
    tree = cKDTree(np.asarray(xy_obs, float))
    dist, idx = tree.query(np.asarray(xy_tgt, float), k=k)

    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    w = 1.0 / (dist + 1e-12) ** power
    return (w * np.asarray(v_obs, float)[idx]).sum(axis=1) / w.sum(axis=1)


def interpolation_of_residuals(
    background,
    grid_lat,
    grid_lon,
    st_lat,
    st_lon,
    st_obs,
    k=5,
    power=2,
    max_dist=1.0,
):
    """Correct a background field toward station observations via Interpolation of Residuals (IoR).

    Algorithm
    ---------
    1. Project coordinates (lon scaled by cos(mean_lat)) for approximate metric distances.
    2. Find the nearest grid point per station and extract the background value there.
    3. Compute residuals: background_at_station - observation.
    4. Spread residuals to the full grid with IDW.
    5. Taper the residual field linearly to zero beyond *max_dist* degrees from the
       nearest station, so the correction vanishes in data-sparse regions.
    6. Return background - tapered_residuals.

    Parameters
    ----------
    background : array-like, shape (n_grid,)
        Background field values on the model grid.
    grid_lat, grid_lon : array-like, shape (n_grid,)
        Latitude and longitude of each grid point [degrees].
    st_lat, st_lon : array-like, shape (n_obs,)
        Latitude and longitude of each station [degrees].
    st_obs : array-like, shape (n_obs,)
        Observed values at each station (same units as *background*).
    k : int
        Number of nearest stations used in IDW interpolation.
    power : float
        IDW distance-decay exponent.
    max_dist : float
        Tapering radius [degrees]; correction fades to zero at this distance
        from the nearest station.

    Returns
    -------
    numpy.ndarray, shape (n_grid,)
        Corrected field values.
    """
    background = np.asarray(background, float).ravel()
    lat = np.asarray(grid_lat, float).ravel()
    lon = np.asarray(grid_lon, float).ravel()
    st_obs = np.asarray(st_obs, float)
    st_lat = np.asarray(st_lat, float)
    st_lon = np.asarray(st_lon, float)

    lat0 = np.deg2rad(np.nanmean(lat))
    grid_xy = np.c_[lon * np.cos(lat0), lat]
    st_xy = np.c_[st_lon * np.cos(lat0), st_lat]

    ok = np.isfinite(st_obs) & np.isfinite(st_xy).all(axis=1)
    st_xy = st_xy[ok]
    st_obs = st_obs[ok]

    _, gi = cKDTree(grid_xy).query(st_xy, k=1)
    r_at_st = background[gi] - st_obs
    res = idw_points(st_xy, r_at_st, grid_xy, k=k, power=power)

    dmin, _ = cKDTree(st_xy).query(grid_xy, k=1)
    taper = 1.0 - np.clip(dmin / max_dist, 0.0, 1.0)

    return background - taper * res


class NudgeTowardObservation(Filter):
    """Nudge the forecast initial condition toward surface station observations.

    Applies Interpolation of Residuals (IoR): per-station residuals (background
    minus observation) are spread to the full model grid via IDW and subtracted
    from the background, with a distance-based taper that vanishes the correction
    beyond *max_dist* degrees from the nearest station.

    Observations are loaded from the MeteoSwiss DWH via jretrieve.
    Nudging is applied exactly once — to the initial-condition time step only.
    """

    def __init__(
        self,
        icon_grid_dir: str = "/scratch/mch/llanzila/sruc/aux_files",
        k: int = 3,
        power: float = 4.0,
        max_dist: float = 0.5,
        jretrieve_bbox: list = None,
        jretrieve_src_path: str = "/scratch/mch/llanzila/sruc/evalml/src",
        nudge_variables: list = None,
        use_limitation: int = None,
        run_mode: str = "depl",
    ):
        """Initialise the nudging filter.

        Parameters
        ----------
        icon_grid_dir : str
            Directory containing ``icon_grid_0001_R19B08_mch.nc``.
        k : int
            Number of nearest stations used in IDW interpolation.
        power : float
            IDW distance-decay exponent.
        max_dist : float
            Tapering radius [degrees]; correction fades to zero beyond this
            distance from the nearest station.
        jretrieve_bbox : list, optional
            Bounding box ``[minlat, maxlat, minlon, maxlon]`` for station
            selection. Defaults to ``[40.5, 53.0, 0.0, 17.5]`` (Switzerland
            and surroundings).
        jretrieve_src_path : str
            Directory containing ``jretrieve.py``.
        nudge_variables : list, optional
            GRIB shortNames to nudge (must be keys of PARAM_MAP). Defaults to
            all variables in PARAM_MAP.
        use_limitation : int, optional
            Passed to jretrieve ``--use-limitation``; limits the observation
            time window in minutes (e.g. 50 means observations within ±50 min).
        run_mode : str
            ``'depl'`` (default): ref_time = minimum valid_time across all
            fields. ``'devt'``: ref_time = valid_time of the first field.
        """
        if run_mode not in ("devt", "depl"):
            raise ValueError(f"run_mode must be 'devt' or 'depl', got {run_mode!r}")

        self.icon_grid_dir = Path(icon_grid_dir)
        self.k = k
        self.power = power
        self.max_dist = max_dist
        self.jretrieve_bbox = (
            jretrieve_bbox if jretrieve_bbox is not None else [40.5, 53.0, 0.0, 17.5]
        )
        self.jretrieve_src_path = jretrieve_src_path
        self.use_limitation = use_limitation
        self.run_mode = run_mode
        self._nudging_done = False

        if nudge_variables is not None:
            unknown = set(nudge_variables) - PARAM_MAP.keys()
            if unknown:
                raise ValueError(
                    f"Unknown nudge variables: {unknown}. Valid: {list(PARAM_MAP)}"
                )
            self.param_map = {k: PARAM_MAP[k] for k in nudge_variables}
        else:
            self.param_map = PARAM_MAP

        LOG.info(
            "Nudging filter initialised: variables=%s, k=%d, power=%.1f, max_dist=%.2f",
            list(self.param_map.keys()),
            self.k,
            self.power,
            self.max_dist,
        )
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply observation nudging to the initial-condition fields.

        Nudging is applied only once. Subsequent calls return the input unchanged.

        Parameters
        ----------
        data : ekd.FieldList
            Forecast fields, potentially spanning multiple time steps.

        Returns
        -------
        ekd.FieldList
            Same fields with nudgeable surface parameters corrected toward
            station observations at the initial-condition time step.
        """
        if self._nudging_done:
            return data

        ref_time = (
            data[0].datetime()["valid_time"]
            if self.run_mode == "devt"
            else min(f.datetime()["valid_time"] for f in data)
        )
        LOG.info("Nudging initial condition at %s", ref_time)

        ds = xr.open_dataset(self.icon_grid_dir / "icon_grid_0001_R19B08_mch.nc")
        lat_icon = np.degrees(ds["clat"])
        lon_icon = np.degrees(ds["clon"])

        stations = self._load_stations(ref_time)
        LOG.info("Stations loaded: %d", len(stations))

        nudged = {}
        for field in data.sel(shortName=list(self.param_map.keys())):
            shortname = field.metadata("shortName")
            if field.datetime()["valid_time"] != ref_time:
                continue

            col, offset = self.param_map[shortname]
            if col not in stations.columns or stations[col].isna().all():
                LOG.warning("No observations available for '%s', skipping", shortname)
                continue

            valid = stations[col].notna()
            st_lat = stations.loc[valid, "latitude"].to_numpy()
            st_lon = stations.loc[valid, "longitude"].to_numpy()
            st_obs = stations.loc[valid, col].to_numpy() + offset

            corrected = interpolation_of_residuals(
                field.values,
                lat_icon,
                lon_icon,
                st_lat,
                st_lon,
                st_obs,
                k=self.k,
                power=self.power,
                max_dist=self.max_dist,
            )
            nudged[shortname] = new_field_from_numpy(
                corrected,
                template=field,
                validityDate=field.metadata("validityDate"),
                validityTime=field.metadata("validityTime"),
                dataDate=field.metadata("dataDate"),
                dataTime=field.metadata("dataTime"),
            )
            LOG.info("Nudged '%s' using %d stations", shortname, int(valid.sum()))

        result = [
            nudged.get(f.metadata("shortName"), f)
            if f.datetime()["valid_time"] == ref_time
            else f
            for f in data
        ]
        self._nudging_done = True
        LOG.info("Nudging complete: %d/%d fields updated", len(nudged), len(result))
        return new_fieldlist_from_list(result)

    def _load_stations(self, ref_time) -> pd.DataFrame:
        """Load station observations at *ref_time* from the MeteoSwiss DWH via jretrieve.

        Fetches the DWH parameters needed for the configured nudge variables,
        converts units to SI (°C→K, hPa→Pa, speed+direction→U/V), and returns
        a DataFrame indexed by station abbreviation.

        Parameters
        ----------
        ref_time : datetime
            Target validity time for which observations are requested.

        Returns
        -------
        pandas.DataFrame
            Station observations indexed by ``nat_abbr``, with columns matching
            the station-column names in PARAM_MAP plus ``latitude`` and
            ``longitude``.
        """
        src_path = self.jretrieve_src_path
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        import jretrieve as jr

        needed_cols = {col for col, _ in self.param_map.values()}
        jr_params = list(
            dict.fromkeys(
                p for col in needed_cols for p in _STATION_COL_TO_JR_PARAMS.get(col, [])
            )
        )
        if not jr_params:
            raise ValueError(
                f"No jretrieve parameters found for station columns: {needed_cols}"
            )

        jr.check_prerequisites()

        stations_sel = {"bbox": self.jretrieve_bbox}
        meta = jr.fetch_meta(stations=stations_sel, params=jr_params)
        catalog = jr.StationCatalog.from_meta(meta)
        LOG.info("Station catalog: %d stations", catalog.n)

        df = jr.fetch_data(
            stations=stations_sel,
            params=jr_params,
            start=ref_time,
            end=ref_time,
            increment_minutes=60,
            use_limitation=self.use_limitation,
        )

        df["nat_abbr"] = df["station"].map(
            dict(zip(catalog.station_id, catalog.nat_abbr))
        )
        df["latitude"] = df["station"].map(
            dict(zip(catalog.station_id, catalog.latitude))
        )
        df["longitude"] = df["station"].map(
            dict(zip(catalog.station_id, catalog.longitude))
        )
        df = df.dropna(subset=["nat_abbr"]).set_index("nat_abbr")
        df.index.name = "station"

        if "tre200s0" in df.columns:
            df["2t"] = df["tre200s0"] + 273.15  # °C -> K
        if "tde200s0" in df.columns:
            df["2d"] = df["tde200s0"] + 273.15  # °C -> K
        if "fkl010z0" in df.columns and "dkl010z0" in df.columns:
            dd_rad = np.deg2rad(df["dkl010z0"])
            df["10u"] = -df["fkl010z0"] * np.sin(dd_rad)
            df["10v"] = -df["fkl010z0"] * np.cos(dd_rad)
        if "pp0qffs0" in df.columns:
            df["msl"] = df["pp0qffs0"] * 100.0  # hPa -> Pa
        if "rre150h0" in df.columns:
            df["tp"] = df["rre150h0"]  # mm = kg m-2, no conversion needed
        if "fkl010z1" in df.columns:
            df["vmax"] = df["fkl010z1"]  # m/s, no conversion needed

        result_cols = [c for c in needed_cols if c in df.columns] + [
            "latitude",
            "longitude",
        ]
        return df[result_cols].copy()
