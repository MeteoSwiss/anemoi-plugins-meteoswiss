import logging
import sys
from datetime import timedelta
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from peakweather.dataset import PeakWeatherDataset
from scipy.spatial import cKDTree
import xarray as xr

LOG = logging.getLogger(__name__)

# Maps GRIB shortName -> (station DataFrame column, unit offset applied to obs before nudging)
# Uses COSMO/ICON shortNames as output by the LAM forecaster
# Maps GRIB shortName -> (station DataFrame column, unit offset applied to obs before nudging)
PARAM_MAP = {
    "T_2M":     ("2t",   0.0),  # 2 m temperature        [K]
    "TD_2M":    ("2d",   0.0),  # 2 m dewpoint           [K]
    "U_10M":    ("10u",  0.0),  # 10 m wind U component  [m/s]
    "V_10M":    ("10v",  0.0),  # 10 m wind V component  [m/s]
    "PMSL":     ("msl",  0.0),  # mean sea-level pressure [Pa]
    "TOT_PREC": ("tp",   0.0),  # hourly precipitation   [kg m-2]
    "VMAX_10M": ("vmax", 0.0),  # 10 m wind gust         [m/s]
}

# Maps station column -> PeakWeather parameter names required to produce it
_STATION_COL_TO_PW_PARAMS = {
    "2t":   ["temperature"],
    "2d":   [],                    # not available in PeakWeather
    "10u":  ["wind_u", "wind_v"],  # U and V are derived together
    "10v":  ["wind_u", "wind_v"],
    "msl":  [],                    # not available in PeakWeather
    "tp":   ["precipitation"],
    "vmax": [],                    # not available in PeakWeather
}

# Maps station column -> DWH (jretrieve) parameter names required to produce it
_STATION_COL_TO_JR_PARAMS = {
    "2t":   ["tre200s0"],                   # 2 m temperature    [°C -> K]
    "2d":   ["tde200s0"],                   # 2 m dewpoint       [°C -> K]
    "10u":  ["fkl010z0", "dkl010z0"],       # wind speed + dir   -> U/V [m/s]
    "10v":  ["fkl010z0", "dkl010z0"],
    "msl":  ["pp0qffs0"],                   # MSLP               [hPa -> Pa]
    "tp":   ["rre150h0"],                   # hourly precip      [mm = kg m-2]
    "vmax": ["fkl010z1"],                   # 10 m wind gust     [m/s]
}



def idw_points(xy_obs, v_obs, xy_tgt, k=5, power=2):
    """IDW from scattered obs -> scattered targets (all in 2D)."""
    tree = cKDTree(np.asarray(xy_obs, float))
    dist, idx = tree.query(np.asarray(xy_tgt, float), k=k)

    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    dist = dist + 1e-12
    w = 1.0 / (dist**power)
    v_obs = np.asarray(v_obs, float)

    return (w * v_obs[idx]).sum(axis=1) / w.sum(axis=1)


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
    """Interpolation of Residuals (IoR).

    1. Project coordinates (lon scaled by cos(lat)) for approximate metric distances.
    2. Find nearest grid point per station; extract background value there.
    3. Compute residuals: background_at_station - observation.
    4. Spread residuals to full grid with IDW.
    5. Taper with a linear ramp to zero beyond max_dist degrees from nearest station.
    6. Return background - tapered_residuals.
    """
    background = np.asarray(background, float).ravel()
    lat = np.asarray(grid_lat, float).ravel()
    lon = np.asarray(grid_lon, float).ravel()
    st_obs = np.asarray(st_obs, float)
    st_lon = np.asarray(st_lon, float)
    st_lat = np.asarray(st_lat, float)

    lat0 = np.deg2rad(np.nanmean(lat))

    grid_xy = np.c_[lon * np.cos(lat0), lat]
    st_xy = np.c_[st_lon * np.cos(lat0), st_lat]

    ok = np.isfinite(st_obs) & np.isfinite(st_xy).all(axis=1)
    st_xy = st_xy[ok]
    st_obs = st_obs[ok]

    grid_tree = cKDTree(grid_xy)
    _, gi = grid_tree.query(st_xy, k=1)
    b_at_st = background[gi]

    r_at_st = b_at_st - st_obs
    res = idw_points(st_xy, r_at_st, grid_xy, k=k, power=power)

    tree = cKDTree(st_xy)
    dmin, _ = tree.query(grid_xy, k=1)
    w = 1.0 - np.clip(dmin / max_dist, 0.0, 1.0)
    res = w * res

    return background - res


class NudgeTowardObservation(Filter):
    """A filter that nudges the forecast initial condition toward station observations.

    Uses Interpolation of Residuals (IoR): station residuals (background minus obs)
    are spread to the full grid via IDW and subtracted from the background field,
    with a distance-based taper beyond max_dist degrees from the nearest station.
    """

    def __init__(
        self,
        path_to_observation: str = "/scratch/mch/llanzila/sruc/evalml/output/data/observation/PeakWeather",
        k: int = 3,
        power: float = 4.0,
        max_dist: float = 0.5,
        backend: str = "peakweather",
        jretrieve_bbox: list = None,
        jretrieve_src_path: str = "/scratch/mch/llanzila/sruc/evalml/src",
        nudge_variables: list = None,
        use_limitation: int = None,
    ):
        """Initialize the filter.

        Parameters
        ----------
        path_to_observation : str
            Root path of the PeakWeather dataset (used when backend='peakweather').
        k : int
            Number of nearest observation stations used in IDW interpolation.
        power : float
            IDW distance-decay exponent.
        max_dist : float
            Tapering radius in degrees; correction fades to zero beyond this distance
            from the nearest station.
        backend : str
            Observation source: 'peakweather' (default) or 'jretrieve'.
        jretrieve_bbox : list, optional
            Bounding box [minlat, maxlat, minlon, maxlon] for station selection when
            backend='jretrieve'. Defaults to a wide domain covering Switzerland and
            surroundings: [40.5, 53.0, 0.0, 17.5].
        jretrieve_src_path : str
            Path to the directory containing the ``data_input`` package with
            ``jretrieve.py`` (used when backend='jretrieve').
        nudge_variables : list, optional
            GRIB shortNames to nudge, e.g. ``["T_2M"]``. Must be keys of PARAM_MAP.
            Defaults to all variables in PARAM_MAP.
        use_limitation : int, optional
            Passed to jretrieve ``--use-limitation`` flag (used when backend='jretrieve').
            Limits the time window (in minutes) used to select observations. E.g. 50
            means only observations within ±50 minutes of the target time are used.
        """
        if backend not in ("peakweather", "jretrieve"):
            raise ValueError(f"backend must be 'peakweather' or 'jretrieve', got {backend!r}")
        self.path_to_observation = Path(path_to_observation)
        self.k = k
        self.power = power
        self.max_dist = max_dist
        self.backend = backend
        self.jretrieve_bbox = jretrieve_bbox if jretrieve_bbox is not None else [40.5, 53.0, 0.0, 17.5]
        self.jretrieve_src_path = jretrieve_src_path
        self.use_limitation = use_limitation
        if nudge_variables is not None:
            unknown = set(nudge_variables) - PARAM_MAP.keys()
            if unknown:
                raise ValueError(f"Unknown variables for nudging: {unknown}. Valid: {list(PARAM_MAP)}")
            self.param_map = {k: PARAM_MAP[k] for k in nudge_variables}
        else:
            self.param_map = PARAM_MAP
        self._nudging_done = False  # nudging is applied exactly once (first forward call)
        LOG.info(
            "Initialised nudging filter: backend='%s', observations='%s', variables=%s, k=%d, power=%.1f, max_dist=%.2f",
            self.backend,
            self.path_to_observation,
            list(self.param_map.keys()),
            self.k,
            self.power,
            self.max_dist,
        )
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply nudging toward observations.

        Parameters
        ----------
        data : ekd.FieldList
            Forecast initial condition fields.

        Returns
        -------
        ekd.FieldList
            Same fields as the input, with nudgeable surface parameters corrected
            toward station observations.
        """
        LOG.info("Fields in data (%d total):", len(data))
        # for f in data:
        #     LOG.info(
        #         "  shortName=%s, levtype=%s, level=%s, validityDate=%s validityTime=%s, shape=%s",
        #         f.metadata("shortName"),
        #         f.metadata("typeOfLevel"),
        #         f.metadata("level"),
        #         f.metadata("validityDate"),
        #         f.metadata("validityTime"),
        #         f.shape,
        #     )
        
        if self._nudging_done:
            LOG.info("Nudging already applied, passing through unchanged")
            return data

        LOG.info(
            "Applying nudging at date=%s, observations='%s'",
            data[0].datetime()["valid_time"], self.path_to_observation,
        )

        LOG.info("Loading grid coordinates from NC file")
        # ds = xr.open_dataset("/scratch/mch/llanzila/sruc/aux_files/icon_grid_0001_R19B08_mch.nc")
        ds = xr.open_dataset("/data/aux_files/icon_grid_0001_R19B08_mch.nc")
        lat_icon = np.degrees(ds["clat"])
        lon_icon = np.degrees(ds["clon"])
        LOG.info("Grid loaded: %d points, lat=[%.3f, %.3f], lon=[%.3f, %.3f]",
                 len(lat_icon), float(lat_icon.min()), float(lat_icon.max()),
                 float(lon_icon.min()), float(lon_icon.max()))

        ref_time = data[0].datetime()["valid_time"]
        LOG.info("Reference time: '%s'", ref_time)

        LOG.info("Loading station observations from '%s'", self.path_to_observation)
        stations = self._load_stations(ref_time)
        LOG.info("Stations loaded: %d total, columns=%s", len(stations), list(stations.columns))
        for col in ["2t", "10u", "10v"]:
            if col in stations.columns:
                LOG.info("  %s: %d valid / %d total", col, int(stations[col].notna().sum()), len(stations))

        nudgeable_shortnames = list(self.param_map.keys())
        fields_to_nudge = data.sel(shortName=nudgeable_shortnames)
        LOG.info("Fields selected for nudging: %d (shortNames=%s, valid_time=%s)", len(fields_to_nudge), nudgeable_shortnames, ref_time)

        nudged = {}
        for field in fields_to_nudge:
            shortname = field.metadata("shortName")
            step = field.datetime()["valid_time"]

            if step != ref_time:
                LOG.info("Skipping '%s' at step %s (not initial condition)", shortname, step)
                continue

            pw_param, offset = self.param_map[shortname]
            LOG.info("Processing field '%s' -> pw_param='%s', step=%s", shortname, pw_param, step)

            if pw_param not in stations.columns or stations[pw_param].isna().all():
                LOG.warning("No observations for '%s' (pw_param='%s'), skipping", shortname, pw_param)
                continue

            valid = stations[pw_param].notna()
            st_lat = stations.loc[valid, "latitude"].to_numpy()
            st_lon = stations.loc[valid, "longitude"].to_numpy()
            st_obs = stations.loc[valid, pw_param].to_numpy() + offset
            LOG.info("  background shape=%s, n_stations=%d", field.values.shape, int(valid.sum()))
            LOG.info("  background: min=%.4f, max=%.4f, mean=%.4f", float(field.values.min()), float(field.values.max()), float(field.values.mean()))
            LOG.info("  obs: min=%.4f, max=%.4f, mean=%.4f", float(st_obs.min()), float(st_obs.max()), float(st_obs.mean()))

            LOG.info("  Running interpolation_of_residuals (k=%d, power=%.1f, max_dist=%.2f)", self.k, self.power, self.max_dist)
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
            LOG.info("  corrected: min=%.4f, max=%.4f, mean=%.4f", float(corrected.min()), float(corrected.max()), float(corrected.mean()))

            LOG.info("  Creating new field from numpy")
            nudged[shortname] = new_field_from_numpy(
                corrected,
                template=field,
                validityDate=field.metadata("validityDate"),
                validityTime=field.metadata("validityTime"),
                dataDate=field.metadata("dataDate"),
                dataTime=field.metadata("dataTime"),
            )
            LOG.info("Nudged '%s' at step %s using %d stations", shortname, step, int(valid.sum()))

        result = [nudged.get(f.metadata("shortName"), f) if f.datetime()["valid_time"] == ref_time else f for f in data]
        LOG.info("Nudging complete: %d fields returned, %d nudged", len(result), len(nudged))
        self._nudging_done = True # apply nudging to initial condition only
        
        LOG.info("Fields in data (%d total):", len(new_fieldlist_from_list(result)))
        # for f in new_fieldlist_from_list(result):
        #     LOG.info(
        #         "  shortName=%s, levtype=%s, level=%s, validityDate=%s validityTime=%s, shape=%s",
        #         f.metadata("shortName"),
        #         f.metadata("typeOfLevel"),
        #         f.metadata("level"),
        #         f.metadata("validityDate"),
        #         f.metadata("validityTime"),
        #         f.shape,
        #     )
        
        return new_fieldlist_from_list(result)  #data #

    def _load_stations(self, ref_time) -> pd.DataFrame:
        """Dispatch to the configured observation backend."""
        if self.backend == "jretrieve":
            return self._load_stations_jretrieve(ref_time)
        return self._load_stations_peakweather(ref_time)

    def _load_stations_peakweather(self, ref_time) -> pd.DataFrame:
        """Load observations at ref_time from a local PeakWeather dataset."""
        needed_cols = {col for col, _ in self.param_map.values()}
        pw_params = list(dict.fromkeys(
            p for col in needed_cols for p in _STATION_COL_TO_PW_PARAMS.get(col, [])
        ))
        LOG.info("Initialising PeakWeatherDataset from '%s'", self.path_to_observation)
        peakweather = PeakWeatherDataset(
            root=self.path_to_observation, freq="1h", compute_uv=True
        )
        first_date = f"{ref_time - timedelta(hours=1):%Y-%m-%d %H:%M}"
        last_date = f"{ref_time:%Y-%m-%d %H:%M}"
        LOG.info("Fetching observations: first_date='%s', last_date='%s', params=%s", first_date, last_date, pw_params)
        obs, mask = peakweather.get_observations(
            parameters=pw_params,
            first_date=first_date,
            last_date=last_date,
            return_mask=True,
        )
        LOG.info("Raw obs shape: %s, valid mask sum: %d", obs.shape, int(mask.iloc[0].sum()))

        obs = obs.loc[:, mask.iloc[0]]
        obs = obs.iloc[0]
        obs.index = obs.index.set_names(["station", "parameter"])
        obs = obs.unstack("parameter").sort_index().sort_index(axis=1)
        LOG.info("Obs after unstacking: %d stations, columns=%s", len(obs), list(obs.columns))

        peakweather.stations_table.index.names = ["station"]
        stations = pd.concat([obs, peakweather.stations_table], axis=1)
        stations = stations.rename(columns={
            "temperature":   "2t",
            "wind_u":        "10u",
            "wind_v":        "10v",
            "precipitation": "tp",
        })
        if "2t" in stations.columns:
            stations["2t"] += 273.15  # °C -> K
        LOG.info("Stations table built: %d rows", len(stations))

        return stations

    def _load_stations_jretrieve(self, ref_time) -> pd.DataFrame:
        """Load observations at ref_time from the DWH via jretrieve."""
        src_path = self.jretrieve_src_path
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from data_input import jretrieve as jr

        needed_cols = {col for col, _ in self.param_map.values()}
        jr_params = list(dict.fromkeys(
            p for col in needed_cols for p in _STATION_COL_TO_JR_PARAMS.get(col, [])
        ))
        if not jr_params:
            raise ValueError(f"No jretrieve parameters found for station columns: {needed_cols}")

        stations_sel = {"bbox": self.jretrieve_bbox}
        LOG.info("Checking jretrieve prerequisites")
        jr.check_prerequisites()

        LOG.info("Fetching station metadata (bbox=%s, params=%s)", self.jretrieve_bbox, jr_params)
        meta = jr.fetch_meta(stations=stations_sel, params=jr_params)
        catalog = jr.StationCatalog.from_meta(meta)
        LOG.info("Catalog built: %d stations", catalog.n)

        LOG.info("Fetching observations at %s (use_limitation=%s)", ref_time, self.use_limitation)
        df = jr.fetch_data(
            stations=stations_sel,
            params=jr_params,
            start=ref_time,
            end=ref_time,
            increment_minutes=60,
            use_limitation=self.use_limitation,
        )

        id_to_abbr = dict(zip(catalog.station_id, catalog.nat_abbr))
        id_to_lat  = dict(zip(catalog.station_id, catalog.latitude))
        id_to_lon  = dict(zip(catalog.station_id, catalog.longitude))

        df["nat_abbr"]  = df["station"].map(id_to_abbr)
        df["latitude"]  = df["station"].map(id_to_lat)
        df["longitude"] = df["station"].map(id_to_lon)
        df = df.dropna(subset=["nat_abbr"]).set_index("nat_abbr")
        df.index.name = "station"

        if "tre200s0" in df.columns:
            df["2t"] = df["tre200s0"] + 273.15       # °C -> K
        if "tde200s0" in df.columns:
            df["2d"] = df["tde200s0"] + 273.15       # °C -> K
        if "fkl010z0" in df.columns and "dkl010z0" in df.columns:
            dd_rad    = np.deg2rad(df["dkl010z0"])
            df["10u"] = -df["fkl010z0"] * np.sin(dd_rad)
            df["10v"] = -df["fkl010z0"] * np.cos(dd_rad)
        if "pp0qffs0" in df.columns:
            df["msl"] = df["pp0qffs0"] * 100.0       # hPa -> Pa
        if "rre150h0" in df.columns:
            df["tp"] = df["rre150h0"]                # mm = kg m-2, no conversion
        if "fkl010z1" in df.columns:
            df["vmax"] = df["fkl010z1"]              # m/s, no conversion

        result_cols = [c for c in needed_cols if c in df.columns] + ["latitude", "longitude"]
        stations = df[result_cols].copy()
        LOG.info("Stations table built: %d rows, columns=%s", len(stations), result_cols)
        return stations
