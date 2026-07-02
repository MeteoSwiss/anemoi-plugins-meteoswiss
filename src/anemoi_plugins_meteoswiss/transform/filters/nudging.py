import logging
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

# Maps GRIB shortName -> (PeakWeather column, unit offset applied to obs before nudging)
# Uses COSMO/ICON shortNames as output by the LAM forecaster
PARAM_MAP = {
    "T_2M":   ("2t",            0.0),  # already in K after conversion below
    "U_10M":  ("10u",           0.0),
    "V_10M":  ("10v",           0.0),
    "TOT_PREC": ("precipitation", 0.0),
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
    ):
        """Initialize the filter.

        Parameters
        ----------
        path_to_observation : str
            Root path of the PeakWeather dataset.
        k : int
            Number of nearest observation stations used in IDW interpolation.
        power : float
            IDW distance-decay exponent.
        max_dist : float
            Tapering radius in degrees; correction fades to zero beyond this distance
            from the nearest station.
        """
        self.path_to_observation = Path(path_to_observation)
        self.k = k
        self.power = power
        self.max_dist = max_dist
        self._nudging_done = False  # nudging is applied exactly once (first forward call)
        LOG.info(
            "Initialised nudging filter: observations='%s', k=%d, power=%.1f, max_dist=%.2f",
            self.path_to_observation,
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
        ds = xr.open_dataset("/scratch/mch/llanzila/sruc/aux_files/icon_grid_0001_R19B08_mch.nc")
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

        nudgeable_shortnames = list(PARAM_MAP.keys())
        fields_to_nudge = data.sel(shortName=nudgeable_shortnames)
        LOG.info("Fields selected for nudging: %d (shortNames=%s, valid_time=%s)", len(fields_to_nudge), nudgeable_shortnames, ref_time)

        nudged = {}
        for field in fields_to_nudge:
            shortname = field.metadata("shortName")
            step = field.datetime()["valid_time"]

            if step != ref_time:
                LOG.info("Skipping '%s' at step %s (not initial condition)", shortname, step)
                continue

            pw_param, offset = PARAM_MAP[shortname]
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
        """Load PeakWeather observations at ref_time and join with station metadata."""
        LOG.info("Initialising PeakWeatherDataset from '%s'", self.path_to_observation)
        peakweather = PeakWeatherDataset(
            root=self.path_to_observation, freq="1h", compute_uv=True
        )
        first_date = f"{ref_time - timedelta(hours=1):%Y-%m-%d %H:%M}"
        last_date = f"{ref_time:%Y-%m-%d %H:%M}"
        LOG.info("Fetching observations: first_date='%s', last_date='%s'", first_date, last_date)
        obs, mask = peakweather.get_observations(
            parameters=["temperature", "humidity", "wind_u", "wind_v"],
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
        stations = stations.rename(columns={"temperature": "2t", "wind_u": "10u", "wind_v": "10v"})
        stations["2t"] += 273.15  # °C -> K
        LOG.info("Stations table built: %d rows", len(stations))

        return stations
