import logging
import sys
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)

# GRIB shortName -> station DataFrame column
_PARAM_TO_COL = {
    "T_2M": "2t",
    "TD_2M": "2d",
    "U_10M": "10u",
    "V_10M": "10v",
    "PMSL": "msl",
    "PS": "sp",
    "TOT_PREC": "tp",
    "VMAX_10M": "vmax",
}

# Station column -> DWH (jretrieve) parameter names
_COL_TO_JR_PARAMS = {
    "2t": ["tre200s0"],
    "2d": ["tde200s0"],
    "10u": ["fkl010z0", "dkl010z0"],
    "10v": ["fkl010z0", "dkl010z0"],
    "msl": ["pp0qffs0"],
    "sp": ["prestas0"],
    "tp": ["rre150h0"],
    "vmax": ["fkl010z1"],
}


class RetrieveObservation(Filter):
    """Fetch surface station observations from the MeteoSwiss DWH via jretrieve.

    Retrieves the DWH parameters required for the requested variables,
    applies unit conversions to SI (°C→K, hPa→Pa, speed+direction→U/V),
    and saves the result as a Parquet file for use by the nudging filter.
    The data is passed through unchanged; this filter is used for its
    side effect of writing the observations file before nudging runs.

    Parameters
    ----------
    obs_path : str
        Path where the output Parquet file will be written.
    jretrieve_src_path : str
        Directory containing ``jretrieve.py``.
    station_group : str, optional
        DWH station group IDs (comma-separated) passed to jretrieve
        ``-a stn_group_id``. Mutually exclusive with ``retrieval_bbox``.
        Use this to match the truth data station selector exactly.
    retrieval_bbox : list, optional
        Bounding box ``[minlat, maxlat, minlon, maxlon]`` for station
        selection. Mutually exclusive with ``station_group``.
        Defaults to ``[40.5, 53.0, 0.0, 17.5]`` when neither station_group
        nor retrieval_bbox is specified.
    variables : list of str, optional
        GRIB shortNames to fetch (must be keys of ``_PARAM_TO_COL``).
        Defaults to all available variables.
    use_limitation : int, optional
        Passed to jretrieve ``--use-limitation``; limits the observation
        time window in minutes (e.g. 50 means observations within ±50 min).
    run_mode : str
        ``'depl'`` (default): ref_time = minimum valid_time across all
        fields. ``'devt'``: ref_time = valid_time of the first field.
    station_filter_mode : str, optional
        Extra trim applied to the retrieved stations, on top of
        *station_group*/*retrieval_bbox* — matches the station selection in
        ``notebooks/d_eff_generator.ipynb`` so the set of stations
        retrieved here stays a subset of whatever ``NudgeTowardObservation``'s
        ``d_eff_file`` cache was built from (a station outside that cache
        raises an error there rather than silently recomputing). One of:
        - ``None`` (default): no extra trim — retrieves exactly whatever
          *station_group*/*retrieval_bbox* selects, unchanged from before
          this parameter existed.
        - ``"domain"``: keep only stations inside *trim_bbox*.
        - ``"switzerland"``: keep only stations inside the real Swiss
          national border (Natural Earth ``admin_0_countries``,
          ``ADM0_A3 == "CHE"``).
    trim_bbox : list, optional
        ``[lat_min, lat_max, lon_min, lon_max]`` used to trim stations when
        *station_filter_mode* is ``"domain"``. Required in that case; unused
        otherwise. Not the same thing as *retrieval_bbox* above —
        *retrieval_bbox* controls what jretrieve itself queries, *trim_bbox*
        is a second, independent trim applied afterwards to the stations
        jretrieve returned.
    """

    def __init__(
        self,
        obs_path: str,
        jretrieve_src_path: str,
        station_group: str = None,
        retrieval_bbox: list = None,
        variables: list = None,
        use_limitation: int = None,
        run_mode: str = "depl",
        station_filter_mode: str = None,
        trim_bbox: list = None,
    ):
        if run_mode not in ("devt", "depl"):
            raise ValueError(f"run_mode must be 'devt' or 'depl', got {run_mode!r}")
        if station_group is not None and retrieval_bbox is not None:
            raise ValueError(
                "Specify at most one of 'station_group' or 'retrieval_bbox', not both."
            )
        if station_filter_mode not in (None, "domain", "switzerland"):
            raise ValueError(
                f"station_filter_mode must be None, 'domain', or 'switzerland', "
                f"got {station_filter_mode!r}"
            )
        if station_filter_mode == "domain" and trim_bbox is None:
            raise ValueError("trim_bbox is required when station_filter_mode='domain'.")

        self.obs_path = obs_path
        self.jretrieve_src_path = str(jretrieve_src_path)
        self.station_group = station_group
        self.retrieval_bbox = (
            retrieval_bbox
            if (retrieval_bbox is not None or station_group is not None)
            else [40.5, 53.0, 0.0, 17.5]
        )
        self.use_limitation = use_limitation
        self.run_mode = run_mode
        self.station_filter_mode = station_filter_mode
        self.trim_bbox = list(trim_bbox) if trim_bbox is not None else None

        if variables is not None:
            unknown = set(variables) - _PARAM_TO_COL.keys()
            if unknown:
                raise ValueError(
                    f"Unknown variables: {unknown}. Valid: {list(_PARAM_TO_COL)}"
                )
            self.cols = {_PARAM_TO_COL[v] for v in variables}
        else:
            self.cols = set(_PARAM_TO_COL.values())

        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Retrieve observations and write Parquet, then return *data* unchanged.

        Parameters
        ----------
        data : ekd.FieldList
            Forecast fields (used only to determine the reference time).

        Returns
        -------
        ekd.FieldList
            The input data, unchanged.
        """
        ref_time = (
            data[0].datetime()["valid_time"]
            if self.run_mode == "devt"
            else min(f.datetime()["valid_time"] for f in data)
        )
        LOG.info("Retrieving observations for %s", ref_time)
        self._retrieve(ref_time)
        return data

    def _retrieve(self, ref_time) -> None:
        if self.jretrieve_src_path not in sys.path:
            sys.path.insert(0, self.jretrieve_src_path)
        import jretrieve as jr

        jr_params = list(
            dict.fromkeys(
                p for col in self.cols for p in _COL_TO_JR_PARAMS.get(col, [])
            )
        )
        if not jr_params:
            raise ValueError(f"No jretrieve parameters found for columns: {self.cols}")

        jr.check_prerequisites()

        stations_sel = (
            {"group": self.station_group}
            if self.station_group is not None
            else {"bbox": self.retrieval_bbox}
        )
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
            stage="prod",
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
        df["elevation"] = df["station"].map(
            dict(zip(catalog.station_id, catalog.elevation))
        )
        df = df.dropna(subset=["nat_abbr"]).set_index("nat_abbr")
        df.index.name = "station"

        if self.station_filter_mode is not None:
            df = self._trim_stations(df)

        if "tre200s0" in df.columns:
            df["2t"] = df["tre200s0"] + 273.15
        if "tde200s0" in df.columns:
            df["2d"] = df["tde200s0"] + 273.15
        if "fkl010z0" in df.columns and "dkl010z0" in df.columns:
            dd_rad = np.deg2rad(df["dkl010z0"])
            df["10u"] = -df["fkl010z0"] * np.sin(dd_rad)
            df["10v"] = -df["fkl010z0"] * np.cos(dd_rad)
        if "pp0qffs0" in df.columns:
            df["msl"] = df["pp0qffs0"] * 100.0
        if "prestas0" in df.columns:
            df["sp"] = df["prestas0"] * 100.0
        if "rre150h0" in df.columns:
            df["tp"] = df["rre150h0"]
        if "fkl010z1" in df.columns:
            df["vmax"] = df["fkl010z1"]

        result_cols = [c for c in self.cols if c in df.columns] + [
            "latitude",
            "longitude",
            "elevation",
        ]
        df = df[result_cols].copy()

        for col in _PARAM_TO_COL.values():
            if col in df.columns:
                n_valid = int(df[col].notna().sum())
                LOG.info(
                    "Stations with valid %s: %d / %d stations", col, n_valid, len(df)
                )

        Path(self.obs_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.obs_path)
        LOG.info("Saved %d stations to %s", len(df), self.obs_path)

    def _trim_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim retrieved stations to *station_filter_mode* — same logic as
        the station-trim cell in notebooks/d_eff_generator.ipynb, so the
        stations retrieved here stay in sync with whatever
        NudgeTowardObservation's d_eff_file cache was built from."""
        if self.station_filter_mode == "domain":
            lat_min, lat_max, lon_min, lon_max = self.trim_bbox
            mask = (
                (df["latitude"] >= lat_min)
                & (df["latitude"] <= lat_max)
                & (df["longitude"] >= lon_min)
                & (df["longitude"] <= lon_max)
            )
            desc = f"domain bbox {self.trim_bbox}"

        elif self.station_filter_mode == "switzerland":
            import cartopy.io.shapereader as shpreader
            from shapely.geometry import Point

            shp_path = shpreader.natural_earth(
                resolution="10m", category="cultural", name="admin_0_countries"
            )
            ch_country = next(
                r
                for r in shpreader.Reader(shp_path).records()
                if r.attributes["ADM0_A3"] == "CHE"
            )
            swiss_geom = ch_country.geometry

            def _in_switzerland(lat, lon):
                if pd.isna(lat) or pd.isna(lon):
                    return False
                return swiss_geom.contains(Point(lon, lat))

            mask = [
                _in_switzerland(lat, lon)
                for lat, lon in zip(df["latitude"], df["longitude"])
            ]
            desc = "Swiss national border (Natural Earth)"

        else:
            raise ValueError(
                f"Unknown station_filter_mode: {self.station_filter_mode!r} "
                "(expected None, 'domain', or 'switzerland')"
            )

        n_before = len(df)
        df = df[mask]
        LOG.info("Station filter [%s]: %d -> %d stations", desc, n_before, len(df))
        return df
