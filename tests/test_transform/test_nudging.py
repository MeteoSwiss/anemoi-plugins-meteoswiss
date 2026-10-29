import numpy as np
import pytest
import xarray as xr

from anemoi_plugins_meteoswiss.transform.filters.nudging import NudgeTowardObservation
from anemoi_plugins_meteoswiss.transform.filters.nudging import ned_interp

# Placeholder input files for NudgeTowardObservation; the tests mock the loaders.
NUDGE_PATHS = {
    "d_eff_file": "d_eff.nc",
    "icon_orog_file": "icon_orog.nc",
}


def _make_mock_transformer():
    """WGS84 → LV95 transformer (real pyproj)."""
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)


def test_ned_interp_idw():
    """ned_interp produces finite IDW outputs when every POI has stations in range."""
    rng = np.random.default_rng(0)
    n_sta, n_poi = 5, 10
    sta_ids = [f"S{i}" for i in range(n_sta)]
    poi_ids = np.arange(n_poi)

    dist = xr.DataArray(
        rng.uniform(0.05, 0.25, (n_poi, n_sta)).astype(np.float32),
        dims=["poi", "sta"],
        coords={"poi": poi_ids, "sta": sta_ids},
    )
    residuals = xr.Dataset(
        {
            "T_2M": xr.DataArray(
                rng.standard_normal(n_sta).astype(np.float32),
                dims=["sta"],
                coords={"sta": sta_ids},
            )
        }
    )
    result = ned_interp(residuals, dist, max_dist=0.3, weight_power=4.0)
    assert "T_2M" in result
    assert result["T_2M"].shape == (n_poi,)
    assert np.all(np.isfinite(result["T_2M"].values))


def test_ned_interp_max_dist_masking():
    """Stations beyond max_dist contribute zero weight → result is NaN."""
    n_sta, n_poi = 3, 2
    sta_ids = ["A", "B", "C"]
    poi_ids = np.arange(n_poi)

    dist = xr.DataArray(
        np.full((n_poi, n_sta), 1.0, dtype=np.float32),
        dims=["poi", "sta"],
        coords={"poi": poi_ids, "sta": sta_ids},
    )
    residuals = xr.Dataset(
        {
            "T_2M": xr.DataArray(
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                dims=["sta"],
                coords={"sta": sta_ids},
            )
        }
    )
    result = ned_interp(residuals, dist, max_dist=0.5, weight_power=2.0)
    assert np.all(np.isnan(result["T_2M"].values))


def test_nudge_toward_observation_invalid_run_mode(tmp_path):
    """Invalid run_mode raises ValueError at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with (
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
    ):
        with pytest.raises(ValueError, match="run_mode"):
            NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, run_mode="bad")


def test_nudge_toward_observation_loads_cells_and_orography_from_extpar(tmp_path):
    """Cell centres (radians, float32) and orography are read from icon_orog_file."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()
    lat_deg = np.array([46.5, 47.0, 47.5])
    lon_deg = np.array([7.0, 8.0, 9.0])
    orog = np.array([400.0, 1500.0, 3000.0])
    extpar = tmp_path / "extpar.nc"
    xr.Dataset(
        {
            "clat": ("cell", np.radians(lat_deg).astype(np.float32)),
            "clon": ("cell", np.radians(lon_deg).astype(np.float32)),
            "topography_c": ("cell", orog.astype(np.float32)),
        }
    ).to_netcdf(extpar)

    with patch.object(NudgeTowardObservation, "_load_d_eff_cache"):
        filt = NudgeTowardObservation(obs_path=str(obs), **(NUDGE_PATHS | {"icon_orog_file": str(extpar)}))

    np.testing.assert_allclose(filt._lat_icon, lat_deg, atol=1e-5)
    np.testing.assert_allclose(filt._lon_icon, lon_deg, atol=1e-5)
    np.testing.assert_array_equal(filt._icon_orog, orog.astype(np.float32))
    assert filt._grid_xy_km.shape == (3, 2)


def test_nudge_toward_observation_holdout_station_file(tmp_path):
    """Stations listed in holdout_station_file are removed from the nudging set;
    IDs absent from the observations are ignored."""
    from unittest.mock import patch

    import pandas as pd

    obs = tmp_path / "obs.parquet"
    obs.touch()
    holdout_station_file = tmp_path / "holdout_stations.csv"
    holdout_station_file.write_text("nat_abbr\nBBB\nZZZ\nNA\n")

    with (
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
        patch.object(NudgeTowardObservation, "_load_d_eff_cache"),
    ):
        filt = NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, holdout_station_file=str(holdout_station_file))

    # "NA" is a valid station code and must not be read as missing.
    assert filt.holdout_stations == ["BBB", "ZZZ", "NA"]
    stations = pd.DataFrame(
        {"2t": [280.0, 281.0, 282.0, 283.0]}, index=pd.Index(["AAA", "BBB", "CCC", "NA"], name="station")
    )
    assert filt._apply_holdout(stations).index.tolist() == ["AAA", "CCC"]


def test_nudge_toward_observation_invalid_holdout_station_file(tmp_path):
    """A missing holdout_station_file, or a CSV without a nat_abbr column, raises at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()
    no_nat_abbr = tmp_path / "holdout_stations.csv"
    no_nat_abbr.write_text("station\nAAA\n")

    with (
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
    ):
        with pytest.raises(FileNotFoundError, match="Holdout station file"):
            NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, holdout_station_file=str(tmp_path / "missing.csv"))
        with pytest.raises(ValueError, match="'nat_abbr' column"):
            NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, holdout_station_file=str(no_nat_abbr))


def test_nudge_toward_observation_invalid_reliability_min_dist_frac(tmp_path):
    """reliability_min_dist_frac outside [0, 1] raises ValueError at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with (
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
    ):
        with pytest.raises(ValueError, match="reliability_min_dist_frac"):
            NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, reliability_min_dist_frac=1.5)


def test_nudge_toward_observation_invalid_number_of_std(tmp_path):
    """number_of_std <= 0 raises ValueError at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with (
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
    ):
        with pytest.raises(ValueError, match="number_of_std"):
            NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, number_of_std=0.0)


def _make_mock_d_eff_sta_cache(sta_ids: list, sta_xy: np.ndarray) -> dict:
    """Synthetic station<->station d_eff cache for _compute_reliability tests:
    on flat terrain (barrier=elev_diff=0), d_eff reduces to plain Euclidean
    distance. Self-distance is +inf,
    matching generate_d_eff_cache.py's own convention (a station is never its
    own neighbour). Only the two keys _get_d_eff_sta actually reads are
    included — see NudgeTowardObservation._load_d_eff_cache."""
    d = np.sqrt(((sta_xy[:, None, :] - sta_xy[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    d_eff_sta_full = xr.DataArray(
        d.astype(np.float32),
        dims=["sta_i", "sta"],
        coords={"sta_i": sta_ids, "sta": sta_ids},
    )
    return {"d_eff_sta_full": d_eff_sta_full, "sta_set": set(sta_ids)}


def test_compute_reliability_flags_outlier_station(tmp_path):
    """Leave-one-out spatial-consistency check: a station whose residual wildly
    disagrees with its neighbours gets reliability=0; consistent stations stay
    close to 1. Uses a synthetic d_eff cache (see _make_mock_d_eff_sta_cache),
    so no real ICON/d_eff files are needed."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    sta_ids = ["AAA", "BBB", "CCC", "DDD", "BAD"]
    st_lat = np.array([47.00, 47.02, 46.98, 47.01, 46.99])
    st_lon = np.array([8.00, 8.02, 7.98, 8.05, 8.01])
    transformer = _make_mock_transformer()
    sta_x, sta_y = transformer.transform(st_lon, st_lat)
    sta_xy = np.c_[sta_x, sta_y] / 1000.0
    r_at_st = np.array([0.0, -0.2, 0.2, -0.1, 5.0])  # "BAD" wildly disagrees
    d_eff_sta_cache = _make_mock_d_eff_sta_cache(sta_ids, sta_xy)

    with (
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
        patch.object(NudgeTowardObservation, "_load_d_eff_cache", return_value=d_eff_sta_cache),
    ):
        filt = NudgeTowardObservation(
            obs_path=str(obs),
            **NUDGE_PATHS,
            max_dist=50_000.0,  # meters (50 km)
            weight_power=2.0,
            lim_effective=0.0,
            number_of_std=4.0,
            reliability_min_dist_frac=0.1,
        )
    filt._wgs84_to_lv95 = transformer

    sta_res = xr.Dataset({"T_2M": xr.DataArray(r_at_st.astype(np.float32), dims=["sta"], coords={"sta": sta_ids})})

    reliability = filt._compute_reliability(
        "T_2M",
        sta_ids,
        r_at_st,
        sta_res,
        filt._max_dist_by_var["T_2M"],
    )

    assert reliability.dims == ("sta",)
    assert list(reliability["sta"].values) == sta_ids
    values = reliability.values
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert values[-1] == 0.0, f"BAD station should be fully rejected, got {values[-1]}"
    assert np.all(values[:-1] > values[-1]), "consistent stations must outrank BAD"


def test_compute_reliability_isolated_station_does_not_poison_others(tmp_path):
    """A station with no neighbour within max_dist gets e=NaN from ned_interp's
    leave-one-out call (min_count=1 makes a fully-masked POI return NaN, not 0).
    That NaN must not reach the median/MAD: otherwise every station's
    reliability becomes NaN, ned_interp masks every pair (`dist < NaN` is
    always False) and the whole field gets no correction. The isolated station
    itself gets reliability=1.0 (no neighbours to judge it against); every
    other, mutually consistent station stays finite and high."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    # AAA/BBB/CCC/DDD form a consistent cluster; ISO is ~190 km away — well
    # beyond max_dist=50 km, so it has zero neighbours in the leave-one-out check.
    sta_ids = ["AAA", "BBB", "CCC", "DDD", "ISO"]
    st_lat = np.array([47.00, 47.02, 46.98, 47.01, 46.20])
    st_lon = np.array([8.00, 8.02, 7.98, 8.05, 9.90])
    transformer = _make_mock_transformer()
    sta_x, sta_y = transformer.transform(st_lon, st_lat)
    sta_xy = np.c_[sta_x, sta_y] / 1000.0
    r_at_st = np.array([0.0, -0.2, 0.2, -0.1, 3.0])  # all mutually plausible values
    d_eff_sta_cache = _make_mock_d_eff_sta_cache(sta_ids, sta_xy)

    with (
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
        patch.object(NudgeTowardObservation, "_load_d_eff_cache", return_value=d_eff_sta_cache),
    ):
        filt = NudgeTowardObservation(
            obs_path=str(obs),
            **NUDGE_PATHS,
            max_dist=50_000.0,  # meters (50 km)
            weight_power=2.0,
            lim_effective=0.0,
            number_of_std=4.0,
            reliability_min_dist_frac=0.1,
        )
    filt._wgs84_to_lv95 = transformer

    sta_res = xr.Dataset({"T_2M": xr.DataArray(r_at_st.astype(np.float32), dims=["sta"], coords={"sta": sta_ids})})

    reliability = filt._compute_reliability(
        "T_2M",
        sta_ids,
        r_at_st,
        sta_res,
        filt._max_dist_by_var["T_2M"],
    )
    values = reliability.values

    assert not np.any(np.isnan(values)), f"NaN leaked into reliability: {values}"
    assert values[-1] == 1.0, f"isolated station should default to reliability=1.0, got {values[-1]}"
    assert np.all(values[:-1] > 0.5), f"consistent cluster stations should stay trusted, got {values[:-1]}"


def test_nudge_field_writes_diagnostics(tmp_path):
    """With write_diagnostics=True, _nudge_field writes one NetCDF holding the
    holdin and holdout station residuals, the reliability results and the
    gridded correction; with it False, nothing is written."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import pandas as pd
    from scipy.spatial import cKDTree

    obs = tmp_path / "obs.parquet"
    obs.touch()
    transformer = _make_mock_transformer()

    # 15 x 15 grid of "ICON cells" around 8E/47N, flat 500 m orography.
    lon_g, lat_g = np.meshgrid(np.linspace(7.8, 8.2, 15), np.linspace(46.9, 47.1, 15))
    lon_icon, lat_icon = lon_g.ravel(), lat_g.ravel()
    gx, gy = transformer.transform(lon_icon, lat_icon)
    grid_xy = np.c_[gx, gy] / 1000.0

    all_ids = ["AAA", "BBB", "CCC", "DDD", "HHH"]
    all_stations = pd.DataFrame(
        {
            "latitude": [47.00, 47.02, 46.98, 47.01, 47.03],
            "longitude": [8.00, 8.02, 7.98, 8.05, 7.95],
            "elevation": [500.0] * 5,
            "2t": [280.0, 280.2, 279.8, 280.1, 280.3],
        },
        index=pd.Index(all_ids, name="station"),
    )
    stations = all_stations.drop(index=["HHH"])
    held_out = all_stations.loc[["HHH"]]
    sta_ids = stations.index.tolist()

    sx, sy = transformer.transform(stations["longitude"].to_numpy(), stations["latitude"].to_numpy())
    sta_xy = np.c_[sx, sy] / 1000.0
    d_poi = np.sqrt(((grid_xy[:, None, :] - sta_xy[None, :, :]) ** 2).sum(axis=-1)).astype(np.float32)
    d_eff_poi_full = xr.DataArray(d_poi, dims=["poi", "sta"], coords={"poi": np.arange(len(lon_icon)), "sta": sta_ids})
    cache = _make_mock_d_eff_sta_cache(sta_ids, sta_xy) | {
        "d_eff_poi_full": d_eff_poi_full,
        "poi_index": pd.Index(d_eff_poi_full["poi"].values),
        "sta_index": pd.Index(sta_ids),
        "poi_set": set(range(len(lon_icon))),
    }

    def _make_filter(write_diagnostics):
        with (
            patch.object(NudgeTowardObservation, "_load_icon_orog"),
            patch.object(NudgeTowardObservation, "_project_icon_grid"),
            patch.object(NudgeTowardObservation, "_load_d_eff_cache", return_value=cache),
        ):
            filt = NudgeTowardObservation(
                obs_path=str(obs),
                **NUDGE_PATHS,
                nudge_variables=["T_2M"],
                use_reliability_check=True,
                write_diagnostics=write_diagnostics,
                diagnostics_dir=str(tmp_path / "diag"),
            )
        filt._lat_icon, filt._lon_icon = lat_icon, lon_icon
        filt._icon_orog = np.full(len(lon_icon), 500.0, dtype=np.float32)
        filt._wgs84_to_lv95 = transformer
        filt._grid_xy_km = grid_xy
        filt._grid_tree = cKDTree(grid_xy)
        return filt

    field = SimpleNamespace(values=np.full(len(lon_icon), 281.0, dtype=np.float32))
    ref_time = pd.Timestamp("2026-01-01 06:00").to_pydatetime()

    _make_filter(False)._nudge_field(field, stations, "T_2M", "2t", ref_time, held_out)
    assert not (tmp_path / "diag").exists()

    corrected = _make_filter(True)._nudge_field(field, stations, "T_2M", "2t", ref_time, held_out)
    ds = xr.open_dataset(tmp_path / "diag" / "nudging_diag_T_2M_202601010600.nc")

    assert list(ds["station"].values) == ["AAA", "BBB", "CCC", "DDD", "HHH"]
    assert list(ds["is_holdout"].values) == [0, 0, 0, 0, 1]
    np.testing.assert_allclose(ds["residual_pre"].values, 281.0 - all_stations["2t"].to_numpy(), rtol=1e-6)
    assert np.all(np.isfinite(ds["reliability"].values[:4]))
    assert np.isnan(ds["reliability"].values[4])
    np.testing.assert_allclose(
        ds["correction"].values, field.values[ds["cell"].values] - corrected[ds["cell"].values], atol=1e-4
    )
    assert ds.attrs["variable"] == "T_2M"
    assert ds.sizes["qc_station"] == 0
