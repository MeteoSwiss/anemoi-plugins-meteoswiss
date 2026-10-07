import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField
from earthkit.meteo.thermo import relative_humidity_from_dewpoint

from anemoi_plugins_meteoswiss.transform import filters
from anemoi_plugins_meteoswiss.transform.filters import GaussianSmoother
from anemoi_plugins_meteoswiss.transform.filters import IconRemapToRegLatLon
from anemoi_plugins_meteoswiss.transform.filters import Keep
from anemoi_plugins_meteoswiss.transform.filters import ModelToPressureLevel
from anemoi_plugins_meteoswiss.transform.filters import SurfaceDiagnostics
from anemoi_plugins_meteoswiss.transform.filters.nudging import NudgeTowardObservation
from anemoi_plugins_meteoswiss.transform.filters.nudging import ned_interp

ICONREMAP_WEIGHTS = "/store_new/mch/msopr/icon_workflow_2/iconremap-weights/icon-ch1-eps-rotlatlon.nc"

# Placeholder input files for NudgeTowardObservation; the tests mock the loaders.
NUDGE_PATHS = {
    "icon_grid_file": "icon_grid.nc",
    "d_eff_file": "d_eff.nc",
    "icon_orog_file": "icon_orog.nc",
}


def test_filter_imports():
    # Minimal test to prevent pytest exit code 5 (no tests collected) in CI.
    assert hasattr(filters, "AverageFluxToCumulativeQuantity")
    assert hasattr(filters, "AssignGrid")
    assert hasattr(filters, "GeopotentialFromHeight")


def test_icon_remap_to_reg_lat_lon(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    regridder = IconRemapToRegLatLon(ICONREMAP_WEIGHTS)

    assert regridder.ny == 786
    assert regridder.nx == 1170

    # Geographic lat/lon for ICON-CH1 domain (Switzerland + surroundings)
    assert regridder._latitudes.min() > 40.0
    assert regridder._latitudes.max() < 52.0
    assert regridder._longitudes.min() > 0.0
    assert regridder._longitudes.max() < 22.0

    fn = str(data_dir / "iaf2025010100")
    fieldlist = ekd.from_source("file", fn).sel(shortName="T_2M")

    result = regridder.forward(fieldlist)

    n_out = regridder.ny * regridder.nx
    assert len(result) == len(fieldlist)
    for src_field, out_field in zip(fieldlist, result):
        src_values = src_field.to_numpy(flatten=True)
        values = out_field.to_numpy(flatten=True)
        assert values.shape == (n_out,)
        # All output points are valid for this weights file (no sentinel zeros)
        assert not np.any(np.isnan(values))
        # Conservative interpolation: output stays within the source's range.
        assert values.min() >= src_values.min() - 1e-6
        assert values.max() <= src_values.max() + 1e-6


def test_gaussian_smoother(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    regridder = IconRemapToRegLatLon(ICONREMAP_WEIGHTS)
    smoother = GaussianSmoother(sigma=5, params=["T_2M"])

    fn = str(data_dir / "iaf2025010100")
    fs = ekd.from_source("file", fn)
    # T_2M (smoothed) + one level of W (pass-through, not in params)
    fieldlist = new_fieldlist_from_list(list(fs.sel(shortName="T_2M")) + [fs.sel(shortName="W")[0]])

    regridded = list(regridder.forward(fieldlist))
    synthetic = np.zeros((regridder.ny, regridder.nx))
    synthetic[::20, ::20] = 100.0
    regridded[0] = new_field_from_numpy(synthetic.ravel(), template=regridded[0])
    regridded = new_fieldlist_from_list(regridded)

    smoothed = smoother.forward(regridded)

    assert len(smoothed) == len(regridded) == 2

    t2m_raw = regridded[0].to_numpy(flatten=True)
    t2m_smo = smoothed[0].to_numpy(flatten=True)
    w_raw = regridded[1].to_numpy(flatten=True)
    w_smo = smoothed[1].to_numpy(flatten=True)

    # Smoothed T_2M must differ from raw (sigma=5 is a meaningful kernel)
    assert not np.allclose(t2m_raw, t2m_smo)
    # Smoothed values stay within the original range (conservative property)
    assert t2m_smo.min() >= t2m_raw.min() - 1e-6
    assert t2m_smo.max() <= t2m_raw.max() + 1e-6

    # W was not in params — must be bit-for-bit identical
    np.testing.assert_array_equal(w_raw, w_smo)


# ── NudgeTowardObservation unit tests ─────────────────────────────────────


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
        patch.object(NudgeTowardObservation, "_load_icon_grid"),
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
    ):
        with pytest.raises(ValueError, match="run_mode"):
            NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, run_mode="bad")


def test_nudge_toward_observation_holdout_station_file(tmp_path):
    """Stations listed in holdout_station_file are removed from the nudging set;
    IDs absent from the observations are ignored."""
    from unittest.mock import patch

    import pandas as pd

    obs = tmp_path / "obs.parquet"
    obs.touch()
    holdout_station_file = tmp_path / "holdout.yaml"
    holdout_station_file.write_text("- BBB\n- ZZZ\n")

    with (
        patch.object(NudgeTowardObservation, "_load_icon_grid"),
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
        patch.object(NudgeTowardObservation, "_load_d_eff_cache"),
    ):
        filt = NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, holdout_station_file=str(holdout_station_file))

    assert filt.holdout_stations == ["BBB", "ZZZ"]
    stations = pd.DataFrame({"2t": [280.0, 281.0, 282.0]}, index=pd.Index(["AAA", "BBB", "CCC"], name="station"))
    assert filt._apply_holdout(stations).index.tolist() == ["AAA", "CCC"]


def test_nudge_toward_observation_invalid_holdout_station_file(tmp_path):
    """A missing holdout_station_file, or one that is not a list of station IDs, raises at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()
    not_a_list = tmp_path / "holdout.yaml"
    not_a_list.write_text("stations: [AAA]\n")

    with (
        patch.object(NudgeTowardObservation, "_load_icon_grid"),
        patch.object(NudgeTowardObservation, "_load_icon_orog"),
        patch.object(NudgeTowardObservation, "_project_icon_grid"),
    ):
        with pytest.raises(FileNotFoundError, match="Holdout station file"):
            NudgeTowardObservation(
                obs_path=str(obs), **NUDGE_PATHS, holdout_station_file=str(tmp_path / "missing.yaml")
            )
        with pytest.raises(ValueError, match="YAML list"):
            NudgeTowardObservation(obs_path=str(obs), **NUDGE_PATHS, holdout_station_file=str(not_a_list))


def test_nudge_toward_observation_invalid_reliability_min_dist_frac(tmp_path):
    """reliability_min_dist_frac outside [0, 1] raises ValueError at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with (
        patch.object(NudgeTowardObservation, "_load_icon_grid"),
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
        patch.object(NudgeTowardObservation, "_load_icon_grid"),
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
        patch.object(NudgeTowardObservation, "_load_icon_grid"),
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
        patch.object(NudgeTowardObservation, "_load_icon_grid"),
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
            patch.object(NudgeTowardObservation, "_load_icon_grid"),
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


def _synthetic_components(data_dir):
    template = ekd.from_source("file", str(data_dir / "iaf2025010100")).sel(shortName="T_2M")[0]

    rng = np.random.default_rng(0)
    n = template.to_numpy(flatten=True).size
    u_values = rng.uniform(-20, 20, n)
    v_values = rng.uniform(-20, 20, n)
    t_values = rng.uniform(250.0, 300.0, n)
    # Dewpoint never exceeds temperature; keep it 0-20 K below.
    td_values = t_values - rng.uniform(0.0, 20.0, n)

    u = new_field_from_numpy(u_values, template=template, param="10u", shortName="10u")
    v = new_field_from_numpy(v_values, template=template, param="10v", shortName="10v")
    t = new_field_from_numpy(t_values, template=template, param="2t", shortName="2t")
    td = new_field_from_numpy(td_values, template=template, param="2d", shortName="2d")
    fieldlist = new_fieldlist_from_list([u, v, t, td])
    return fieldlist, u_values, v_values, t_values, td_values


def test_surface_diagnostics_all_variables(data_dir):
    fieldlist, u_values, v_values, t_values, td_values = _synthetic_components(data_dir)

    diagnostics = SurfaceDiagnostics(
        u_component="10u",
        v_component="10v",
        temperature="2t",
        dewpoint="2d",
        variables=["SP_10M", "DD_10M", "RELHUM_2M"],
    )
    result = diagnostics.forward(fieldlist)

    assert len(result) == 7
    by_param = {f.metadata("param"): f for f in result}
    assert set(by_param) == {"10u", "10v", "2t", "2d", "SP_10M", "DD_10M", "RELHUM_2M"}

    # Originals are passed through untouched.
    np.testing.assert_array_equal(by_param["10u"].to_numpy(flatten=True), u_values)
    np.testing.assert_array_equal(by_param["10v"].to_numpy(flatten=True), v_values)
    np.testing.assert_array_equal(by_param["2t"].to_numpy(flatten=True), t_values)
    np.testing.assert_array_equal(by_param["2d"].to_numpy(flatten=True), td_values)

    expected_speed = np.sqrt(u_values**2 + v_values**2)
    np.testing.assert_allclose(by_param["SP_10M"].to_numpy(flatten=True), expected_speed)

    expected_direction = np.mod(np.degrees(np.arctan2(-u_values, -v_values)), 360.0)
    np.testing.assert_allclose(by_param["DD_10M"].to_numpy(flatten=True), expected_direction)

    expected_rh = relative_humidity_from_dewpoint(t_values, td_values)
    np.testing.assert_allclose(by_param["RELHUM_2M"].to_numpy(flatten=True), expected_rh)


def test_surface_diagnostics_subset_of_variables(data_dir):
    fieldlist, u_values, v_values, t_values, td_values = _synthetic_components(data_dir)

    # Only SP_10M requested: DD_10M and RELHUM_2M are skipped, even though
    # temperature/dewpoint are still required matches.
    diagnostics = SurfaceDiagnostics(
        u_component="10u",
        v_component="10v",
        temperature="2t",
        dewpoint="2d",
        variables="SP_10M",
    )
    result = diagnostics.forward(fieldlist)

    by_param = {f.metadata("param"): f for f in result}
    assert set(by_param) == {"10u", "10v", "2t", "2d", "SP_10M"}

    expected_speed = np.sqrt(u_values**2 + v_values**2)
    np.testing.assert_allclose(by_param["SP_10M"].to_numpy(flatten=True), expected_speed)


def test_surface_diagnostics_requires_at_least_one_variable():
    with pytest.raises(ValueError, match="at least one"):
        SurfaceDiagnostics(
            u_component="10u",
            v_component="10v",
            temperature="2t",
            dewpoint="2d",
            variables=[],
        )


def test_surface_diagnostics_rejects_unknown_variable():
    with pytest.raises(ValueError, match="Unsupported variable"):
        SurfaceDiagnostics(
            u_component="10u",
            v_component="10v",
            temperature="2t",
            dewpoint="2d",
            variables=["TOT_PREC"],
        )


def test_keep(caplog):
    fieldlist = new_fieldlist_from_list([ArrayField(np.zeros(4), RawMetadata({"param": p})) for p in ["t", "q", "z"]])

    kept = Keep(param=["t", "z"]).forward(fieldlist)
    assert [f.metadata("param") for f in kept] == ["t", "z"]

    with caplog.at_level("WARNING"):
        Keep(param=["t", "missing"]).forward(fieldlist)
    assert "missing" in caplog.text


def test_pipe_or_fdb_xarray_returns_piped_value_when_present(data_dir):
    fn = str(data_dir / "iaf2025010100")
    fieldlist = ekd.from_source("file", fn).sel(shortName="T_2M")

    interpolator = ModelToPressureLevel(interpolate_levels=[500])
    da = interpolator._get_field(fieldlist, "T_2M").to_xarray()["T_2M"]
    assert da.shape == (1147980,)
