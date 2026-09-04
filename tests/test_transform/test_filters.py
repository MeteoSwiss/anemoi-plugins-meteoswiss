import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from scipy.interpolate import RegularGridInterpolator

from anemoi_plugins_meteoswiss.transform import filters
from anemoi_plugins_meteoswiss.transform.filters import GaussianSmoother
from anemoi_plugins_meteoswiss.transform.filters import IconRemapToRegLatLon
from anemoi_plugins_meteoswiss.transform.filters.nudging import NudgeTowardObservation
from anemoi_plugins_meteoswiss.transform.filters.nudging import barrier_distances
from anemoi_plugins_meteoswiss.transform.filters.nudging import ned_interp

ICONREMAP_WEIGHTS = (
    "/store_new/mch/msopr/icon_workflow_2/iconremap-weights/icon-ch1-eps-rotlatlon.nc"
)


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
    fieldlist = ekd.from_source("file", fn).sel(shortName="2t")

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
    smoother = GaussianSmoother(sigma=5, params=["2t"])

    fn = str(data_dir / "iaf2025010100")
    fs = ekd.from_source("file", fn)
    # T_2M (smoothed) + one level of W (pass-through, not in params)
    fieldlist = new_fieldlist_from_list(
        list(fs.sel(shortName="2t")) + [fs.sel(shortName="wz")[0]]
    )

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


def _make_mock_dem_rgi():
    """Flat 100 m DEM on a 200 × 200 km LV95 grid."""
    x = np.linspace(2_500_000, 2_700_000, 50)
    y = np.linspace(1_100_000, 1_300_000, 50)
    z = np.full((len(y), len(x)), 100.0)
    return RegularGridInterpolator((y, x), z, method="linear", bounds_error=False, fill_value=0.0)


def _make_mock_transformer():
    """WGS84 → LV95 transformer (real pyproj)."""
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)


def test_ned_interp_no_topo():
    """ned_interp without topo falls back to pure IDW and produces finite outputs."""
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
        {"T_2M": xr.DataArray(
            rng.standard_normal(n_sta).astype(np.float32),
            dims=["sta"], coords={"sta": sta_ids},
        )}
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
        {"T_2M": xr.DataArray(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            dims=["sta"], coords={"sta": sta_ids},
        )}
    )
    result = ned_interp(residuals, dist, max_dist=0.5, weight_power=2.0)
    assert np.all(np.isnan(result["T_2M"].values))


def test_barrier_distances_flat_terrain():
    """On a flat DEM with same station/POI elevation, barrier=0 and d_eff ≈ d_euc."""
    dem_rgi = _make_mock_dem_rgi()
    transformer = _make_mock_transformer()

    poi_lon = np.array([8.0], dtype=np.float32)
    poi_lat = np.array([47.0], dtype=np.float32)
    sta_lon = np.array([8.0], dtype=np.float32)
    sta_lat = np.array([47.1], dtype=np.float32)
    sta_elev = np.array([100.0], dtype=np.float32)   # same as flat DEM → elev_diff = 0

    lat0 = np.deg2rad(47.05)
    d_raw = float(np.sqrt((47.0 - 47.1) ** 2))       # delta-lon=0, delta-lat=0.1
    d_euc = np.array([[d_raw]], dtype=np.float32)

    d_eff = barrier_distances(
        poi_lon, poi_lat, sta_lon, sta_lat,
        d_euc, max_dist=0.5,
        sta_elev=sta_elev,
        dem_rgi=dem_rgi, wgs84_to_lv95=transformer,
        n_samples=10, elev_scale=2000.0, elev_diff_scale=4000.0,
        n_barrier_width_samples=1, barrier_width=0.0,
    )
    np.testing.assert_allclose(d_eff, d_euc, rtol=1e-4)


def test_barrier_distances_same_valley_no_penalty():
    """Two points at the same altitude in a flat valley get no barrier penalty."""
    dem_rgi = _make_mock_dem_rgi()
    transformer = _make_mock_transformer()

    poi_lon = np.array([8.0])
    poi_lat = np.array([47.0])
    sta_lon = np.array([8.2])
    sta_lat = np.array([47.0])
    sta_elev = np.array([100.0])   # matches flat DEM → elev_diff = 0, barrier = 0

    lat0 = np.deg2rad(47.0)
    d_raw = float(np.sqrt((0.2 * np.cos(lat0)) ** 2))
    d_euc = np.array([[d_raw]], dtype=np.float32)

    d_eff = barrier_distances(
        poi_lon, poi_lat, sta_lon, sta_lat,
        d_euc, max_dist=0.5,
        sta_elev=sta_elev,
        dem_rgi=dem_rgi, wgs84_to_lv95=transformer,
        n_samples=5, elev_scale=2000.0, elev_diff_scale=4000.0,
        n_barrier_width_samples=1, barrier_width=0.0,
    )
    np.testing.assert_allclose(d_eff, d_euc, rtol=1e-4)


def test_barrier_distances_no_close_pairs():
    """When no pairs are within max_dist, d_euc is returned unchanged."""
    dem_rgi = _make_mock_dem_rgi()
    transformer = _make_mock_transformer()

    d_euc = np.array([[1.0, 2.0]], dtype=np.float32)
    d_eff = barrier_distances(
        np.array([8.0]), np.array([47.0]),
        np.array([8.0, 8.5]), np.array([48.0, 48.0]),
        d_euc, max_dist=0.3,
        sta_elev=np.array([100.0, 200.0]),
        dem_rgi=dem_rgi, wgs84_to_lv95=transformer,
    )
    np.testing.assert_array_equal(d_eff, d_euc)


def test_nudge_toward_observation_invalid_run_mode(tmp_path):
    """Invalid run_mode raises ValueError at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with patch.object(NudgeTowardObservation, "_load_icon_grid"), \
         patch.object(NudgeTowardObservation, "_load_topo"), \
         patch.object(NudgeTowardObservation, "_load_dem"):
        with pytest.raises(ValueError, match="run_mode"):
            NudgeTowardObservation(obs_path=str(obs), run_mode="bad")


def test_nudge_toward_observation_mutual_exclusion(tmp_path):
    """holdout_fraction and exclude_stations together raise ValueError."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with patch.object(NudgeTowardObservation, "_load_icon_grid"), \
         patch.object(NudgeTowardObservation, "_load_topo"), \
         patch.object(NudgeTowardObservation, "_load_dem"):
        with pytest.raises(ValueError, match="mutually exclusive"):
            NudgeTowardObservation(
                obs_path=str(obs),
                holdout_fraction=0.1,
                exclude_stations=["ABC"],
            )


def test_nudge_toward_observation_invalid_reliability_min_dist_frac(tmp_path):
    """reliability_min_dist_frac outside [0, 1] raises ValueError at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with patch.object(NudgeTowardObservation, "_load_icon_grid"), \
         patch.object(NudgeTowardObservation, "_load_topo"), \
         patch.object(NudgeTowardObservation, "_load_dem"):
        with pytest.raises(ValueError, match="reliability_min_dist_frac"):
            NudgeTowardObservation(obs_path=str(obs), reliability_min_dist_frac=1.5)


def test_nudge_toward_observation_invalid_number_of_std(tmp_path):
    """number_of_std <= 0 raises ValueError at construction."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with patch.object(NudgeTowardObservation, "_load_icon_grid"), \
         patch.object(NudgeTowardObservation, "_load_topo"), \
         patch.object(NudgeTowardObservation, "_load_dem"):
        with pytest.raises(ValueError, match="number_of_std"):
            NudgeTowardObservation(obs_path=str(obs), number_of_std=0.0)


def test_compute_reliability_flags_outlier_station(tmp_path):
    """Leave-one-out spatial-consistency check: a station whose residual wildly
    disagrees with its neighbours gets reliability=0; consistent stations stay
    close to 1. Uses the same mock DEM/transformer as the barrier_distances tests
    above — no real ICON/DEM/topo files needed."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with patch.object(NudgeTowardObservation, "_load_icon_grid"), \
         patch.object(NudgeTowardObservation, "_load_topo"), \
         patch.object(NudgeTowardObservation, "_load_dem"):
        filt = NudgeTowardObservation(
            obs_path=str(obs),
            max_dist=50_000.0,  # meters (50 km)
            weight_power=2.0,
            min_topo_w=0.2,
            lim_effective=0.0,
            number_of_std=4.0,
            reliability_min_dist_frac=0.1,
        )
    filt._dem_rgi = _make_mock_dem_rgi()
    filt._wgs84_to_lv95 = _make_mock_transformer()

    sta_ids = ["AAA", "BBB", "CCC", "DDD", "BAD"]
    st_lat = np.array([47.00, 47.02, 46.98, 47.01, 46.99])
    st_lon = np.array([8.00, 8.02, 7.98, 8.05, 8.01])
    st_elev = np.full(5, 100.0)  # matches the flat mock DEM → barrier/elev_diff = 0
    sta_x, sta_y = filt._wgs84_to_lv95.transform(st_lon, st_lat)
    sta_xy = np.c_[sta_x, sta_y] / 1000.0
    r_at_st = np.array([0.0, -0.2, 0.2, -0.1, 5.0])  # "BAD" wildly disagrees

    # A single topo descriptor is enough to exercise ned_interp's topo-similarity
    # branch (its importance normalises to 1 trivially with only one descriptor).
    sta_topo = xr.Dataset(
        {"ELEV": xr.DataArray(st_elev.astype(np.float32), dims=["sta"], coords={"sta": sta_ids})}
    )

    reliability = filt._compute_reliability(
        "T_2M", sta_ids, st_lon, st_lat, st_elev, sta_xy, r_at_st, sta_topo,
    )

    assert reliability.dims == ("sta",)
    assert list(reliability["sta"].values) == sta_ids
    values = reliability.values
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert values[-1] == 0.0, f"BAD station should be fully rejected, got {values[-1]}"
    assert np.all(values[:-1] > values[-1]), "consistent stations must outrank BAD"


def test_compute_reliability_isolated_station_does_not_poison_others(tmp_path):
    """Regression test: a station with no neighbour within max_dist gets e=NaN
    from ned_interp's leave-one-out call (min_count=1 makes a fully-masked POI
    return NaN, not 0). Before the fix, a single NaN silently propagated through
    np.median/np.abs(...).median() into EVERY station's reliability (all NaN),
    which then made ned_interp mask every pair (`dist < NaN` is always False),
    zeroing the correction for the whole field — exactly what was observed in
    production (see dashboard investigation, 2026-08-19). The isolated station
    itself should land at reliability=1.0 (no evidence to judge it against);
    every other, mutually-consistent station should stay finite and high."""
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with patch.object(NudgeTowardObservation, "_load_icon_grid"), \
         patch.object(NudgeTowardObservation, "_load_topo"), \
         patch.object(NudgeTowardObservation, "_load_dem"):
        filt = NudgeTowardObservation(
            obs_path=str(obs),
            max_dist=50_000.0,  # meters (50 km)
            weight_power=2.0,
            min_topo_w=0.2,
            lim_effective=0.0,
            number_of_std=4.0,
            reliability_min_dist_frac=0.1,
        )
    filt._dem_rgi = _make_mock_dem_rgi()
    filt._wgs84_to_lv95 = _make_mock_transformer()

    # AAA/BBB/CCC/DDD form a consistent cluster; ISO is ~190 km away — well
    # beyond max_dist=50 km, so it has zero neighbours in the leave-one-out check.
    sta_ids = ["AAA", "BBB", "CCC", "DDD", "ISO"]
    st_lat = np.array([47.00, 47.02, 46.98, 47.01, 46.20])
    st_lon = np.array([8.00, 8.02, 7.98, 8.05, 9.90])
    st_elev = np.full(5, 100.0)
    sta_x, sta_y = filt._wgs84_to_lv95.transform(st_lon, st_lat)
    sta_xy = np.c_[sta_x, sta_y] / 1000.0
    r_at_st = np.array([0.0, -0.2, 0.2, -0.1, 3.0])  # all mutually plausible values

    sta_topo = xr.Dataset(
        {"ELEV": xr.DataArray(st_elev.astype(np.float32), dims=["sta"], coords={"sta": sta_ids})}
    )

    reliability = filt._compute_reliability(
        "T_2M", sta_ids, st_lon, st_lat, st_elev, sta_xy, r_at_st, sta_topo,
    )
    values = reliability.values

    assert not np.any(np.isnan(values)), f"NaN leaked into reliability: {values}"
    assert values[-1] == 1.0, f"isolated station should default to reliability=1.0, got {values[-1]}"
    assert np.all(values[:-1] > 0.5), f"consistent cluster stations should stay trusted, got {values[:-1]}"
