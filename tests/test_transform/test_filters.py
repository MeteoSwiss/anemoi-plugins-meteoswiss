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
        n_barrier_width_samples=1, barrier_width_m=0.0,
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
        n_barrier_width_samples=1, barrier_width_m=0.0,
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


def test_nudge_toward_observation_deprecated_params(tmp_path):
    """Deprecated 'k' and 'power' parameters emit DeprecationWarning."""
    import warnings
    from unittest.mock import patch

    obs = tmp_path / "obs.parquet"
    obs.touch()

    with patch.object(NudgeTowardObservation, "_load_icon_grid"), \
         patch.object(NudgeTowardObservation, "_load_topo"), \
         patch.object(NudgeTowardObservation, "_load_dem"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NudgeTowardObservation(obs_path=str(obs), power=2.0, k=5)

    messages = [str(warning.message) for warning in w]
    assert any("power" in m for m in messages), "Expected DeprecationWarning for 'power'"
    assert any("k" in m for m in messages), "Expected DeprecationWarning for 'k'"


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
