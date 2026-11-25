"""Unit tests for the iconremap regridding filter, using small synthetic weight files."""

import numpy as np
import pytest
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField

from anemoi_plugins_meteoswiss.transform.filters.iconremap import IconRemapToRegLatLon
from anemoi_plugins_meteoswiss.transform.filters.iconremap import _rot_to_geo

torch = pytest.importorskip("torch")

# ICON-CH rotated pole.
POLLON, POLLAT = 190.0, 43.0

# Synthetic setup: 8 source cells regridded onto a 2 x 3 rotated grid with 3-point stencils.
NY, NX = 2, 3
N_SOURCE = 8
INDICES = np.array(
    [
        [1, 2, 3],
        [2, 3, 4],
        [0, 4, 5],  # contains the out-of-domain sentinel -> NaN
        [5, 6, 7],
        [3, 6, 0],  # sentinel in last position -> NaN
        [1, 7, 4],
    ],
    dtype=np.int32,
)
WEIGHTS = np.array(
    [
        [0.2, 0.3, 0.5],
        [1.5, -0.25, -0.25],  # RBF-style negative weights that overshoot -> clipped
        [0.3, 0.3, 0.4],
        [-0.5, 0.75, 0.75],  # overshoot on the other side -> clipped
        [0.5, 0.5, 0.0],
        [1 / 3, 1 / 3, 1 / 3],
    ]
)
VALID = np.array([True, True, False, True, False, True])


def _write_weights(path, indices=INDICES, weights=WEIGHTS, ny=NY, nx=NX, pollon=POLLON, pollat=POLLAT):
    ds = xr.Dataset(
        {
            "rbf_B_glbidx": (("n_out", "n_stencil"), indices),
            "rbf_B_wgt": (("n_out", "n_stencil"), weights),
        },
        attrs={
            "nx": nx,
            "ny": ny,
            "xmin": -1.0,
            "dx": 1.0,
            "ymin": -0.5,
            "dy": 1.0,
            "north_pole_lon": pollon,
            "north_pole_lat": pollat,
        },
    )
    ds.to_netcdf(path)
    return str(path)


def _reference_remap(values):
    """Plain-numpy implementation of the iconremap application (the pre-torch implementation)."""
    stencil = values[INDICES]
    out = np.clip(np.sum(stencil * WEIGHTS, axis=-1), stencil.min(axis=-1), stencil.max(axis=-1))
    out[~VALID] = np.nan
    return out


def _fieldlist(arrays):
    fields = []
    for i, values in enumerate(arrays):
        base = ArrayField(values, RawMetadata({"param": f"p{i}", "levelist": i}))
        fields.append(new_field_from_numpy(values, template=base))
    return new_fieldlist_from_list(fields)


@pytest.fixture
def weights_file(tmp_path):
    return _write_weights(tmp_path / "weights.nc")


@pytest.fixture
def source_values():
    rng = np.random.default_rng(42)
    return [rng.uniform(250.0, 300.0, N_SOURCE) for _ in range(5)]


# --- _rot_to_geo ---------------------------------------------------------------------------------


def test_rot_to_geo_rotated_origin_maps_to_rotation_centre():
    lat, lon = _rot_to_geo(np.array([0.0]), np.array([0.0]), POLLON, POLLAT)
    np.testing.assert_allclose(lat, [90.0 - POLLAT])
    np.testing.assert_allclose(lon, [POLLON - 180.0])


def test_rot_to_geo_matches_pyproj_reference():
    # Reference values from pyproj (CF rotated_latitude_longitude -> EPSG:4326).
    rlon = np.array([0.0, 1.0, -1.0, 0.0, 5.0])
    rlat = np.array([0.0, 0.0, 0.0, 1.0, -3.0])
    expected_lon = np.array([10.0, 11.466108018379666, 8.533891981620311, 10.0, 16.923653358864104])
    expected_lat = np.array([47.0, 46.990642874481914, 46.990642874481914, 48.0, 43.77904558268683])

    lat, lon = _rot_to_geo(rlon, rlat, POLLON, POLLAT)

    np.testing.assert_allclose(lat, expected_lat, atol=1e-9)
    np.testing.assert_allclose(lon, expected_lon, atol=1e-9)


def test_rot_to_geo_unrotated_pole_is_identity():
    rlon = np.array([-170.0, -45.0, 0.0, 30.0, 120.0])
    rlat = np.array([-60.0, -10.0, 0.0, 45.0, 80.0])

    lat, lon = _rot_to_geo(rlon, rlat, 180.0, 90.0)

    np.testing.assert_allclose(lat, rlat, atol=1e-9)
    np.testing.assert_allclose(lon, rlon, atol=1e-9)


def test_rot_to_geo_output_ranges():
    rlon, rlat = np.meshgrid(np.linspace(-180, 180, 37), np.linspace(-90, 90, 19))
    lat, lon = _rot_to_geo(rlon.ravel(), rlat.ravel(), POLLON, POLLAT)
    assert np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))
    assert lat.min() >= -90.0 and lat.max() <= 90.0
    assert lon.min() >= -180.0 and lon.max() < 180.0


# --- IconRemapToRegLatLon ------------------------------------------------------------------------


def test_init_reads_weights_file(weights_file):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu")

    assert (regridder.ny, regridder.nx) == (NY, NX)
    np.testing.assert_array_equal(regridder.indices, INDICES)
    np.testing.assert_array_equal(regridder.weights, WEIGHTS)
    np.testing.assert_array_equal(regridder.valid_mask, VALID)
    assert regridder.device == torch.device("cpu")

    # Output coordinates are the row-major (ny, nx) rotated grid converted to geographic.
    rlon, rlat = np.meshgrid(-1.0 + np.arange(NX), -0.5 + np.arange(NY))
    exp_lat, exp_lon = _rot_to_geo(rlon.ravel(), rlat.ravel(), POLLON, POLLAT)
    np.testing.assert_allclose(regridder._latitudes, exp_lat)
    np.testing.assert_allclose(regridder._longitudes, exp_lon)


def test_forward_matches_numpy_reference(weights_file, source_values):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu")

    result = regridder.forward(_fieldlist(source_values))

    assert len(result) == len(source_values)
    for src, out in zip(source_values, result):
        np.testing.assert_allclose(out.to_numpy(flatten=True), _reference_remap(src), rtol=1e-6, equal_nan=True)


def test_forward_sets_nan_only_where_stencil_has_sentinel(weights_file, source_values):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu")

    values = regridder.forward(_fieldlist(source_values))[0].to_numpy(flatten=True)

    np.testing.assert_array_equal(np.isnan(values), ~VALID)


def test_forward_clips_to_stencil_range(weights_file):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu")
    src = np.arange(N_SOURCE, dtype=np.float64) * 10.0  # stencil of row 1 = [20, 30, 40]; row 3 = [50, 60, 70]

    values = regridder.forward(_fieldlist([src]))[0].to_numpy(flatten=True)

    # Unclipped: row 1 -> 1.5*20 - 0.25*30 - 0.25*40 = 12.5 (< 20); row 3 -> -25 + 45 + 52.5 = 72.5 (> 70).
    assert values[1] == pytest.approx(20.0)
    assert values[3] == pytest.approx(70.0)
    # Non-overshooting rows are untouched by the clip.
    assert values[0] == pytest.approx(0.2 * 10 + 0.3 * 20 + 0.5 * 30)
    assert values[5] == pytest.approx((10 + 70 + 40) / 3)


def test_forward_constant_field_is_preserved(weights_file):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu")

    values = regridder.forward(_fieldlist([np.full(N_SOURCE, 273.15)]))[0].to_numpy(flatten=True)

    np.testing.assert_allclose(values[VALID], 273.15)


def test_forward_output_fields(weights_file, source_values):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu")

    result = regridder.forward(_fieldlist(source_values))

    for i, field in enumerate(result):
        assert field.to_numpy().shape == (NY, NX)
        assert field.metadata("param") == f"p{i}"
        assert field.metadata("levelist") == i
        latlon = field.to_latlon(flatten=True)
        np.testing.assert_allclose(latlon["lat"], regridder._latitudes)
        np.testing.assert_allclose(latlon["lon"], regridder._longitudes)


@pytest.mark.parametrize("fields_per_chunk", [1, 2, 3, 5, 16])
def test_forward_result_is_independent_of_chunk_size(weights_file, source_values, fields_per_chunk):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu", fields_per_chunk=fields_per_chunk)

    result = regridder.forward(_fieldlist(source_values))

    assert len(result) == len(source_values)
    for src, out in zip(source_values, result):
        np.testing.assert_allclose(out.to_numpy(flatten=True), _reference_remap(src), rtol=1e-6, equal_nan=True)


def test_forward_empty_fieldlist(weights_file):
    regridder = IconRemapToRegLatLon(weights_file, device="cpu")
    assert len(regridder.forward(new_fieldlist_from_list([]))) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
def test_forward_cuda_matches_cpu(weights_file, source_values):
    cpu = IconRemapToRegLatLon(weights_file, device="cpu").forward(_fieldlist(source_values))
    gpu_regridder = IconRemapToRegLatLon(weights_file, device="cuda")
    assert gpu_regridder.device.type == "cuda"
    gpu = gpu_regridder.forward(_fieldlist(source_values))

    for a, b in zip(cpu, gpu):
        np.testing.assert_allclose(a.to_numpy(flatten=True), b.to_numpy(flatten=True), rtol=1e-6, equal_nan=True)
