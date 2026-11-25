"""Unit tests for the GaussianSmoother filter, using small synthetic fields."""

import numpy as np
import pytest
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField
from scipy.ndimage import gaussian_filter

from anemoi_plugins_meteoswiss.transform.filters.smoothing import GaussianSmoother

torch = pytest.importorskip("torch")

NY, NX = 40, 50
SIGMA = 2.0
# Points further than this from the border are unaffected by boundary handling (scipy truncates at 4 sigma).
MARGIN = int(4 * SIGMA) + 1


def _field(values, param):
    values = np.asarray(values, dtype=np.float64)
    return new_field_from_numpy(values, template=ArrayField(values, RawMetadata({"param": param, "units": "K"})))


def _fieldlist(arrays, params):
    return new_fieldlist_from_list([_field(a, p) for a, p in zip(arrays, params)])


def _values(field):
    return field.to_numpy(flatten=True).reshape(NY, NX)


@pytest.fixture
def random_fields():
    rng = np.random.default_rng(0)
    return [rng.normal(280.0, 5.0, (NY, NX)) for _ in range(5)]


def _interior(a):
    return a[MARGIN:-MARGIN, MARGIN:-MARGIN]


# --- Parameter selection -------------------------------------------------------------------------


def test_matches_all_when_params_omitted():
    smoother = GaussianSmoother(sigma=SIGMA)
    assert smoother.matches("t")
    assert smoother.matches(None)


def test_matches_names_and_glob_patterns():
    smoother = GaussianSmoother(sigma=SIGMA, params=["2t", "z_*"])
    assert smoother.matches("2t")
    assert smoother.matches("z_500")
    assert not smoother.matches("z")
    assert not smoother.matches("t")
    assert not smoother.matches(None)


def test_only_selected_params_are_smoothed_and_order_is_kept(random_fields):
    params = ["2t", "w", "z_500", "q", "z_850"]
    smoother = GaussianSmoother(sigma=SIGMA, params=["2t", "z_*"], device="cpu")

    result = smoother.forward(_fieldlist(random_fields, params))

    assert [f.metadata("param") for f in result] == params
    for src, out, param in zip(random_fields, result, params):
        if smoother.matches(param):
            assert not np.allclose(_values(out), src)
        else:
            np.testing.assert_array_equal(_values(out), src)


def test_no_matching_params_returns_fields_unchanged(random_fields):
    smoother = GaussianSmoother(sigma=SIGMA, params=["nothing"], device="cpu")

    result = smoother.forward(_fieldlist(random_fields[:2], ["t", "q"]))

    assert len(result) == 2
    for src, out in zip(random_fields, result):
        np.testing.assert_array_equal(_values(out), src)


def test_empty_fieldlist():
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")
    assert len(smoother.forward(new_fieldlist_from_list([]))) == 0


# --- Smoothing behaviour -------------------------------------------------------------------------


def test_matches_scipy_gaussian_filter_in_interior(random_fields):
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    result = smoother.forward(_fieldlist(random_fields, ["t"] * len(random_fields)))

    for src, out in zip(random_fields, result):
        np.testing.assert_allclose(_interior(_values(out)), _interior(gaussian_filter(src, SIGMA)), rtol=1e-5)


def test_boundary_matches_scipy_reflect_mode(random_fields):
    # Pins the current boundary handling (scipy default: mode="reflect", truncate=4.0).
    # Relax this if the GPU implementation deliberately treats edges differently.
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = _values(smoother.forward(_fieldlist(random_fields[:1], ["t"]))[0])

    np.testing.assert_allclose(out, gaussian_filter(random_fields[0], SIGMA), rtol=1e-5)


def test_constant_field_is_unchanged():
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = _values(smoother.forward(_fieldlist([np.full((NY, NX), 273.15)], ["t"]))[0])

    np.testing.assert_allclose(out, 273.15, rtol=1e-6)


def test_impulse_spreads_symmetrically_and_conserves_mass():
    src = np.zeros((NY, NX))
    cy, cx = NY // 2, NX // 2
    src[cy, cx] = 100.0
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = _values(smoother.forward(_fieldlist([src], ["t"]))[0])

    assert out.argmax() == cy * NX + cx
    assert 0.0 < out.max() < 100.0
    assert out.sum() == pytest.approx(100.0, rel=1e-6)
    # Gaussian kernel is isotropic: equal response at equal distances.
    assert out[cy + 3, cx] == pytest.approx(out[cy - 3, cx])
    assert out[cy, cx + 3] == pytest.approx(out[cy + 3, cx])
    # ...and it decays with distance.
    assert out[cy, cx] > out[cy, cx + 1] > out[cy, cx + 3]


def test_output_stays_within_input_range(random_fields):
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = _values(smoother.forward(_fieldlist(random_fields[:1], ["t"]))[0])

    assert out.min() >= random_fields[0].min() - 1e-9
    assert out.max() <= random_fields[0].max() + 1e-9


def test_larger_sigma_smooths_more(random_fields):
    fl = _fieldlist(random_fields[:1], ["t"])

    stds = [
        _interior(_values(GaussianSmoother(sigma=s, device="cpu").forward(fl)[0])).std() for s in (0.5, 1.0, 2.0, 4.0)
    ]

    assert stds == sorted(stds, reverse=True)


def test_output_keeps_metadata(random_fields):
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = smoother.forward(_fieldlist(random_fields[:1], ["2t"]))[0]

    assert out.metadata("param") == "2t"
    assert out.metadata("units") == "K"
    assert out.to_numpy(flatten=True).shape == (NY * NX,)


# --- NaN handling --------------------------------------------------------------------------------


def test_nan_cells_do_not_contaminate_neighbours():
    src = np.full((NY, NX), 10.0)
    src[15:20, 20:25] = np.nan  # hole, e.g. out-of-domain points from regridding
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = _values(smoother.forward(_fieldlist([src], ["t"]))[0])

    valid = ~np.isnan(src)
    assert np.all(np.isfinite(out[valid]))
    # Normalising by the smoothed validity mask keeps a constant field constant right up to the hole.
    np.testing.assert_allclose(out[valid], 10.0, rtol=1e-6)


def test_nan_aware_smoothing_matches_normalised_reference():
    rng = np.random.default_rng(1)
    src = rng.normal(0.0, 1.0, (NY, NX))
    src[:, :5] = np.nan  # NaN strip along the western edge
    src[25, 30] = np.nan
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = _values(smoother.forward(_fieldlist([src], ["t"]))[0])

    nan = np.isnan(src)
    expected = gaussian_filter(np.where(nan, 0.0, src), SIGMA) / gaussian_filter((~nan).astype(float), SIGMA)
    np.testing.assert_allclose(out[~nan], expected[~nan], rtol=1e-5, atol=1e-9)


def test_all_nan_field_stays_nan():
    smoother = GaussianSmoother(sigma=SIGMA, device="cpu")

    out = _values(smoother.forward(_fieldlist([np.full((NY, NX), np.nan)], ["t"]))[0])

    assert np.all(np.isnan(out))


# --- Batching / device ---------------------------------------------------------------------------


@pytest.mark.parametrize("fields_per_chunk", [1, 2, 3, 16])
def test_result_is_independent_of_chunk_size(random_fields, fields_per_chunk):
    params = ["2t", "w", "2t", "2t", "w"]
    fl = _fieldlist(random_fields, params)
    reference = [_values(f) for f in GaussianSmoother(sigma=SIGMA, params=["2t"], device="cpu").forward(fl)]

    result = GaussianSmoother(sigma=SIGMA, params=["2t"], device="cpu", fields_per_chunk=fields_per_chunk).forward(fl)

    assert [f.metadata("param") for f in result] == params
    for ref, out in zip(reference, result):
        np.testing.assert_allclose(_values(out), ref, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
def test_cuda_matches_cpu(random_fields):
    fl = _fieldlist(random_fields, ["t"] * len(random_fields))
    cpu = GaussianSmoother(sigma=SIGMA, device="cpu").forward(fl)
    gpu_smoother = GaussianSmoother(sigma=SIGMA, device="cuda")
    assert gpu_smoother.device.type == "cuda"
    gpu = gpu_smoother.forward(fl)

    for a, b in zip(cpu, gpu):
        np.testing.assert_allclose(_values(a), _values(b), rtol=1e-5, atol=1e-6)
