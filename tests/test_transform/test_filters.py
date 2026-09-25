import earthkit.data as ekd
import numpy as np
import pytest
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

ICONREMAP_WEIGHTS = "/store_new/mch/msopr/icon_workflow_2/iconremap-weights/icon-ch1-eps-rotlatlon.nc"


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
