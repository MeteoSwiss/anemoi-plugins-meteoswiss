import functools
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField
from earthkit.meteo.thermo import relative_humidity_from_dewpoint
from earthkit.meteo.thermo import wet_bulb_temperature_from_dewpoint
from earthkit.meteo.thermo import wet_bulb_temperature_from_specific_humidity

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

    # Only SP_10M requested: DD_10M and RELHUM_2M are skipped, and
    # temperature/dewpoint are not matched but passed through.
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


def test_surface_diagnostics_requires_inputs_of_requested_variables():
    with pytest.raises(ValueError, match="surface_pressure"):
        SurfaceDiagnostics(temperature="2t", dewpoint="2d", variables=["TW_2M"])


def test_surface_diagnostics_vertical_requires_orography():
    with pytest.raises(ValueError, match="orography_file"):
        SurfaceDiagnostics(
            temperature="2t",
            surface_pressure="sp",
            temperature_levels="t",
            geopotential_levels="z",
            variables=["HZEROCL"],
        )


# Four columns on 1000/850/700/500/300 hPa:
# 0: lowland, single crossing between 850 and 700 hPa
# 1: valley cold pool (T_2M < 0 degC) under a warm layer: uppermost crossing wins
# 2: whole column below 0 degC: no crossing
# 3: 1000 hPa below ground and warm; the crossing is between surface and 850 hPa
_LEVELS = [1000, 850, 700, 500, 300]
_LEVEL_HEIGHTS = [100.0, 1500.0, 3000.0, 5600.0, 9200.0]
_OROGRAPHY = np.array([0.0, 500.0, 1500.0, 300.0])
_SP = np.array([101000.0, 95000.0, 85000.0, 97000.0])
_T_2M = np.array([285.0, 270.0, 265.0, 275.0])
_TD_2M = _T_2M - np.array([2.0, 1.0, 3.0, 4.0])
_T_LEVELS = np.array(
    [
        [283.0, 272.0, 262.0, 290.0],
        [275.0, 276.0, 260.0, 270.0],
        [268.0, 271.0, 258.0, 262.0],
        [250.0, 255.0, 240.0, 250.0],
        [225.0, 228.0, 220.0, 226.0],
    ]
)
_Q_LEVELS = np.array(
    [
        [8e-3, 3e-3, 2e-3, 9e-3],
        [5e-3, 5e-3, 1.5e-3, 3e-3],
        [3e-3, 3e-3, 1e-3, 2e-3],
        [1e-3, 1e-3, 5e-4, 1e-3],
        [1e-4, 1e-4, 1e-4, 1e-4],
    ]
)


@functools.cache
def _template():
    return ekd.from_source("file", str(Path(__file__).parent.parent / "data" / "iaf2025010100")).sel(shortName="T_2M")[
        0
    ]


def _tiled(values):
    """Repeat the four test columns over the whole template grid."""
    return np.resize(np.asarray(values, dtype=float), _template().to_numpy(flatten=True).size)


def _array_field(values, **metadata):
    return new_field_from_numpy(_tiled(values), template=_template(), shortName=metadata["param"], **metadata)


def _vertical_fieldlist(**metadata):
    """The test columns as a fieldlist; ``metadata`` (e.g. ``step``) is set on every field."""
    # Geopotential that converts back to exactly _LEVEL_HEIGHTS.
    from earthkit.meteo import constants

    heights = np.array(_LEVEL_HEIGHTS)
    z = constants.g * heights * constants.R_earth / (constants.R_earth + heights)
    fields = [
        _array_field(_T_2M, param="2t", **metadata),
        _array_field(_TD_2M, param="2d", **metadata),
        _array_field(_SP, param="sp", **metadata),
    ]
    for k, level in enumerate(_LEVELS):
        fields.append(_array_field(_T_LEVELS[k], param="t", levelist=level, **metadata))
        fields.append(_array_field(_Q_LEVELS[k], param="q", levelist=level, **metadata))
        fields.append(_array_field(np.full(4, z[k]), param="z", levelist=level, **metadata))
    return new_fieldlist_from_list(fields)


@pytest.fixture
def orography_file(tmp_path):
    import xarray as xr

    path = tmp_path / "extpar.nc"
    xr.Dataset({"topography_c": ("cell", _tiled(_OROGRAPHY))}).to_netcdf(path)
    return path


def _reference_uppermost_crossing(surface_values, level_values, threshold):
    """Per-column loop: scan top-down for the first pair with value >= threshold
    below and < threshold above; NaN if none."""
    out = np.full(len(_OROGRAPHY), np.nan)
    for i in range(len(_OROGRAPHY)):
        points = [(_OROGRAPHY[i] + 2.0, surface_values[i])]
        for k, level in enumerate(_LEVELS):
            if level * 100.0 < _SP[i]:
                points.append((_LEVEL_HEIGHTS[k], level_values[k][i]))
        for (h0, v0), (h1, v1) in reversed(list(zip(points[:-1], points[1:]))):
            if v0 >= threshold > v1:
                out[i] = h0 + (threshold - v0) / (v1 - v0) * (h1 - h0)
                break
    return out


def _vertical_diagnostics(orography_file, variables, **kwargs):
    return SurfaceDiagnostics(
        temperature="2t",
        dewpoint="2d",
        surface_pressure="sp",
        temperature_levels="t",
        specific_humidity_levels="q",
        geopotential_levels="z",
        orography_file=str(orography_file),
        variables=variables,
        **kwargs,
    )


def test_surface_diagnostics_tw_2m():
    fieldlist = _vertical_fieldlist()
    result = SurfaceDiagnostics(temperature="2t", dewpoint="2d", surface_pressure="sp", variables="TW_2M").forward(
        fieldlist
    )

    by_name = {f.metadata("param"): f for f in result if f.metadata("param") == "TW_2M"}
    expected = wet_bulb_temperature_from_dewpoint(_T_2M, _TD_2M, _SP)
    np.testing.assert_allclose(by_name["TW_2M"].to_numpy(flatten=True)[:4], expected)
    assert np.all(expected <= _T_2M) and np.all(expected >= _TD_2M)
    # All inputs, including the untouched pressure levels, are still there.
    assert len(result) == len(fieldlist) + 1


def test_surface_diagnostics_vertical_keeps_pressure_levels(orography_file):
    fieldlist = _vertical_fieldlist()
    result = _vertical_diagnostics(orography_file, ["TW_2M", "HZEROCL", "SNOWLMT", "MELTLAYER_TOP"]).forward(fieldlist)

    assert len(result) == len(fieldlist) + 4
    params = [f.metadata("param") for f in result]
    assert {"TW_2M", "HZEROCL", "SNOWLMT", "MELTLAYER_TOP"} <= set(params)
    for param in ("t", "q", "z"):
        levels = sorted(f.metadata("levelist") for f in result if f.metadata("param") == param)
        assert levels == sorted(_LEVELS)
    t_850 = next(f for f in result if f.metadata("param") == "t" and f.metadata("levelist") == 850)
    np.testing.assert_array_equal(t_850.to_numpy(flatten=True), _tiled(_T_LEVELS[1]))


def test_surface_diagnostics_rejects_orography_size_mismatch(tmp_path):
    import xarray as xr

    path = tmp_path / "extpar.nc"
    xr.Dataset({"topography_c": ("cell", np.zeros(7))}).to_netcdf(path)
    with pytest.raises(ValueError, match="native grid"):
        _vertical_diagnostics(path, ["HZEROCL"]).forward(_vertical_fieldlist())


def test_surface_diagnostics_height_levels(orography_file):
    result = _vertical_diagnostics(orography_file, ["HZEROCL", "SNOWLMT", "MELTLAYER_TOP"]).forward(
        _vertical_fieldlist()
    )

    by_name = {f.metadata("param"): f.to_numpy(flatten=True)[:4] for f in result}
    z0, zs, zm = by_name["HZEROCL"], by_name["SNOWLMT"], by_name["MELTLAYER_TOP"]
    pressure = np.repeat(np.array(_LEVELS, dtype=float) * 100.0, 4)
    tw_levels = wet_bulb_temperature_from_specific_humidity(_T_LEVELS.ravel(), _Q_LEVELS.ravel(), pressure).reshape(
        _T_LEVELS.shape
    )
    tw_2m = wet_bulb_temperature_from_dewpoint(_T_2M, _TD_2M, _SP)

    # Crossings: the uppermost one, with no search ceiling.
    expected_z0 = _reference_uppermost_crossing(_T_2M, _T_LEVELS, 273.15)
    np.testing.assert_allclose(z0[[0, 1, 3]], expected_z0[[0, 1, 3]])
    np.testing.assert_allclose(zs[[0, 1]], _reference_uppermost_crossing(tw_2m, tw_levels, 273.15 + 1.3)[[0, 1]])
    expected_zm = _reference_uppermost_crossing(tw_2m, tw_levels, 273.15 + 0.2)
    np.testing.assert_allclose(zm[[0, 1, 3]], expected_zm[[0, 1, 3]])

    # Column 2 (all below 0 degC): Z0 extrapolated below ground with 0.65 K/100 m.
    np.testing.assert_allclose(z0[2], _OROGRAPHY[2] - (273.15 - _T_2M[2]) / 0.0065)
    assert z0[2] < _OROGRAPHY[2]
    # ...ZS extrapolated with 0.5 K/100 m and clamped at -100 m, ZM pushed 75 m above it.
    assert zs[2] == -100.0
    np.testing.assert_allclose(zm[2], -25.0)
    # Column 3: ZM has a crossing but ZS doesn't, so ZS is extrapolated below ground.
    np.testing.assert_allclose(zs[3], _OROGRAPHY[3] - (273.15 + 1.3 - tw_2m[3]) / 0.005)
    assert zs[3] < _OROGRAPHY[3]
    assert np.all(zm - zs >= 75.0)


def test_cape_cin_integration():
    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _cape_cin

    # Four columns, bottom-up, 1000 m apart; buoyancy is linear between points.
    height = np.repeat(np.arange(0.0, 5000.0, 1000.0)[:, None], 4, axis=1)
    pressure = np.repeat(np.array([100000.0, 90000.0, 80000.0, 70000.0, 60000.0])[:, None], 4, axis=1)
    p_lcl = np.array([90000.0, 80000.0, 80000.0, 90000.0])
    buoyancy = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.01, 0.01, -0.01, -0.01],
            [0.01, -0.01, 0.01, -0.02],
            [0.01, 0.01, 0.01, -0.01],
            [-0.01, 0.01, 0.01, -0.03],
        ]
    )
    cape, cin = _cape_cin(buoyancy, height, pressure, p_lcl, pressure[0])

    # 0: LFC and EL mid-segment: CAPE = 2.5 + 10 + 2.5, CIN = -5 - 2.5.
    # 1: positive buoyancy below the LCL (superadiabatic) is ignored; LFC above
    #    the LCL, no EL: CAPE = 2.5 + 10, CIN = -2.5 - 2.5.
    # 2: buoyant at the LCL itself: LFC at the LCL, CAPE = 10 + 10, CIN = -5 - 2.5.
    # 3: never buoyant: no LFC, CAPE 0 and CIN missing.
    np.testing.assert_allclose(cape, [15.0, 12.5, 20.0, 0.0])
    np.testing.assert_allclose(cin, [-7.5, -5.0, -7.5, np.nan])


def test_cape_cin_ignores_layers_below_parcel_start():
    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _cape_cin

    height = np.arange(0.0, 5000.0, 1000.0)[:, None]
    pressure = np.array([100000.0, 90000.0, 80000.0, 70000.0, 60000.0])[:, None]
    buoyancy = np.array([[0.0], [-0.02], [0.0], [0.01], [0.01]])
    # Parcel starting at 900 hPa with its LCL there: the negative layer below
    # 900 hPa doesn't count as CIN.
    cape, cin = _cape_cin(buoyancy, height, pressure, np.array([90000.0]), np.array([90000.0]))
    np.testing.assert_allclose(cin, [-10.0])
    np.testing.assert_allclose(cape, [15.0])


def test_mixed_layer_parcel():
    from earthkit.meteo.thermo import potential_temperature
    from earthkit.meteo.thermo import temperature_from_potential_temperature

    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _mixed_layer_parcel

    # Surface at 1000 hPa, levels at 975 and 900 hPa: the 50 hPa layer spans
    # the surface, 975 hPa and the 950 hPa point interpolated in log p.
    p = np.array([[100000.0], [97500.0], [90000.0]])
    t = np.array([[300.0], [298.0], [294.0]])
    q = np.array([[0.012], [0.010], [0.006]])
    p0, t0, q0 = _mixed_layer_parcel(p, np.zeros_like(p), t, q)

    w = np.log(97500.0 / 95000.0) / np.log(97500.0 / 90000.0)
    theta = potential_temperature(t[:, 0], p[:, 0])
    theta_top, q_top = theta[1] + w * (theta[2] - theta[1]), q[1, 0] + w * (q[2, 0] - q[1, 0])
    theta_ml = (0.5 * (theta[0] + theta[1]) * 2500.0 + 0.5 * (theta[1] + theta_top) * 2500.0) / 5000.0
    q_ml = (0.5 * (q[0, 0] + q[1, 0]) * 2500.0 + 0.5 * (q[1, 0] + q_top) * 2500.0) / 5000.0
    assert p0[0] == 100000.0
    np.testing.assert_allclose(t0, [temperature_from_potential_temperature(theta_ml, 100000.0)])
    np.testing.assert_allclose(q0, [q_ml])


def test_most_unstable_parcel():
    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _most_unstable_parcel

    # Column 0: moist surface. Column 1: cold, dry surface under warm, moist
    # air at 850 hPa. Column 2: the most energetic air (500 hPa) is above
    # the 300 hPa search depth, so the surface wins.
    p = np.repeat(np.array([[97000.0], [92500.0], [85000.0], [50000.0]]), 3, axis=1)
    t = np.array([[300.0, 270.0, 290.0], [296.0, 275.0, 286.0], [290.0, 285.0, 282.0], [260.0, 255.0, 330.0]])
    q = np.array([[0.015, 0.002, 0.010], [0.012, 0.004, 0.008], [0.008, 0.012, 0.006], [0.001, 0.001, 0.010]])
    p0, t0, q0 = _most_unstable_parcel(p, np.zeros_like(p), t, q)
    np.testing.assert_array_equal(p0, [97000.0, 85000.0, 97000.0])
    np.testing.assert_array_equal(t0, [300.0, 285.0, 290.0])
    np.testing.assert_array_equal(q0, [0.015, 0.012, 0.010])


@pytest.mark.parametrize("kind", ["ML", "MU"])
def test_surface_diagnostics_cape_cin(orography_file, kind):
    fieldlist = _vertical_fieldlist()
    result = _vertical_diagnostics(orography_file, [f"CAPE_{kind}", f"CIN_{kind}"]).forward(fieldlist)

    # Inputs, including the pressure levels, are kept.
    assert len(result) == len(fieldlist) + 2
    cape = next(f for f in result if f.metadata("param") == f"CAPE_{kind}").to_numpy(flatten=True)[:4]
    cin = next(f for f in result if f.metadata("param") == f"CIN_{kind}").to_numpy(flatten=True)[:4]
    assert np.all(cape >= 0.0)
    assert np.all(np.isnan(cin) | (cin <= 0.0))
    # Without an LFC there is no CAPE and CIN is missing; e.g. the column below 0 degC.
    assert cape[2] == 0.0 and np.isnan(cin[2])
    np.testing.assert_array_equal(np.isnan(cin), cape == 0.0)


def test_surface_diagnostics_cape_cin_ml_and_mu(orography_file):
    result = _vertical_diagnostics(orography_file, ["CAPE_ML", "CIN_ML", "CAPE_MU", "CIN_MU"]).forward(
        _vertical_fieldlist()
    )
    by_name = {f.metadata("param"): f.to_numpy(flatten=True)[:4] for f in result}
    # The moist lowland column is conditionally unstable for both parcels,
    # which start from different air and so give different values.
    assert by_name["CAPE_ML"][0] > 0.0 and by_name["CAPE_MU"][0] > 0.0
    assert by_name["CAPE_ML"][0] != by_name["CAPE_MU"][0]
    # Each pair matches what is computed when it is requested alone.
    alone = _vertical_diagnostics(orography_file, ["CAPE_MU", "CIN_MU"]).forward(_vertical_fieldlist())
    cape_mu = next(f for f in alone if f.metadata("param") == "CAPE_MU").to_numpy(flatten=True)[:4]
    np.testing.assert_array_equal(by_name["CAPE_MU"], cape_mu)


def test_precipitation_type():
    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _precipitation_type

    # SNOWLMT 1000 m, MELTLAYER_TOP 1100 m: melting-layer depth clamped to 150 m.
    n = 8
    snowlmt, meltlayer_top = np.full(n, 1000.0), np.full(n, 1100.0)
    rate = np.array([0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan])
    orography = np.array([1200.0, 1200.0, 1050.0, 900.0, 800.0, 800.0, 800.0, 800.0])
    t_2m = np.array([270.0, 270.0, 272.0, 274.0, 280.0, 273.45, 274.15, 280.0])
    ground = np.array([270.0, 270.0, 272.0, 274.0, 280.0, 272.15, 272.15, 280.0])
    prec_type = _precipitation_type(rate, snowlmt, meltlayer_top, t_2m, ground_temperature=ground, orography=orography)
    # none (< 0.1 mm/h), snow, mixed above / below SNOWLMT, rain, freezing rain,
    # rain on frozen ground under air above 0.5 degC, missing rate.
    np.testing.assert_array_equal(prec_type, [0, 3, 2, 2, 1, 4, 1, np.nan])


@pytest.mark.parametrize(("method", "expected"), [("linear", [0.0, 0.5, 1.0]), ("logistic", [0.0, 0.5, 1.0])])
def test_snow_fraction(method, expected):
    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _snow_fraction

    # Melting layer 300 m deep above SNOWLMT at 1000 m.
    fraction = _snow_fraction(np.full(3, 1000.0), np.full(3, 1300.0), np.array([900.0, 1150.0, 1400.0]), method)
    if method == "logistic":
        expected = [1.0 / (1.0 + np.exp(5.0)), 0.5, 1.0 / (1.0 + np.exp(-5.0))]
    np.testing.assert_allclose(fraction, expected)


def _with_precipitation(total, hours):
    from datetime import timedelta

    step = timedelta(hours=hours)
    fields = list(_vertical_fieldlist(step=step))
    fields.append(_array_field(np.full(4, total), param="tp", step=step))
    return new_fieldlist_from_list(fields)


def test_surface_diagnostics_precipitation_type_and_snow(orography_file):
    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _precipitation_type
    from anemoi_plugins_meteoswiss.transform.filters.surface_diagnostics import _snow_fraction

    diagnostics = _vertical_diagnostics(
        orography_file, ["SNOWLMT", "MELTLAYER_TOP", "PREC_TYPE", "TOT_SNOW"], precipitation="tp"
    )

    def step(total, hours):
        result = diagnostics.forward(_with_precipitation(total, hours))
        return {f.metadata("param"): f.to_numpy(flatten=True)[:4] for f in result}

    # First step: no previous precipitation.
    first = step(0.0, 0)
    assert np.all(np.isnan(first["PREC_TYPE"]))
    np.testing.assert_array_equal(first["TOT_SNOW"], 0.0)

    # 3 mm in 1 h everywhere.
    second = step(3.0, 1)
    zs, zm = second["SNOWLMT"], second["MELTLAYER_TOP"]
    np.testing.assert_array_equal(
        second["PREC_TYPE"],
        _precipitation_type(np.full(4, 3.0), zs, zm, _T_2M, ground_temperature=_T_2M, orography=_OROGRAPHY),
    )
    snow = 3.0 * _snow_fraction(zs, zm, _OROGRAPHY, "linear")
    np.testing.assert_allclose(second["TOT_SNOW"], snow)
    # The cold column snows; the lowland column rains.
    assert second["PREC_TYPE"][2] == 3 and second["PREC_TYPE"][0] == 1

    # No precipitation in the next hour: type 0, snow unchanged.
    third = step(3.0, 2)
    np.testing.assert_array_equal(third["PREC_TYPE"], 0)
    np.testing.assert_allclose(third["TOT_SNOW"], snow)

    # A new forecast (lead time back to 0) starts over.
    restart = step(0.0, 0)
    assert np.all(np.isnan(restart["PREC_TYPE"]))
    np.testing.assert_array_equal(restart["TOT_SNOW"], 0.0)


def test_surface_diagnostics_precipitation_type_requires_precipitation(orography_file):
    with pytest.raises(ValueError, match="precipitation"):
        _vertical_diagnostics(orography_file, ["PREC_TYPE"])


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
