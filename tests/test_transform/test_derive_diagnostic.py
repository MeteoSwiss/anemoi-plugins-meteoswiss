import functools
from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pytest
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.meteo.thermo import relative_humidity_from_dewpoint
from earthkit.meteo.thermo import wet_bulb_temperature_from_dewpoint
from earthkit.meteo.thermo import wet_bulb_temperature_from_specific_humidity

from anemoi_plugins_meteoswiss.transform.filters import DeriveDiagnostic


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


def test_derive_diagnostic_all_variables(data_dir):
    fieldlist, u_values, v_values, t_values, td_values = _synthetic_components(data_dir)

    diagnostics = DeriveDiagnostic(
        u_10m="10u",
        v_10m="10v",
        temperature_2m="2t",
        dewpoint_2m="2d",
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


def test_derive_diagnostic_supersaturation_capped(data_dir):
    template = ekd.from_source("file", str(data_dir / "iaf2025010100")).sel(shortName="T_2M")[0]
    n = template.to_numpy(flatten=True).size
    rng = np.random.default_rng(1)
    t_values = rng.uniform(250.0, 300.0, n)
    # Dewpoint up to 2 K above the temperature, as the model sometimes predicts.
    td_values = t_values + rng.uniform(-5.0, 2.0, n)
    sp_values = rng.uniform(70_000.0, 102_000.0, n)
    fieldlist = new_fieldlist_from_list(
        [
            new_field_from_numpy(t_values, template=template, param="2t", shortName="2t"),
            new_field_from_numpy(td_values, template=template, param="2d", shortName="2d"),
            new_field_from_numpy(sp_values, template=template, param="sp", shortName="sp"),
        ]
    )

    result = DeriveDiagnostic(
        temperature_2m="2t", dewpoint_2m="2d", surface_pressure="sp", variables=["RELHUM_2M", "TW_2M"]
    ).forward(fieldlist)
    by_param = {f.metadata("param"): f for f in result}

    rh = by_param["RELHUM_2M"].to_numpy(flatten=True)
    tw = by_param["TW_2M"].to_numpy(flatten=True)
    supersaturated = td_values > t_values
    assert supersaturated.any() and (~supersaturated).any()
    assert np.all(rh <= 100.0)
    np.testing.assert_allclose(rh[supersaturated], 100.0)
    np.testing.assert_allclose(
        rh[~supersaturated], relative_humidity_from_dewpoint(t_values, td_values)[~supersaturated]
    )
    # TD_2M <= TW_2M <= T_2M, up to the GRIB packing precision of the template.
    assert np.all(tw <= t_values + 1e-3) and np.all(tw >= np.minimum(td_values, t_values) - 1e-3)
    np.testing.assert_allclose(tw[supersaturated], t_values[supersaturated], atol=1e-3)
    # The dewpoint itself is passed through unchanged.
    np.testing.assert_array_equal(by_param["2d"].to_numpy(flatten=True), td_values)


def test_derive_diagnostic_subset_of_variables(data_dir):
    fieldlist, u_values, v_values, t_values, td_values = _synthetic_components(data_dir)

    # Only SP_10M requested: DD_10M and RELHUM_2M are skipped, and
    # temperature/dewpoint are not matched but passed through.
    diagnostics = DeriveDiagnostic(
        u_10m="10u",
        v_10m="10v",
        temperature_2m="2t",
        dewpoint_2m="2d",
        variables="SP_10M",
    )
    result = diagnostics.forward(fieldlist)

    by_param = {f.metadata("param"): f for f in result}
    assert set(by_param) == {"10u", "10v", "2t", "2d", "SP_10M"}

    expected_speed = np.sqrt(u_values**2 + v_values**2)
    np.testing.assert_allclose(by_param["SP_10M"].to_numpy(flatten=True), expected_speed)


def test_derive_diagnostic_collects_garbage_each_step(data_dir, monkeypatch):
    """Each call frees the states of the previous steps kept alive by reference cycles."""
    from anemoi_plugins_meteoswiss.transform.filters import derive_diagnostic

    calls = []
    monkeypatch.setattr(derive_diagnostic.gc, "collect", lambda: calls.append(1))
    fieldlist, *_ = _synthetic_components(data_dir)
    diagnostics = DeriveDiagnostic(u_10m="10u", v_10m="10v", variables="SP_10M")
    for _ in range(3):
        diagnostics.forward(fieldlist)
    assert len(calls) == 3


def test_derive_diagnostic_requires_at_least_one_variable():
    with pytest.raises(ValueError, match="at least one"):
        DeriveDiagnostic(
            u_10m="10u",
            v_10m="10v",
            temperature_2m="2t",
            dewpoint_2m="2d",
            variables=[],
        )


def test_derive_diagnostic_rejects_unknown_variable():
    with pytest.raises(ValueError, match="Unsupported variable"):
        DeriveDiagnostic(
            u_10m="10u",
            v_10m="10v",
            temperature_2m="2t",
            dewpoint_2m="2d",
            variables=["TOT_PREC"],
        )


def test_derive_diagnostic_requires_inputs_of_requested_variables():
    with pytest.raises(ValueError, match="surface_pressure"):
        DeriveDiagnostic(temperature_2m="2t", dewpoint_2m="2d", surface_pressure=None, variables=["TW_2M"])


def test_derive_diagnostic_vertical_requires_orography():
    with pytest.raises(ValueError, match="orography_file"):
        DeriveDiagnostic(
            temperature_2m="2t",
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


def _melting_layer():
    """SNOWLMT and the internal MELTLAYER_TOP of the test columns, from the
    filter's own height-level helper (MELTLAYER_TOP is not an output)."""
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _snow_and_melting_levels

    pressure = np.array(_LEVELS, dtype=float) * 100.0
    tw_levels = wet_bulb_temperature_from_specific_humidity(
        _T_LEVELS.ravel(), _Q_LEVELS.ravel(), np.repeat(pressure, 4)
    ).reshape(_T_LEVELS.shape)
    return _snow_and_melting_levels(
        tw_2m=wet_bulb_temperature_from_dewpoint(_T_2M, _TD_2M, _SP),
        orography=_OROGRAPHY,
        tw_levels=tw_levels,
        level_heights=np.repeat(np.array(_LEVEL_HEIGHTS)[:, None], 4, axis=1),
        above_ground=pressure[:, None] < _SP[None, :],
    )


def _vertical_diagnostics(orography_file, variables, **kwargs):
    return DeriveDiagnostic(
        temperature_2m="2t",
        dewpoint_2m="2d",
        surface_pressure="sp",
        temperature_levels="t",
        specific_humidity_levels="q",
        geopotential_levels="z",
        orography_file=str(orography_file),
        variables=variables,
        **kwargs,
    )


def test_derive_diagnostic_tw_2m():
    fieldlist = _vertical_fieldlist()
    result = DeriveDiagnostic(temperature_2m="2t", dewpoint_2m="2d", surface_pressure="sp", variables="TW_2M").forward(
        fieldlist
    )

    by_name = {f.metadata("param"): f for f in result if f.metadata("param") == "TW_2M"}
    expected = wet_bulb_temperature_from_dewpoint(_T_2M, _TD_2M, _SP)
    np.testing.assert_allclose(by_name["TW_2M"].to_numpy(flatten=True)[:4], expected)
    assert np.all(expected <= _T_2M) and np.all(expected >= _TD_2M)
    # All inputs, including the untouched pressure levels, are still there.
    assert len(result) == len(fieldlist) + 1


def test_derive_diagnostic_vertical_keeps_pressure_levels(orography_file):
    fieldlist = _vertical_fieldlist()
    result = _vertical_diagnostics(orography_file, ["TW_2M", "HZEROCL", "SNOWLMT"]).forward(fieldlist)

    assert len(result) == len(fieldlist) + 3
    params = [f.metadata("param") for f in result]
    assert {"TW_2M", "HZEROCL", "SNOWLMT"} <= set(params)
    assert "MELTLAYER_TOP" not in params
    for param in ("t", "q", "z"):
        levels = sorted(f.metadata("levelist") for f in result if f.metadata("param") == param)
        assert levels == sorted(_LEVELS)
    t_850 = next(f for f in result if f.metadata("param") == "t" and f.metadata("levelist") == 850)
    np.testing.assert_array_equal(t_850.to_numpy(flatten=True), _tiled(_T_LEVELS[1]))


def test_derive_diagnostic_rejects_orography_size_mismatch(tmp_path):
    import xarray as xr

    path = tmp_path / "extpar.nc"
    xr.Dataset({"topography_c": ("cell", np.zeros(7))}).to_netcdf(path)
    with pytest.raises(ValueError, match="native grid"):
        _vertical_diagnostics(path, ["HZEROCL"]).forward(_vertical_fieldlist())


def test_derive_diagnostic_height_levels(orography_file):
    result = _vertical_diagnostics(orography_file, ["HZEROCL", "SNOWLMT"]).forward(_vertical_fieldlist())

    by_name = {f.metadata("param"): f.to_numpy(flatten=True)[:4] for f in result}
    z0, zs = by_name["HZEROCL"], by_name["SNOWLMT"]
    melting_layer = _melting_layer()
    np.testing.assert_allclose(melting_layer["SNOWLMT"], zs)
    zm = melting_layer["MELTLAYER_TOP"]
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
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _integrate_cape_cin

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
    cape, cin = _integrate_cape_cin(buoyancy, height, pressure, p_lcl, pressure[0])

    # 0: LFC and EL mid-segment: CAPE = 2.5 + 10 + 2.5, CIN = 5 + 2.5.
    # 1: positive buoyancy below the LCL (superadiabatic) is ignored; LFC above
    #    the LCL, no EL: CAPE = 2.5 + 10, CIN = 2.5 + 2.5.
    # 2: buoyant at the LCL itself: LFC at the LCL, CAPE = 10 + 10, CIN = 5 + 2.5.
    # 3: never buoyant: no LFC, CAPE 0 and CIN missing.
    np.testing.assert_allclose(cape, [15.0, 12.5, 20.0, 0.0])
    np.testing.assert_allclose(cin, [7.5, 5.0, 7.5, np.nan])


def test_cape_cin_ignores_layers_below_parcel_start():
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _integrate_cape_cin

    height = np.arange(0.0, 5000.0, 1000.0)[:, None]
    pressure = np.array([100000.0, 90000.0, 80000.0, 70000.0, 60000.0])[:, None]
    buoyancy = np.array([[0.0], [-0.02], [0.0], [0.01], [0.01]])
    # Parcel starting at 900 hPa with its LCL there: the negative layer below
    # 900 hPa doesn't count as CIN.
    cape, cin = _integrate_cape_cin(buoyancy, height, pressure, np.array([90000.0]), np.array([90000.0]))
    np.testing.assert_allclose(cin, [10.0])
    np.testing.assert_allclose(cape, [15.0])


def test_mixed_layer_parcel():
    from earthkit.meteo.thermo import potential_temperature
    from earthkit.meteo.thermo import temperature_from_potential_temperature

    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _mixed_layer_parcel

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
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _most_unstable_parcel

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
def test_derive_diagnostic_cape_cin(orography_file, kind):
    fieldlist = _vertical_fieldlist()
    result = _vertical_diagnostics(orography_file, [f"CAPE_{kind}", f"CIN_{kind}"]).forward(fieldlist)

    # Inputs, including the pressure levels, are kept.
    assert len(result) == len(fieldlist) + 2
    cape = next(f for f in result if f.metadata("param") == f"CAPE_{kind}").to_numpy(flatten=True)[:4]
    cin = next(f for f in result if f.metadata("param") == f"CIN_{kind}").to_numpy(flatten=True)[:4]
    assert np.all(cape >= 0.0)
    assert np.all(np.isnan(cin) | (cin >= 0.0))
    # Without an LFC there is no CAPE and CIN is missing; e.g. the column below 0 degC.
    assert cape[2] == 0.0 and np.isnan(cin[2])
    np.testing.assert_array_equal(np.isnan(cin), cape == 0.0)


def test_derive_diagnostic_cape_cin_ml_and_mu(orography_file):
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


# ICON-like interface heights for the test columns: 40 layers, thin near the
# ground, terrain-following, model top at 22 km.
_HHL_TOP = 22000.0
_HHL_FLAT = _HHL_TOP * np.linspace(1.0, 0.0, 41) ** 2


def _hhl(orography):
    return _HHL_FLAT[:, None] + np.asarray(orography)[None, :] * (1.0 - _HHL_FLAT[:, None] / _HHL_TOP)


@pytest.fixture
def vertical_grid_file(tmp_path):
    path = tmp_path / "hhl.nc"
    xr.Dataset({"HHL": (("interface", "cell"), _hhl(_tiled(_OROGRAPHY)))}).to_netcdf(path)
    return path


def _icon_cape_cin_outputs(orography_file, vertical_grid_file, variables, **kwargs):
    result = _vertical_diagnostics(
        orography_file, variables, cape_cin_method="icon", vertical_grid_file=str(vertical_grid_file), **kwargs
    ).forward(_vertical_fieldlist())
    return {f.metadata("param"): f.to_numpy(flatten=True) for f in result if f.metadata("param") in variables}


def test_derive_diagnostic_cape_cin_icon(orography_file, vertical_grid_file):
    from anemoi_plugins_meteoswiss.transform.filters import _icon_cape_cin
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _environment_column
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _icon_integration_tops
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _on_model_levels

    variables = ["CAPE_ML", "CIN_ML", "CAPE_MU", "CIN_MU"]
    by_name = {name: x[:4] for name, x in _icon_cape_cin_outputs(orography_file, vertical_grid_file, variables).items()}

    # The ICON port on the test columns put on the model levels, from ICON's model surface.
    hhl = _hhl(_OROGRAPHY)
    pressure = np.array(_LEVELS, dtype=float) * 100.0
    column = _environment_column(
        t_2m=_T_2M,
        td_2m=_TD_2M,
        sp=_SP,
        orography=hhl[-1],
        pressure=pressure,
        t_levels=_T_LEVELS,
        q_levels=_Q_LEVELS,
        level_heights=np.repeat(np.array(_LEVEL_HEIGHTS)[:, None], 4, axis=1),
        above_ground=pressure[:, None] < _SP[None, :],
    )
    te, qve, prs = _on_model_levels(*column, hhl)
    tops = _icon_integration_tops(hhl)
    for kind in ("ML", "MU"):
        cape, cin = _icon_cape_cin.cape_cin(te, qve, prs, hhl.T.copy(), kmoist=tops[kind], parcel=kind.lower())
        np.testing.assert_array_equal(by_name[f"CAPE_{kind}"], cape)
        np.testing.assert_array_equal(by_name[f"CIN_{kind}"], np.where(cin == _icon_cape_cin.MISSING, np.nan, cin))
        # ICON's CIN is missing without CAPE; the moist lowland column has some.
        np.testing.assert_array_equal(np.isnan(by_name[f"CIN_{kind}"]), by_name[f"CAPE_{kind}"] < 1e-8)
        assert by_name[f"CAPE_{kind}"][0] > 0.0
    # Not the integration on the pressure levels.
    default = _vertical_diagnostics(orography_file, ["CAPE_ML"]).forward(_vertical_fieldlist())
    cape_default = next(f for f in default if f.metadata("param") == "CAPE_ML").to_numpy(flatten=True)[:4]
    assert by_name["CAPE_ML"][0] != cape_default[0]


def test_derive_diagnostic_cape_cin_icon_chunks_and_threads(orography_file, vertical_grid_file, monkeypatch):
    """Chunks of points and threads give exactly the single-chunk, single-thread result."""
    from anemoi_plugins_meteoswiss.transform.filters import derive_diagnostic

    variables = ["CAPE_ML", "CIN_ML", "CAPE_MU", "CIN_MU"]
    reference = _icon_cape_cin_outputs(orography_file, vertical_grid_file, variables, n_threads=1)
    monkeypatch.setattr(derive_diagnostic, "ICON_CHUNK_SIZE", 100003)
    chunked = _icon_cape_cin_outputs(orography_file, vertical_grid_file, variables, n_threads=3)
    for name in variables:
        np.testing.assert_array_equal(chunked[name], reference[name])


def test_derive_diagnostic_cape_cin_pressure_levels_ignores_vertical_grid(orography_file, vertical_grid_file):
    """The default method is pressure_levels, and the vertical grid doesn't change it."""
    variables = ["CAPE_ML", "CIN_ML", "CAPE_MU", "CIN_MU"]

    def run(**kwargs):
        result = _vertical_diagnostics(orography_file, variables, **kwargs).forward(_vertical_fieldlist())
        return {f.metadata("param"): f.to_numpy(flatten=True) for f in result if f.metadata("param") in variables}

    default = run()
    explicit = run(cape_cin_method="pressure_levels", vertical_grid_file=str(vertical_grid_file))
    for name in variables:
        np.testing.assert_array_equal(explicit[name], default[name])


def test_derive_diagnostic_cape_cin_icon_requires_vertical_grid(orography_file):
    with pytest.raises(ValueError, match="vertical_grid_file"):
        _vertical_diagnostics(orography_file, ["CAPE_ML"], cape_cin_method="icon")
    # Without CAPE/CIN the vertical grid isn't needed.
    _vertical_diagnostics(orography_file, ["HZEROCL"], cape_cin_method="icon")


def test_derive_diagnostic_rejects_unknown_cape_cin_method(orography_file):
    with pytest.raises(ValueError, match="cape_cin_method"):
        _vertical_diagnostics(orography_file, ["CAPE_ML"], cape_cin_method="cosmo")


def test_derive_diagnostic_rejects_vertical_grid_size_mismatch(orography_file, tmp_path):
    path = tmp_path / "hhl.nc"
    xr.Dataset({"HHL": (("interface", "cell"), _hhl(np.zeros(7)))}).to_netcdf(path)
    diagnostics = _vertical_diagnostics(orography_file, ["CAPE_ML"], cape_cin_method="icon", vertical_grid_file=path)
    with pytest.raises(ValueError, match="native grid"):
        diagnostics.forward(_vertical_fieldlist())


def test_on_model_levels():
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _on_model_levels

    # Bottom-up column: 2 m point, a level collapsed onto it (below ground), two
    # levels above. T and q linear in height, log p linear in height.
    height = np.array([[2.0], [2.0], [1002.0], [3002.0]])
    t = 290.0 - 0.0065 * (height - 2.0)
    q = 0.010 - 2e-6 * (height - 2.0)
    p = 100000.0 * np.exp(-(height - 2.0) / 8000.0)
    # Full levels at 4002 m (above the column top), 2002 m and 502 m.
    hhl = np.array([[5002.0], [3002.0], [1002.0], [2.0]])
    te, qve, prs = _on_model_levels(p, height, t, q, hhl)
    full = 0.5 * (hhl[:-1] + hhl[1:])[:, 0]
    np.testing.assert_allclose(prs[0], 100000.0 * np.exp(-(full - 2.0) / 8000.0))
    # Below the column top T and q are interpolated; above it they're kept constant.
    np.testing.assert_allclose(te[0, 1:], 290.0 - 0.0065 * (full[1:] - 2.0))
    np.testing.assert_allclose(qve[0, 1:], 0.010 - 2e-6 * (full[1:] - 2.0))
    np.testing.assert_allclose([te[0, 0], qve[0, 0]], [t[-1, 0], q[-1, 0]])


def test_icon_integration_tops():
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _icon_integration_tops

    # Full levels at 23000, 21000, 19300 and 9300 m: the first below 22.5 km is
    # level 1; the first above 60 hPa in the standard atmosphere (19300 m:
    # 61 hPa, 21000 m: 47 hPa) is level 2.
    hhl = np.repeat(np.array([[24000.0], [22000.0], [20000.0], [18600.0], [0.0]]), 3, axis=1)
    assert _icon_integration_tops(hhl) == {"ML": 2, "MU": 1}


def test_precipitation_type():
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _precipitation_type

    # SNOWLMT 1000 m, MELTLAYER_TOP 1100 m: melting-layer depth clamped to 150 m.
    n = 8
    snowlmt, meltlayer_top = np.full(n, 1000.0), np.full(n, 1100.0)
    rate = np.array([0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan])
    orography = np.array([1200.0, 1200.0, 1050.0, 900.0, 800.0, 800.0, 800.0, 800.0])
    t_2m = np.array([270.0, 270.0, 272.0, 274.0, 280.0, 273.45, 274.15, 280.0])
    ground = np.array([270.0, 270.0, 272.0, 274.0, 280.0, 272.15, 272.15, 280.0])
    prec_type = _precipitation_type(rate, snowlmt, meltlayer_top, t_2m, ground_temperature=ground, orography=orography)
    # none (< 0.1 mm/h), snow, mixed above / below SNOWLMT, rain, freezing rain,
    # rain on frozen ground under air above 0.5 degC, missing rate (WMO code table 4.201).
    np.testing.assert_array_equal(prec_type, [0, 5, 7, 7, 1, 3, 1, np.nan])


@pytest.mark.parametrize(("method", "expected"), [("linear", [0.0, 0.5, 1.0]), ("logistic", [0.0, 0.5, 1.0])])
def test_snow_fraction(method, expected):
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _snow_fraction

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


def test_derive_diagnostic_precipitation_type_and_snow(orography_file):
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _precipitation_type
    from anemoi_plugins_meteoswiss.transform.filters.derive_diagnostic import _snow_fraction

    diagnostics = _vertical_diagnostics(
        orography_file, ["SNOWLMT", "PTYPE", "TOT_SNOW"], accumulated_precipitation="tp"
    )

    def step(total, hours):
        result = diagnostics.forward(_with_precipitation(total, hours))
        return {f.metadata("param"): f.to_numpy(flatten=True)[:4] for f in result}

    # First step: no previous precipitation.
    first = step(0.0, 0)
    assert np.all(np.isnan(first["PTYPE"]))
    np.testing.assert_array_equal(first["TOT_SNOW"], 0.0)

    # 3 mm in 1 h everywhere.
    second = step(3.0, 1)
    zs, zm = second["SNOWLMT"], _melting_layer()["MELTLAYER_TOP"]
    np.testing.assert_array_equal(
        second["PTYPE"],
        _precipitation_type(np.full(4, 3.0), zs, zm, _T_2M, ground_temperature=_T_2M, orography=_OROGRAPHY),
    )
    snow = 3.0 * _snow_fraction(zs, zm, _OROGRAPHY, "linear")
    np.testing.assert_allclose(second["TOT_SNOW"], snow)
    # The cold column snows; the lowland column rains.
    assert second["PTYPE"][2] == 5 and second["PTYPE"][0] == 1

    # No precipitation in the next hour: type 0, snow unchanged.
    third = step(3.0, 2)
    np.testing.assert_array_equal(third["PTYPE"], 0)
    np.testing.assert_allclose(third["TOT_SNOW"], snow)

    # A new forecast (lead time back to 0) starts over.
    restart = step(0.0, 0)
    assert np.all(np.isnan(restart["PTYPE"]))
    np.testing.assert_array_equal(restart["TOT_SNOW"], 0.0)


def test_derive_diagnostic_initial_state_without_precipitation(orography_file):
    """The initial state has no precipitation yet: it is taken as 0 there."""
    from datetime import timedelta

    diagnostics = _vertical_diagnostics(
        orography_file, ["SNOWLMT", "PTYPE", "TOT_SNOW"], accumulated_precipitation="tp"
    )

    initial = {
        f.metadata("param"): f.to_numpy(flatten=True)[:4]
        for f in diagnostics.forward(_vertical_fieldlist(step=timedelta(0)))
    }
    assert "tp" not in initial
    assert np.all(np.isnan(initial["PTYPE"]))
    np.testing.assert_array_equal(initial["TOT_SNOW"], 0.0)

    # The next step accumulates from the initial state.
    first = {
        f.metadata("param"): f.to_numpy(flatten=True)[:4] for f in diagnostics.forward(_with_precipitation(3.0, 1))
    }
    assert not np.isnan(first["PTYPE"]).any()
    assert first["PTYPE"][2] == 5 and first["PTYPE"][0] == 1


def test_derive_diagnostic_missing_precipitation_after_initial_state_raises(orography_file):
    from datetime import timedelta

    diagnostics = _vertical_diagnostics(orography_file, ["PTYPE"], accumulated_precipitation="tp")

    with pytest.raises(ValueError, match="precipitation 'tp' missing"):
        diagnostics.forward(_vertical_fieldlist(step=timedelta(hours=1)))


def test_derive_diagnostic_rejects_meltlayer_top(orography_file):
    with pytest.raises(ValueError, match="Unsupported"):
        _vertical_diagnostics(orography_file, ["MELTLAYER_TOP"])


def test_derive_diagnostic_precipitation_type_requires_precipitation(orography_file):
    with pytest.raises(ValueError, match="precipitation"):
        _vertical_diagnostics(orography_file, ["PTYPE"], accumulated_precipitation=None)


def test_derive_diagnostic_default_inputs_are_grib_names():
    diagnostics = DeriveDiagnostic(variables=["SP_10M", "RELHUM_2M"])
    assert diagnostics.MATCHING.forward == ("u_10m", "v_10m", "temperature_2m", "dewpoint_2m")
    assert (diagnostics.u_10m, diagnostics.v_10m) == ("U_10M", "V_10M")
    assert (diagnostics.temperature_2m, diagnostics.dewpoint_2m) == ("T_2M", "TD_2M")


def test_derive_diagnostic_threads_match_single_thread(orography_file):
    """Splitting the points over threads gives exactly the single-thread result."""
    variables = ["TW_2M", "HZEROCL", "SNOWLMT", "CAPE_ML", "CIN_ML", "CAPE_MU", "CIN_MU", "PTYPE", "TOT_SNOW"]

    def run(n_threads):
        diagnostics = _vertical_diagnostics(
            orography_file, variables, accumulated_precipitation="tp", n_threads=n_threads
        )
        return [
            {
                f.metadata("param"): f.to_numpy(flatten=True)
                for f in diagnostics.forward(_with_precipitation(total, hours))
            }
            for total, hours in ((1.0, 1), (3.0, 2))
        ]

    for single, threaded in zip(run(1), run(3)):
        for name in variables:
            np.testing.assert_array_equal(threaded[name], single[name])


@pytest.mark.parametrize("n_threads", [0, -1, 1.5])
def test_derive_diagnostic_rejects_invalid_n_threads(orography_file, n_threads):
    with pytest.raises(ValueError, match="n_threads"):
        _vertical_diagnostics(orography_file, ["HZEROCL"], n_threads=n_threads)
