"""Unit tests for the model-to-pressure-level filter, on small synthetic fields (no FDB)."""

import datetime as dt

import earthkit.data as ekd
import numpy as np
import pytest

from anemoi_plugins_meteoswiss.transform.filters import ModelToPressureLevel
from anemoi_plugins_meteoswiss.transform.filters import vertical_interpolation as vi

N_POINTS = 3
N_LEVELS = 10  # full model levels, 1 (top) to N_LEVELS (bottom)
BASE = dt.datetime(2020, 1, 29)

# Pressure on full levels, increasing downwards. Point 0 reaches 1050 hPa, point 1 only 800 hPa
# (mountain, so 850 hPa is below the surface), point 2 reaches 1000 hPa.
P_BOTTOM = np.array([105000.0, 80000.0, 100000.0])
P_TOP = 3000.0
P = np.exp(np.linspace(np.log(P_TOP), np.log(P_BOTTOM), N_LEVELS))  # (level, point)
HSURF = np.array([0.0, 2000.0, 100.0])
T_2M = np.array([285.0, 270.0, 280.0])
PS = P_BOTTOM + 500.0


def _field(param, values, valid, level=None, level_type="surface", units="1"):
    vertical = {"level": level if level is not None else 0, "level_type": level_type}
    return ekd.Field.from_components(
        values=np.asarray(values, dtype=float),
        parameter={"variable": param, "units": units},
        vertical=vertical,
        time={"base_datetime": BASE, "step": valid - BASE},
        geography={"latitudes": np.linspace(45.0, 47.0, N_POINTS), "longitudes": np.linspace(6.0, 10.0, N_POINTS)},
    )


def _model_level_fields(param, values, valid, level_type="general"):
    return [_field(param, v, valid, level=k + 1, level_type=level_type) for k, v in enumerate(values)]


def _t_from_p(p):
    """Temperature linear in log(p): log interpolation reproduces it exactly."""
    return 200.0 + 8.0 * np.log(p / P_TOP)


def _hhl():
    """Half-level heights, N_LEVELS + 1 levels, decreasing downwards to HSURF."""
    return np.linspace(20000.0 + HSURF, HSURF, N_LEVELS + 1)


def _auxiliary(valid, constants=True):
    fields = [_field("T_2M", T_2M, valid), _field("PS", PS, valid)]
    fields += _model_level_fields("P", P, valid)
    if constants:
        fields += [_field("HSURF", HSURF, valid)]
        fields += _model_level_fields("HHL", _hhl(), valid, level_type="generalVertical")
    return fields


def _input(valid, constants=True):
    u = np.tile(np.arange(N_LEVELS, dtype=float)[:, None], (1, N_POINTS))  # U == level index - 1
    w_half = np.tile(np.arange(N_LEVELS + 1, dtype=float)[:, None], (1, N_POINTS))  # W == half level index - 1
    fields = _model_level_fields("T", _t_from_p(P), valid)
    fields += _model_level_fields("U", u, valid)
    fields += _model_level_fields("W", w_half, valid, level_type="generalVertical")
    fields += [_field("CLCT", np.full(N_POINTS, 0.5), valid)]
    return fields + _auxiliary(valid, constants)


def _by_param_level(fieldlist):
    return {(f.get("parameter.variable"), f.get("vertical.level")): f for f in fieldlist}


def test_forward_interpolates_extrapolates_and_sets_pressure_level_metadata():
    valid = BASE + dt.timedelta(hours=12)
    data = ekd.create_fieldlist(list(reversed(_input(valid))))  # order of the input does not matter
    result = ModelToPressureLevel(interpolate_levels=[500, 850], extrapolate_levels=[850]).forward(data)

    out = _by_param_level(result)
    # T, U, W, FI on 2 levels; surface fields (including piped surface auxiliaries) passed through; P, HHL dropped
    surface = {("CLCT", 0), ("T_2M", 0), ("PS", 0), ("HSURF", 0)}
    assert len(result) == len(out) == 4 * 2 + len(surface)
    assert set(out) == {(p, lev) for p in ("T", "U", "W", "FI") for lev in (500, 850)} | surface
    assert [f.get("parameter.variable") for f in result][-len(surface) :] == ["CLCT", "HSURF", "PS", "T_2M"]

    for (param, level), field in out.items():
        assert field.get("time.valid_datetime") == valid
        if (param, level) in surface:
            assert field.get("vertical.level_type") == "surface"
            continue
        assert field.get("vertical.level_type") == "pressure"
        assert str(field.get("vertical.units")) == "hectopascal"
        assert field.to_numpy().dtype == np.float32
        assert np.isfinite(field.to_numpy()).all()

    # interpolation (log in pressure)
    np.testing.assert_allclose(out["T", 500].to_numpy(), _t_from_p(50000.0) * np.ones(N_POINTS), rtol=1e-5)
    t850 = out["T", 850].to_numpy()
    np.testing.assert_allclose(t850[[0, 2]], _t_from_p(85000.0), rtol=1e-5)

    # below the surface (point 1): temperature and geopotential extrapolated from the surface,
    # other parameters take the value of the lowest model level
    t_extrap = vi.extrapolate_temperature_sfc2p(
        vi.xr.DataArray(T_2M), vi.xr.DataArray(HSURF), vi.xr.DataArray(PS), 85000.0
    ).squeeze()
    np.testing.assert_allclose(t850[1], t_extrap.values[1], rtol=1e-6)
    assert out["U", 850].to_numpy()[1] == N_LEVELS - 1
    fi_extrap = vi.extrapolate_geopotential_sfc2p(
        vi.xr.DataArray(HSURF), vi.xr.DataArray(T_2M), vi.xr.DataArray(PS), 85000.0
    ).squeeze()
    np.testing.assert_allclose(out["FI", 850].to_numpy()[1], fi_extrap.values[1], rtol=1e-6)

    # W is destaggered: full level k is the mean of half levels k and k + 1, i.e. k - 0.5
    u500, w500 = out["U", 500].to_numpy(), out["W", 500].to_numpy()
    np.testing.assert_allclose(w500, u500 + 0.5, rtol=1e-6)

    # FI from destaggered HHL
    hhl_full = 0.5 * (_hhl()[:-1] + _hhl()[1:])
    fi_full = (hhl_full * 9.80665).astype(np.float32)
    assert (out["FI", 500].to_numpy() < fi_full[0]).all() and (out["FI", 500].to_numpy() > fi_full[-1]).all()
    assert str(out["FI", 500].get("parameter.units")) == "meter ** 2 / second ** 2"


def test_constant_auxiliary_fetched_once_and_restamped(monkeypatch):
    """HSURF/HHL come from FDB for a fixed date, are cached, and take the time of each timestep (#25)."""
    constant_date = dt.datetime(2020, 1, 1)
    constants = {
        "HSURF": ekd.create_fieldlist([_field("HSURF", HSURF, constant_date)]),
        "HHL": ekd.create_fieldlist(_model_level_fields("HHL", _hhl(), constant_date, level_type="generalVertical")),
    }
    requests = []

    def request_fdb(shortname, spec, time_keys):
        requests.append((shortname, time_keys))
        return constants[shortname]

    interpolator = ModelToPressureLevel(interpolate_levels=[500])
    monkeypatch.setattr(interpolator, "_request_fdb", request_fdb)

    valids = [BASE + dt.timedelta(hours=h) for h in (1, 2)]
    data = ekd.create_fieldlist([f for valid in valids for f in _input(valid, constants=False)])
    result = interpolator.forward(data)

    assert [r[0] for r in requests] == ["HSURF", "HHL"]  # fetched once, then cached
    assert all(r[1] == vi.CONSTANT_TIME_KEYS for r in requests)
    assert all(f.get("time.valid_datetime") == constant_date for fl in constants.values() for f in fl)

    fi = [f for f in result if f.get("parameter.variable") == "FI"]
    assert sorted(f.get("time.valid_datetime") for f in fi) == valids
    for valid in valids:
        restamped = interpolator._get_field(ekd.create_fieldlist(_input(valid, constants=False)), "HSURF")
        assert [f.get("time.valid_datetime") for f in restamped] == [valid]
        np.testing.assert_array_equal(restamped[0].to_numpy(), HSURF)


def test_time_varying_auxiliary_fetched_for_valid_time(monkeypatch):
    valid = BASE + dt.timedelta(hours=7)
    available = ekd.create_fieldlist(_auxiliary(valid))
    requests = []

    def request_fdb(shortname, spec, time_keys):
        requests.append((shortname, time_keys))
        return available.sel({"parameter.variable": shortname})

    interpolator = ModelToPressureLevel(interpolate_levels=[500], add_geopotential=False)
    monkeypatch.setattr(interpolator, "_request_fdb", request_fdb)

    data = ekd.create_fieldlist(_model_level_fields("T", _t_from_p(P), valid))
    result = interpolator.forward(data)

    assert [r[0] for r in requests] == ["T_2M", "PS", "P", "HSURF"]
    time_keys = {"date": "20200129", "time": "0000", "step": "7"}
    assert all(r[1] == time_keys for r in requests if r[0] != "HSURF")
    assert [(f.get("parameter.variable"), f.get("vertical.level")) for f in result] == [("T", 500)]


def test_request_fdb_reads_all_fields(monkeypatch):
    calls = []

    class _Source:
        def to_fieldlist(self, **kwargs):
            calls.append(kwargs)
            return "fieldlist"

    def from_source(name, request):
        calls.append((name, request))
        return _Source()

    monkeypatch.setattr(vi.ekd, "from_source", from_source)
    interpolator = ModelToPressureLevel(interpolate_levels=[500])
    assert interpolator._request_fdb("P", vi.AUXILIARY_VARIABLES["P"], vi.CONSTANT_TIME_KEYS) == "fieldlist"
    (name, request), kwargs = calls
    assert name == "fdb"
    assert request == vi.BASE_REQUEST | {"param": "P", "levtype": "ml", "levelist": "1/to/81"} | vi.CONSTANT_TIME_KEYS
    assert kwargs == {"read_all": True}


def test_to_dataarray_sorts_levels_and_rejects_duplicates():
    valid = BASE
    fields = _model_level_fields("P", P, valid)
    da = vi._to_dataarray(ekd.create_fieldlist(fields[::-1]))
    assert da.dims == ("level", "values")
    assert da.dtype == np.float32
    assert list(da["level"].values) == list(range(1, N_LEVELS + 1))
    np.testing.assert_array_equal(da.values, P.astype(np.float32))

    with pytest.raises(ValueError, match="Duplicate levels"):
        vi._to_dataarray(ekd.create_fieldlist(fields + fields[:1]))

    sfc = vi._to_dataarray(ekd.create_fieldlist([_field("PS", PS, valid)]), vertical=False)
    assert sfc.dims == ("values",)
    with pytest.raises(ValueError, match="single field"):
        vi._to_dataarray(ekd.create_fieldlist(fields), vertical=False)
