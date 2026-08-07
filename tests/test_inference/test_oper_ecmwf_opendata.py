"""Unit tests for OperEcmwfOpenDataInput's run/step walk-back logic.

These need no network/FDB/HPC access — `ecmwf.opendata.Client.latest()` (the
real, network-backed "which run is actually published" check) is mocked out
everywhere here, unlike the `hostname.startswith("balfrin")`-gated tests
elsewhere in this suite.

The module under test lives behind the `oper-ecmwf-opendata` optional
dependency group (anemoi-inference + anemoi-plugins-ecmwf-inference, which
pull in mir/eckit/atlas) — skip this whole file cleanly if it isn't installed.
"""

import os
from datetime import datetime

import pytest

pytest.importorskip("anemoi.plugins.ecmwf.inference.opendata.opendata")

from anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata import OperEcmwfOpenDataInput
from anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata import _param_translation_from_variables_metadata
from anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata import _translate_params
from anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata import _without_eccodes_definition_path_override

# Mirrors the shape of the real (patched) checkpoint variables_metadata:
# mars.param is the KENDA/COSMO name, the variable's own key is the ECMWF one.
VARIABLES_METADATA = {
    "skt": {"mars": {"param": "SKT"}},
    "t_500": {"mars": {"param": "T", "levelist": 500}},
    "w_500": {"mars": {"param": "OMEGA", "levelist": 500}},
    "z_500": {"mars": {"param": "FI", "levelist": 500}},
    "z": {"mars": {"param": "FIS"}},
    "2t": {"mars": {"param": "T_2M"}},
    "cos_latitude": {"computed_forcing": True},  # no "mars" key
}


class _MetadataWithVariables:
    def __init__(self, variables_metadata):
        self.variables_metadata = variables_metadata


def _make_input(**overrides):
    """A OperEcmwfOpenDataInput with its attributes set directly, bypassing
    __init__ (which needs a real anemoi-inference Context/Metadata/checkpoint)."""
    obj = object.__new__(OperEcmwfOpenDataInput)
    obj.frequency_h = overrides.get("frequency_h", 6)
    obj.step_h = overrides.get("step_h", 3)
    obj.max_lead_time_h = overrides.get("max_lead_time_h", 144)
    obj.stored_runs = overrides.get("stored_runs", 12)
    return obj


def test_target_exactly_at_a_run_boundary_uses_step_zero():
    d = _make_input()
    run, step = d._run_and_step(datetime(2026, 8, 5, 6, 0), datetime(2026, 8, 5, 6, 0))
    assert (run, step) == (datetime(2026, 8, 5, 6, 0), 0)


def test_target_exactly_three_hours_ahead_succeeds():
    d = _make_input()
    run, step = d._run_and_step(datetime(2026, 8, 5, 9, 0), datetime(2026, 8, 5, 6, 0))
    assert (run, step) == (datetime(2026, 8, 5, 6, 0), 3)


def test_target_older_than_guaranteed_run_walks_back():
    d = _make_input()
    run, step = d._run_and_step(datetime(2026, 8, 4, 9, 0), datetime(2026, 8, 5, 6, 0))
    assert (run, step) == (datetime(2026, 8, 4, 6, 0), 3)


def test_target_too_old_raises():
    d = _make_input()
    with pytest.raises(ValueError, match="stored open data runs"):
        d._run_and_step(datetime(2026, 7, 1, 0, 0), datetime(2026, 8, 5, 6, 0))


def test_step_exceeding_max_lead_time_raises():
    d = _make_input(frequency_h=12, max_lead_time_h=6, stored_runs=100)
    with pytest.raises(ValueError, match="exceeds open data max lead time"):
        d._run_and_step(datetime(2026, 8, 5, 9, 0), datetime(2026, 8, 5, 0, 0))  # 9h ahead of the 00:00 run

def test_target_exactly_on_step_h_grid_succeeds():
    d = _make_input()
    run, step = d._run_and_step(datetime(2026, 8, 6, 9, 0), datetime(2026, 8, 6, 0, 0))
    assert (run, step) == (datetime(2026, 8, 6, 0, 0), 9)


def test_param_translation_strips_level_suffix_from_pressure_level_variables():
    metadata = _MetadataWithVariables(VARIABLES_METADATA)
    mapping = _param_translation_from_variables_metadata(metadata, ["t_500", "w_500", "z_500"])
    assert mapping == {"T": "t", "OMEGA": "w", "FI": "z"}


def test_param_translation_keeps_single_level_variables_unchanged():
    metadata = _MetadataWithVariables(VARIABLES_METADATA)
    mapping = _param_translation_from_variables_metadata(metadata, ["z", "2t", "skt"])
    assert mapping == {"FIS": "z", "T_2M": "2t", "SKT": "skt"}


def test_param_translation_skips_variables_without_mars_metadata():
    metadata = _MetadataWithVariables(VARIABLES_METADATA)
    mapping = _param_translation_from_variables_metadata(metadata, ["cos_latitude"])
    assert mapping == {}


def test_translate_params_maps_kenda_names_via_derived_mapping():
    # The exact request that failed against the real ECMWF Open Data API.
    request = {
        "type": ["fc"],
        "param": ["FI", "QV", "T", "U", "V"],
        "levelist": ["50", "100", "150"],
        "levtype": ["pl"],
    }
    mapping = {"FI": "z", "QV": "q", "T": "t", "U": "u", "V": "v"}
    translated = _translate_params(request, mapping)
    assert translated["param"] == ["z", "q", "t", "u", "v"]


def test_translate_params_leaves_unmapped_names_unchanged():
    assert _translate_params({"param": ["t", "unmapped"]}, {})["param"] == ["t", "unmapped"]


def test_translate_params_handles_scalar_param():
    assert _translate_params({"param": "OMEGA"}, {"OMEGA": "w"})["param"] == "w"


def test_translate_params_no_param_key_is_a_no_op():
    assert _translate_params({"levtype": "sfc"}, {}) == {"levtype": "sfc"}


class _FakeMetadata:
    grid = "n320"
    area = None
    variables_metadata = {"t_500": {"mars": {"param": "T"}}}

    def mars_requests(self, *, variables, dates, use_grib_paramid, type, patch_request):
        assert len(dates) == 1
        base = {"param": ["T"], "levelist": 500, "step": 999}  # 999: training-time artifact
        return [patch_request(base)]


class _FakeRetrievedField:
    def __init__(self, request):
        self.request = request

    def metadata(self, key, default=None):
        return default


def test_retrieve_requests_one_run_per_target_date_with_patched_step_and_params(monkeypatch):
    """retrieve() should ask _latest_published_run() (not guess a delay) for
    the starting point, call mars_requests once per target date patching
    `step` to the value _run_and_step computed for that date (not the
    checkpoint's training-time step baked into variables_metadata), and
    translate `param` from KENDA/COSMO names to ECMWF Open Data ones derived
    from variables_metadata, not a hardcoded table."""
    obj = _make_input()
    obj.kwargs = {}
    obj.metadata = _FakeMetadata()
    obj.patch_data_request = lambda r: r

    captured_requests = []

    def fake_retrieve_opendata(requests, *, patch, **kwargs):
        captured_requests.extend(requests)
        return [_FakeRetrievedField(r) for r in requests]

    monkeypatch.setattr(
        "anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata._retrieve_opendata",
        fake_retrieve_opendata,
    )
    monkeypatch.setattr(
        "anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata._latest_published_run",
        lambda **params: datetime(2026, 8, 5, 6, 0),
    )

    result = obj.retrieve(["t_500"], [datetime(2026, 8, 5, 6, 0), datetime(2026, 8, 4, 9, 0)])

    assert len(captured_requests) == 2
    assert captured_requests[0]["step"] == 0  # exact run boundary
    assert captured_requests[1]["step"] == 3  # walked back a day, 3h off that run
    assert captured_requests[0]["param"] == ["t"]  # translated from KENDA "T"
    assert [f.request for f in result] == captured_requests  # concatenated in order


def test_latest_published_run_calls_ecmwf_opendata_client(monkeypatch):
    """_latest_published_run() should delegate straight to
    ecmwf.opendata.Client().latest(), the real availability check, passing
    through whatever params it's given."""
    from anemoi_plugins_meteoswiss.inference.inputs import oper_ecmwf_opendata

    captured = {}

    class _FakeClient:
        def latest(self, **params):
            captured.update(params)
            return datetime(2026, 8, 5, 18, 0)

    monkeypatch.setattr(oper_ecmwf_opendata, "_EcmwfOpenDataClient", _FakeClient)

    result = oper_ecmwf_opendata._latest_published_run(type="fc")

    assert result == datetime(2026, 8, 5, 18, 0)
    assert captured == {"type": "fc"}


def test_without_eccodes_definition_path_override_clears_and_restores(monkeypatch):
    monkeypatch.setenv("ECCODES_DEFINITION_PATH", "/some/cosmo/definitions")

    with _without_eccodes_definition_path_override():
        assert "ECCODES_DEFINITION_PATH" not in os.environ

    assert os.environ["ECCODES_DEFINITION_PATH"] == "/some/cosmo/definitions"


def test_without_eccodes_definition_path_override_is_a_no_op_when_unset(monkeypatch):
    monkeypatch.delenv("ECCODES_DEFINITION_PATH", raising=False)

    with _without_eccodes_definition_path_override():
        assert "ECCODES_DEFINITION_PATH" not in os.environ

    assert "ECCODES_DEFINITION_PATH" not in os.environ
