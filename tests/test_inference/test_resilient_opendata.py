"""Unit tests for ResilientOpenDataInput's run/step walk-back logic.

These need no network/FDB/HPC access — they exercise the pure date
arithmetic and the request-building wiring with mocks, unlike the
`hostname.startswith("balfrin")`-gated tests elsewhere in this suite.
"""

from datetime import datetime

import pytest

from anemoi_plugins_meteoswiss.inference.inputs.resilient_opendata import ResilientOpenDataInput
from anemoi_plugins_meteoswiss.inference.inputs.resilient_opendata import _param_translation_from_variables_metadata
from anemoi_plugins_meteoswiss.inference.inputs.resilient_opendata import _translate_params

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
    """A ResilientOpenDataInput with its attributes set directly, bypassing
    __init__ (which needs a real anemoi-inference Context/Metadata/checkpoint)."""
    obj = object.__new__(ResilientOpenDataInput)
    obj.frequency_h = overrides.get("frequency_h", 6)
    obj.step_h = overrides.get("step_h", 3)
    obj.max_lead_time_h = overrides.get("max_lead_time_h", 144)
    obj.stored_runs = overrides.get("stored_runs", 12)
    obj.delivery_delay_h = overrides.get("delivery_delay_h", 7)
    return obj


def test_guaranteed_run_rounds_down_to_frequency_boundary():
    d = _make_input()
    # 15:30 - 7h delay = 08:30, rounds down to the 6h boundary -> 06:00
    assert d._guaranteed_run(datetime(2026, 8, 5, 15, 30)) == datetime(2026, 8, 5, 6, 0)


def test_target_exactly_at_a_run_boundary_uses_step_zero():
    d = _make_input()
    guaranteed = d._guaranteed_run(datetime(2026, 8, 5, 15, 30))
    run, step = d._run_and_step(datetime(2026, 8, 5, 6, 0), guaranteed)
    assert (run, step) == (datetime(2026, 8, 5, 6, 0), 0)


def test_target_between_runs_uses_nearest_step_boundary():
    d = _make_input()
    guaranteed = d._guaranteed_run(datetime(2026, 8, 5, 15, 30))
    run, step = d._run_and_step(datetime(2026, 8, 5, 9, 0), guaranteed)
    assert (run, step) == (datetime(2026, 8, 5, 6, 0), 3)


def test_target_older_than_guaranteed_run_walks_back():
    d = _make_input()
    guaranteed = d._guaranteed_run(datetime(2026, 8, 5, 15, 30))  # 2026-08-05 06:00
    run, step = d._run_and_step(datetime(2026, 8, 4, 9, 0), guaranteed)
    assert (run, step) == (datetime(2026, 8, 4, 6, 0), 3)


def test_target_too_old_raises():
    d = _make_input()
    guaranteed = d._guaranteed_run(datetime(2026, 8, 5, 15, 30))
    with pytest.raises(ValueError, match="stored open data runs"):
        d._run_and_step(datetime(2026, 7, 1, 0, 0), guaranteed)


def test_step_exceeding_max_lead_time_raises():
    d = _make_input(frequency_h=12, max_lead_time_h=6, stored_runs=100)
    guaranteed = d._guaranteed_run(datetime(2026, 8, 5, 15, 30))  # 2026-08-05 00:00
    with pytest.raises(ValueError, match="exceeds open data max lead time"):
        d._run_and_step(datetime(2026, 8, 5, 9, 0), guaranteed)  # 9h ahead of the 00:00 run


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


def test_retrieve_requests_one_run_per_target_date_with_patched_step_and_params(monkeypatch):
    """retrieve() should call mars_requests once per target date, patching
    `step` to the value _run_and_step computed for that date (not the
    checkpoint's training-time step baked into variables_metadata) and
    translating `param` from KENDA/COSMO names to ECMWF Open Data ones
    derived from variables_metadata, not a hardcoded table."""
    obj = _make_input()
    obj.kwargs = {}
    obj.metadata = _FakeMetadata()
    obj.patch_data_request = lambda r: r

    captured_requests = []

    def fake_retrieve_opendata(requests, *, patch, **kwargs):
        captured_requests.extend(requests)
        return requests  # stand-in FieldList: a plain list supports + for this test

    monkeypatch.setattr(
        "anemoi_plugins_meteoswiss.inference.inputs.resilient_opendata._retrieve_opendata",
        fake_retrieve_opendata,
    )

    guaranteed = obj._guaranteed_run(datetime(2026, 8, 5, 15, 30))
    monkeypatch.setattr(obj, "_guaranteed_run", lambda: guaranteed)

    result = obj.retrieve(["t_500"], [datetime(2026, 8, 5, 6, 0), datetime(2026, 8, 4, 9, 0)])

    assert len(captured_requests) == 2
    assert captured_requests[0]["step"] == 0  # exact run boundary
    assert captured_requests[1]["step"] == 3  # walked back a day, 3h off that run
    assert captured_requests[0]["param"] == ["t"]  # translated from KENDA "T"
    assert result == captured_requests  # concatenated in order
