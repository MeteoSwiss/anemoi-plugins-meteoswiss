"""Minimal integration test for OperEcmwfOpenDataInput -- hits the real ECMWF Open Data service.

Skipped by default: set ``RUN_INTEGRATION_TESTS=1`` to run it. It has no control over the
service's current state, so it targets whatever run ``Client.latest()`` reports right now rather
than a fixed date.
"""

import os

import pytest

pytest.importorskip("anemoi.plugins.ecmwf.inference.opendata.opendata")

from anemoi.inference.metadata import Metadata

from anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata import OperEcmwfOpenDataInput
from anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata import _latest_published_run

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION_TESTS") != "1",
        reason="hits the real ECMWF Open Data service -- set RUN_INTEGRATION_TESTS=1 to run",
    ),
]


class _FakeContext:
    """Just enough of ``anemoi.inference.context.Context`` for the input to run standalone."""

    use_grib_paramid = False

    def __init__(self, reference_date):
        self.reference_date = reference_date

    def patch_data_request(self, request, dataset_name):
        return request


def _fake_metadata(variables_to_mars: dict) -> Metadata:
    """A real ``Metadata`` wrapping just enough fabricated checkpoint metadata to exercise
    ``mars_requests()``, without needing an actual checkpoint file."""
    return Metadata(
        {
            "dataset": {
                "variables_metadata": {
                    variable: {"mars": mars}
                    for variable, mars in variables_to_mars.items()
                },
                "variables": list(variables_to_mars),
                "data_request": {"grid": "N320", "area": [90.0, 0.0, -90.0, 359.719]},
            }
        }
    )


def test_retrieve_fetches_real_fields_from_ecmwf_open_data():
    """Regression check for the ``levelist: None`` -> ``"None"`` index-matching bug (see
    ``_drop_null_levelist``): ``z`` here carries an explicit ``levelist: None`` in its MARS
    metadata, exactly like the surface orography field that used to crash this input.

    ``param`` uses real COSMO/KENDA names (``T_2M``, ``FIS``), matching what a real checkpoint's
    ``mars`` metadata actually carries (see ``forecaster.yaml``'s namer): eccodes is COSMO-locked
    process-wide (see ``anemoi_plugins_meteoswiss._use_cosmo_grib_definitions``), so the fields
    ECMWF Open Data returns decode with these COSMO ``shortName``s, not the ECMWF ones. The
    ``param`` metadata key is what ``_cosmo_to_ecmwf_field_param`` actually guarantees is correct
    (translated back via ``_param_translation_from_variables_metadata``) -- ``shortName`` itself
    stays COSMO-flavored and is not something this input promises to fix.
    """
    reference_date = _latest_published_run(type="fc")
    metadata = _fake_metadata(
        {
            "2t": {"levtype": "sfc", "param": "T_2M", "stream": "oper"},
            "z": {"levtype": "sfc", "param": "FIS", "stream": "oper", "levelist": None},
        }
    )
    input_ = OperEcmwfOpenDataInput(
        _FakeContext(reference_date), metadata, variables=[]
    )

    result = input_.retrieve(variables=["2t", "z"], dates=[reference_date])

    assert {field.metadata("param") for field in result} == {"2t", "z"}
