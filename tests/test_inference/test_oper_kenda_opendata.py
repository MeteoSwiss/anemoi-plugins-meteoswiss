"""Minimal integration test for OperKendaOpenDataInput -- hits the real MeteoSwiss STAC open-data API.

Skipped by default: set ``RUN_INTEGRATION_TESTS=1`` to run it. Downloads the full KENDA-CH1 grid
constants collection (a few hundred MB) plus one hourly analysis field, since
``OperKendaOpenDataInput.retrieve()`` always needs the constants collection to know which
requested variables are constants (see ``_available_constant_names``).
"""

import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

pytest.importorskip("anemoi.inference.inputs.mars")

from anemoi.inference.metadata import Metadata

from anemoi_plugins_meteoswiss.inference.inputs.oper_kenda_opendata import OperKendaOpenDataInput

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION_TESTS") != "1",
        reason="hits the real MeteoSwiss STAC API and downloads ~200MB of grid constants -- "
        "set RUN_INTEGRATION_TESTS=1 to run",
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
                "data_request": {},
            }
        }
    )


def test_retrieve_resolves_each_variable_from_exactly_one_source():
    """``t2m`` only exists in the hourly STAC items; ``hsurf`` only exists in the static grid
    constants -- checks both sources are actually reachable and correctly routed to."""
    valid_time = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) - timedelta(hours=3)
    metadata = _fake_metadata({"t2m": {"param": "T_2M"}, "hsurf": {"param": "HSURF"}})
    input_ = OperKendaOpenDataInput(_FakeContext(valid_time), metadata, variables=[])

    result = input_.retrieve(variables=["t2m", "hsurf"], dates=[valid_time])

    assert {field.metadata("shortName") for field in result} == {"T_2M", "HSURF"}
