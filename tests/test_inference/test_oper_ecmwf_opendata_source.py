"""Unit tests for forwarding the ``source`` mirror option from OperEcmwfOpenDataInput to the download call.

No network access: the download and run-availability lookups are mocked.
"""

from datetime import datetime
from unittest import mock

import pytest

pytest.importorskip("anemoi.plugins.ecmwf.inference.opendata.opendata")

import earthkit.data as ekd
from anemoi.inference.metadata import Metadata

from anemoi_plugins_meteoswiss.inference.inputs import oper_ecmwf_opendata
from anemoi_plugins_meteoswiss.inference.inputs.oper_ecmwf_opendata import OperEcmwfOpenDataInput

RUN = datetime(2026, 1, 1, 0)


class _FakeContext:
    """Just enough of ``anemoi.inference.context.Context`` for the input to run standalone."""

    use_grib_paramid = False
    reference_date = RUN

    def patch_data_request(self, request, dataset_name):
        return request


def _fake_metadata() -> Metadata:
    return Metadata(
        {
            "dataset": {
                "variables_metadata": {"2t": {"mars": {"levtype": "sfc", "param": "T_2M", "stream": "oper"}}},
                "variables": ["2t"],
                "data_request": {"grid": "N320", "area": [90.0, 0.0, -90.0, 359.719]},
            }
        }
    )


def _retrieve_kwargs(**input_kwargs) -> dict:
    """Build the input, run ``retrieve()`` with the download mocked, and return the kwargs it was called with."""
    with mock.patch("anemoi.plugins.ecmwf.inference.opendata.opendata.OrographyProcessor"):
        input_ = OperEcmwfOpenDataInput(_FakeContext(), _fake_metadata(), variables=[], **input_kwargs)

    with (
        mock.patch.object(oper_ecmwf_opendata, "_latest_published_run", return_value=RUN),
        mock.patch.object(oper_ecmwf_opendata, "_retrieve_opendata", return_value=ekd.SimpleFieldList([])) as spy,
    ):
        input_.retrieve(variables=["2t"], dates=[RUN])

    spy.assert_called_once()
    return spy.call_args.kwargs


def test_retrieve_forwards_configured_source():
    assert _retrieve_kwargs(source="azure")["source"] == "azure"


def test_retrieve_defaults_source_to_ecmwf():
    assert _retrieve_kwargs()["source"] == "ecmwf"
