"""Tests for the average-flux-to-cumulative-quantity filter."""

import datetime

import numpy as np
import pytest
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filters import create_filter_by_name
from grib_helpers import grib_field
from grib_helpers import time_metadata

from anemoi_plugins_meteoswiss.transform.filters.time_processing import AverageFluxToCumulativeQuantity
from anemoi_plugins_meteoswiss.transform.filters.time_processing import CumulativeQuantityToAverageFlux
from anemoi_plugins_meteoswiss.transform.filters.time_processing import _step_seconds

VALUES = np.array([0.0, 1.5, 100.0, 250.25, 1000.0])


@pytest.mark.parametrize(
    ("step", "unit", "seconds"),
    [(2, 1, 7200), ("2", 1, 7200), (10, 0, 600), ("710m", 0, 42600), ("710m", 1, 42600), ("1h", 0, 3600)],
)
def test_step_seconds(step, unit, seconds):
    assert _step_seconds(step, unit) == seconds


def test_step_seconds_unsupported_unit():
    with pytest.raises(ValueError, match="Unsupported step unit"):
        _step_seconds(1, 2)


@pytest.mark.parametrize(
    ("kwargs", "seconds"),
    [
        (dict(start_step=2, end_step=3, step_units="h"), 3600),
        (dict(start_step=0, end_step=6, step_units="h"), 6 * 3600),
        (dict(start_step=710, end_step=720, step_units="m"), 600),
    ],
)
def test_forward(kwargs, seconds):
    field = grib_field(VALUES, step_type="avg", **kwargs)
    before = time_metadata(field)

    (result,) = AverageFluxToCumulativeQuantity().forward(new_fieldlist_from_list([field]))

    np.testing.assert_array_equal(result.values, VALUES * seconds)
    after = time_metadata(result)
    assert after["stepType"] == "accum"
    assert result.metadata("stepType") == "accum"
    for key in ("startStep", "endStep", "stepUnits", "dataDate", "dataTime", "validityDate", "validityTime"):
        assert after[key] == before[key], key
    assert result.get("time.valid_datetime") == field.get("time.valid_datetime")
    assert result.get("time.step") == field.get("time.step")


def test_backward_roundtrip():
    field = grib_field(VALUES, step_type="avg", start_step=2, end_step=3)
    accumulated = AverageFluxToCumulativeQuantity().forward(new_fieldlist_from_list([field]))
    (result,) = CumulativeQuantityToAverageFlux().forward(accumulated)

    np.testing.assert_allclose(result.values, VALUES)
    assert result.metadata("stepType") == "avg"
    assert time_metadata(result)["stepRange"] == "2-3"


def test_masked_raw_keys():
    """Raw GRIB keys are masked once a component is set (e.g. by a flavour): use the time collection."""
    field = new_field_with_metadata(grib_field(VALUES, step_type="avg", start_step=2, end_step=3), levelist=None)
    assert field.get("metadata.stepType") is None

    (result,) = AverageFluxToCumulativeQuantity().forward(new_fieldlist_from_list([field]))

    np.testing.assert_array_equal(result.values, VALUES * 3600)
    assert time_metadata(result)["stepType"] == "accum"


@pytest.mark.parametrize("step_type", ["accum", "instant", "max"])
def test_forward_rejects_non_average(step_type):
    field = grib_field(VALUES, step_type=step_type, start_step=0 if step_type == "instant" else 2, end_step=3)
    with pytest.raises(ValueError, match="Expected a time-averaged field"):
        AverageFluxToCumulativeQuantity().forward(new_fieldlist_from_list([field]))


def test_backward_rejects_non_cumulative():
    field = grib_field(VALUES, step_type="avg", start_step=2, end_step=3)
    with pytest.raises(ValueError, match="Expected a cumulative field"):
        CumulativeQuantityToAverageFlux().forward(new_fieldlist_from_list([field]))


def test_entry_point():
    filter = create_filter_by_name("average-flux-to-cumulative-quantity")
    field = grib_field(VALUES, step_type="avg", start_step=0, end_step=1)
    (result,) = filter.forward(new_fieldlist_from_list([field]))
    np.testing.assert_array_equal(result.values, VALUES * 3600)
    assert result.get("time.valid_datetime") == datetime.datetime(2020, 1, 29, 1)
