"""Tests for the assign-grid and geopotential-from-height filters."""

import numpy as np
import pytest
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filters import create_filter_by_name
from grib_helpers import grib_field
from grib_helpers import time_metadata
from grib_helpers import write_icon_grid

from anemoi_plugins_meteoswiss.transform.filters.geopotential_from_height import G
from anemoi_plugins_meteoswiss.transform.filters.geopotential_from_height import GeopotentialFromHeight
from anemoi_plugins_meteoswiss.transform.filters.grid import AssignGrid

VALUES = np.array([0.0, 500.0, 1200.5, 3000.0, 4500.0])
LATITUDES = np.array([45.5, 46.0, 46.5, 47.0, 47.5])
LONGITUDES = np.array([6.0, 7.5, 8.0, 9.5, 10.0])


@pytest.fixture
def grid_definition(tmp_path):
    return {"icon": {"path": str(write_icon_grid(tmp_path / "grid.nc", LATITUDES, LONGITUDES))}}


def _field(param, values=VALUES):
    # the parameter name depends on the ecCodes definitions: set it explicitly, like a flavour would
    return new_field_with_metadata(grib_field(values, step_type="instant", start_step=0, end_step=0), param=param)


def test_assign_grid(grid_definition):
    field = grib_field(VALUES, step_type="accum", start_step=0, end_step=3)

    (result,) = AssignGrid(grid_definition).forward(new_fieldlist_from_list([field]))

    np.testing.assert_allclose(result.geography.latitudes(), LATITUDES)
    np.testing.assert_allclose(result.geography.longitudes(), LONGITUDES)
    np.testing.assert_array_equal(result.values, VALUES)
    for key in ("parameter.variable", "time.valid_datetime", "time.step", "vertical.level"):
        assert result.get(key) == field.get(key), key
    assert time_metadata(result) == time_metadata(field)


def test_assign_grid_entry_point(grid_definition):
    filter = create_filter_by_name("assign-grid", grid_definition=grid_definition)
    (result,) = filter.forward(new_fieldlist_from_list([grib_field(VALUES)]))
    np.testing.assert_allclose(result.geography.latitudes(), LATITUDES)


@pytest.mark.parametrize(("param", "expected"), [("HSURF", "FIS"), ("HHL", "FI"), ("h", "FI")])
def test_geopotential(param, expected):
    field = _field(param)

    (result,) = GeopotentialFromHeight().forward(new_fieldlist_from_list([field]))

    assert result.get("parameter.variable") == expected
    assert result.get("labels.shortName") == expected
    assert str(result.get("parameter.units")) == "meter ** 2 / second ** 2"
    np.testing.assert_allclose(result.values, VALUES * G)
    assert result.get("time.valid_datetime") == field.get("time.valid_datetime")


def test_geopotential_passes_other_fields_through():
    fields = [_field("T_2M", VALUES + 273.15), _field("HSURF")]

    result = GeopotentialFromHeight().forward(new_fieldlist_from_list(fields))

    assert [f.get("parameter.variable") for f in result] == ["T_2M", "FIS"]
    assert result[0] is fields[0]


def test_geopotential_after_assign_grid(grid_definition):
    """The grid masks the raw GRIB keys (shortName): the filter must use the parameter component."""
    gridded = AssignGrid(grid_definition).forward(new_fieldlist_from_list([_field("HSURF")]))
    assert gridded[0].get("metadata.shortName") is None

    (result,) = create_filter_by_name("geopotential-from-height").forward(gridded)

    assert result.get("parameter.variable") == "FIS"
    np.testing.assert_allclose(result.values, VALUES * G)
    np.testing.assert_allclose(result.geography.latitudes(), LATITUDES)
