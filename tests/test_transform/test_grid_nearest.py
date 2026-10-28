"""Unit tests for the grid-nearest filter (no DWH / no ICON grid file needed)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from anemoi.datasets.create.sources.xarray_support import XarrayFieldList

from anemoi_plugins_meteoswiss.transform.filters import GridNearest
from anemoi_plugins_meteoswiss.transform.filters import grid_nearest

# A tiny 4-cell target grid (two latitudes x two longitudes), used in place of the
# real ICON-CH1 grid so these tests run anywhere.
GRID_LAT = np.array([46.0, 46.0, 48.0, 48.0])
GRID_LON = np.array([7.0, 9.0, 7.0, 9.0])


@pytest.fixture
def patched_grid(monkeypatch):
    monkeypatch.setattr(
        grid_nearest,
        "cropped_icon_grid",
        lambda grid, crop: (GRID_LAT, GRID_LON, np.ones(GRID_LAT.size, bool)),
    )


def _stations(lats, lons, values):
    """Build a one-param, one-time station FieldList like the synop-dwh source."""
    ds = xr.Dataset(
        data_vars={"t": (("time", "station"), np.asarray(values, dtype=np.float32)[None, :])},
        coords={
            "time": ("time", pd.DatetimeIndex(["2024-06-15T00:00"]).values),
            "station": ("station", np.array([f"S{i}" for i in range(len(lats))], dtype=object)),
            "latitude": ("station", np.asarray(lats, dtype=float)),
            "longitude": ("station", np.asarray(lons, dtype=float)),
        },
    )
    ds["latitude"].attrs.update(units="degrees_north", standard_name="latitude")
    ds["longitude"].attrs.update(units="degrees_east", standard_name="longitude")
    ds["station"].attrs.update(standard_name="station")
    return XarrayFieldList.from_xarray(ds)


def test_scatters_to_nearest_cell_rest_nan(patched_grid):
    # Two stations just off cell 0 (46,7) and cell 3 (48,9).
    fl = _stations([46.01, 47.99], [7.01, 8.99], [10.0, 20.0])
    out = GridNearest(grid={}, max_distance_km=50.0).forward(fl)

    assert len(out) == 1
    arr = out[0].to_numpy()
    assert arr.size == 4
    np.testing.assert_array_equal(np.isfinite(arr), [True, False, False, True])
    assert arr[0] == pytest.approx(10.0)
    assert arr[3] == pytest.approx(20.0)
    # output rides on the target grid
    glat, glon = out[0].grid_points()
    np.testing.assert_array_equal(glat, GRID_LAT)
    np.testing.assert_array_equal(glon, GRID_LON)


def test_max_distance_drops_far_station(patched_grid):
    # Station far from every cell -> dropped, whole grid stays NaN.
    fl = _stations([0.0], [0.0], [42.0])
    out = GridNearest(grid={}, max_distance_km=5.0).forward(fl)
    assert np.isnan(out[0].to_numpy()).all()


def test_nan_observation_leaves_cell_nan(patched_grid):
    # A missing observation must not populate its cell.
    fl = _stations([46.01], [7.01], [np.nan])
    out = GridNearest(grid={}, max_distance_km=50.0).forward(fl)
    assert np.isnan(out[0].to_numpy()).all()
