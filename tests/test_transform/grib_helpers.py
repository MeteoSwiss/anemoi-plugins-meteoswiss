"""Small synthetic GRIB fields for unit tests (no FDB, no data files)."""

from pathlib import Path

import earthkit.data as ekd
import eccodes
import numpy as np
from earthkit.data import from_source  # bound here: tests may monkeypatch ekd.from_source


def grib_field(
    values,
    *,
    date: int = 20200129,
    time: int = 0,
    start_step: int = 0,
    end_step: int = 1,
    step_units: str = "h",
    step_type: str = "avg",
    type_of_level: str = "surface",
    level: int = 0,
) -> ekd.Field:
    """Create a GRIB2 field on a 1 x len(values) regular lat-lon grid from the ecCodes GRIB2 sample.

    The parameter is whatever the sample (and the ecCodes definitions in use) make of it:
    tests must not rely on it.
    """
    values = np.asarray(values, dtype=float)
    handle = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        keys = {
            "Ni": len(values),
            "Nj": 1,
            "latitudeOfFirstGridPointInDegrees": 47.0,
            "latitudeOfLastGridPointInDegrees": 47.0,
            "longitudeOfFirstGridPointInDegrees": 7.0,
            "longitudeOfLastGridPointInDegrees": 7.0 + len(values) - 1,
            "iDirectionIncrementInDegrees": 1.0,
            "jDirectionIncrementInDegrees": 1.0,
            "dataDate": date,
            "dataTime": time,
            "typeOfLevel": type_of_level,
            "level": level,
        }
        for key, value in keys.items():
            eccodes.codes_set(handle, key, value)
        if step_type != "instant":
            eccodes.codes_set(handle, "productDefinitionTemplateNumber", 8)
        eccodes.codes_set(handle, "stepUnits", step_units)
        eccodes.codes_set(handle, "stepType", step_type)
        eccodes.codes_set(handle, "startStep", start_step)
        eccodes.codes_set(handle, "endStep", end_step)
        eccodes.codes_set(handle, "bitsPerValue", 24)
        eccodes.codes_set_values(handle, values)
        message = eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)
    return from_source("memory", message).to_fieldlist()[0]


def time_metadata(field: ekd.Field) -> dict:
    """The raw GRIB time keys of a field (also when the raw keys are masked by a filter or grid)."""
    return field.get(collections="metadata.time")


def write_icon_grid(path: Path, latitudes, longitudes) -> Path:
    """Write a minimal ICON grid file (cell centres in radians)."""
    import xarray as xr

    ds = xr.Dataset(
        {
            "clat": ("cell", np.deg2rad(np.asarray(latitudes, dtype=float))),
            "clon": ("cell", np.deg2rad(np.asarray(longitudes, dtype=float))),
        }
    )
    ds.to_netcdf(path)
    return path
