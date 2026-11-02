import eccodes
import numpy as np
import pytest
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filters import filter_registry
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField

from anemoi_plugins_meteoswiss.transform.filters import FillMissing


def _field(name, values):
    return ArrayField(
        np.array(values, dtype=float), RawMetadata({"shortName": name, "param": name})
    )


def _fieldlist():
    return new_fieldlist_from_list(
        [
            _field("HZEROCL", [np.nan, 1500.0, np.nan, 2000.0]),
            _field("T_2M", [np.nan, 270.0, 271.0, 272.0]),
        ]
    )


def _grib_file(path, values):
    """Write a 2x2 regular lat/lon GRIB file and return the short name eccodes gives it."""
    gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    eccodes.codes_set(gid, "Ni", 2)
    eccodes.codes_set(gid, "Nj", 2)
    eccodes.codes_set_values(gid, np.asarray(values, dtype=float))
    short_name = eccodes.codes_get(gid, "shortName")
    with open(path, "wb") as f:
        eccodes.codes_write(gid, f)
    eccodes.codes_release(gid)
    return short_name


def test_fill_missing_value():
    out = FillMissing(param="HZEROCL", value=-1.0).forward(_fieldlist())
    np.testing.assert_array_equal(out[0].values, [-1.0, 1500.0, -1.0, 2000.0])
    assert out[0].metadata("shortName") == "HZEROCL"
    # other parameters pass through unchanged, NaNs included
    np.testing.assert_array_equal(out[1].values, [np.nan, 270.0, 271.0, 272.0])


def test_fill_missing_from_grib(tmp_path):
    path = tmp_path / "const.grib"
    name = _grib_file(path, [10.0, 20.0, 30.0, 40.0])

    filter = filter_registry.create(
        "fill-missing", param=["HZEROCL"], path=str(path), fill_param=name
    )
    out = filter.forward(_fieldlist())
    np.testing.assert_array_equal(out[0].values, [10.0, 1500.0, 30.0, 2000.0])


def test_fill_missing_errors(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        FillMissing(param="HZEROCL")
    with pytest.raises(ValueError, match="exactly one"):
        FillMissing(param="HZEROCL", value=0.0, path="x.grib")

    path = tmp_path / "const.grib"
    name = _grib_file(path, [10.0, 20.0, 30.0, 40.0])
    with pytest.raises(ValueError, match="Expected one"):
        FillMissing(param="HZEROCL", path=str(path), fill_param="NOPE").forward(
            _fieldlist()
        )

    short = new_fieldlist_from_list([_field("HZEROCL", [np.nan, 1.0])])
    with pytest.raises(ValueError, match="shape"):
        FillMissing(param="HZEROCL", path=str(path), fill_param=name).forward(short)
