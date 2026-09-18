import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filters import filter_registry
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField
from meteodatalab import data_source
from meteodatalab import grib_decoder
from numpy.testing import assert_array_equal

from anemoi_plugins_meteoswiss.helpers import from_meteodatalab
from anemoi_plugins_meteoswiss.transform.filters import ClipLateralBoundaries
from anemoi_plugins_meteoswiss.transform.filters import Destagger


def test_clip_lateral_boundaries(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    from meteodatalab.operators.clip import clip_lateral_boundary_strip

    fn = str(data_dir / "iaf2025010100")
    gridfile_fn = "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/icon_grid_0001_R19B08_mch.nc"
    strip_idx = 14

    filter: ClipLateralBoundaries = filter_registry.create(
        "clip_lateral_boundaries", strip_idx, gridfile_fn
    )

    # expected
    source = data_source.FileDataSource(datafiles=[fn])
    ds = grib_decoder.load(source, {"param": ["T_2M"]})
    ds["T_2M"] = clip_lateral_boundary_strip(ds["T_2M"], strip_idx)

    # actual
    fieldlist = ekd.from_source("file", fn)
    res = filter.forward(fieldlist)

    assert_array_equal(ds["T_2M"].values.ravel(), res[0].values)


def test_destagger_matches_meteodatalab(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")
    from meteodatalab.operators.destagger import destagger

    # test vertical destaggering
    fn = str(data_dir / "iaf2025010100")
    param_dim = {"W": "z"}

    filter: Destagger = filter_registry.create("destagger", param=["W"])

    source = data_source.FileDataSource(datafiles=[fn])
    ds = grib_decoder.load(source, {"param": ["W"]})
    ds_desired = {k: destagger(v, param_dim[k]) for k, v in ds.items()}
    desired = from_meteodatalab(ds_desired)

    fieldlist = ekd.from_source("file", fn).sel(param="W")
    actual = filter.forward(fieldlist)
    print(actual.values)
    np.testing.assert_array_equal(actual.values, desired.values)
    np.testing.assert_array_equal(actual.values, ds_desired["W"].values.squeeze())


def _half_level_fields(name, levels, valid_datetime="2020-01-01T00:00:00"):
    return [
        ArrayField(
            np.full(4, float(k)),
            RawMetadata(
                {
                    "shortName": name,
                    "param": name,
                    "level": k,
                    "valid_datetime": valid_datetime,
                }
            ),
        )
        for k in levels
    ]


def test_destagger():
    fieldlist = new_fieldlist_from_list(
        _half_level_fields("W", [1, 2, 3, 4, 5])
        + _half_level_fields("W", [1, 2], valid_datetime="2020-01-01T01:00:00")
        + _half_level_fields("T", [1, 2])
    )

    out = Destagger(param=["W"]).forward(fieldlist)

    w = [f for f in out if f.metadata("shortName") == "W"]
    assert [(f.metadata("valid_datetime"), f.metadata("level")) for f in w] == [
        ("2020-01-01T00:00:00", 1),
        ("2020-01-01T00:00:00", 2),
        ("2020-01-01T00:00:00", 3),
        ("2020-01-01T00:00:00", 4),
        ("2020-01-01T01:00:00", 1),
    ]
    np.testing.assert_array_equal(w[0].values, np.full(4, 1.5))
    np.testing.assert_array_equal(w[3].values, np.full(4, 4.5))
    assert all(f.metadata("typeOfLevel") == "generalVerticalLayer" for f in w)
    # other parameters pass through unchanged
    assert [f.metadata("level") for f in out if f.metadata("shortName") == "T"] == [
        1,
        2,
    ]


def test_destagger_selected_levels():
    fieldlist = new_fieldlist_from_list(_half_level_fields("W", [1, 2, 3, 4, 5]))

    out = Destagger(param="W", levels=[1, 4]).forward(fieldlist)
    assert [f.metadata("level") for f in out] == [1, 4]
    np.testing.assert_array_equal(out[1].values, np.full(4, 4.5))

    with pytest.raises(ValueError, match="half levels k and k\\+1"):
        Destagger(param="W", levels=[5]).forward(fieldlist)
