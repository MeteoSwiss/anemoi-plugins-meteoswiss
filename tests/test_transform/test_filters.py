import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.filters import filter_registry
from numpy.testing import assert_array_equal

from anemoi_plugins_meteoswiss.transform.filters import ClipLateralBoundaries
from anemoi_plugins_meteoswiss.transform.filters import Destagger


def test_clip_lateral_boundaries(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    from meteodatalab import data_source
    from meteodatalab import grib_decoder
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


def test_destagger(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    from meteodatalab import data_source
    from meteodatalab import grib_decoder
    from meteodatalab.operators.destagger import destagger

    from anemoi_plugins_meteoswiss.helpers import from_meteodatalab

    # test vertical destaggering
    fn = str(data_dir / "iaf2025010100")
    param_dim = {"W": "z"}

    filter: Destagger = filter_registry.create("destagger", param_dim)

    source = data_source.FileDataSource(datafiles=[fn])
    ds = grib_decoder.load(source, {"param": ["W"]})
    ds_desired = {k: destagger(v, param_dim[k]) for k, v in ds.items()}
    desired = from_meteodatalab(ds_desired)

    fieldlist = ekd.from_source("file", fn).sel(param="W")
    actual = filter.forward(fieldlist)
    print(actual.values)
    np.testing.assert_array_equal(actual.values, desired.values)
    np.testing.assert_array_equal(actual.values, ds_desired["W"].values.squeeze())
