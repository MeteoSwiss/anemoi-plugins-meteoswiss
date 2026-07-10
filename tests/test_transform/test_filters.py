import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.filters import filter_registry
from meteodatalab import data_source
from meteodatalab import grib_decoder

from anemoi_plugins_meteoswiss.transform.filters import Destagger
from anemoi_plugins_meteoswiss.transform.filters.destaggering import from_meteodatalab


def test_destagger(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")
    from meteodatalab.operators.destagger import destagger

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
