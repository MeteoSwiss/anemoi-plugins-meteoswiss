import earthkit.data as ekd
import numpy as np
import pytest


def test_earthkit_meteodatalab_roundtrip(data_dir, hostname):
    """Test conversion to and from meteodatalab."""
    from anemoi_plugins_meteoswiss.helpers import from_meteodatalab
    from anemoi_plugins_meteoswiss.helpers import to_meteodatalab

    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")
    fl_original = ekd.from_source("file", data_dir / "iaf2025010100")
    ds = to_meteodatalab(fl_original)
    fl_rountrip = from_meteodatalab(ds)
    np.testing.assert_array_equal(fl_original.values, fl_rountrip.values)


def test_meteodatalab_earthkit_roundtrip(data_dir, hostname):
    """Test conversion to and from meteodatalab."""
    from meteodatalab import data_source
    from meteodatalab import grib_decoder

    from anemoi_plugins_meteoswiss.helpers import from_meteodatalab
    from anemoi_plugins_meteoswiss.helpers import to_meteodatalab

    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")
    fds = data_source.FileDataSource(datafiles=[str(data_dir / "iaf2025010100")])
    ds_original = grib_decoder.load(fds, {"param": ["T"]})
    fl = from_meteodatalab(ds_original)
    ds_roundtrip = to_meteodatalab(fl)
    np.testing.assert_array_equal(ds_original["T"].values, ds_roundtrip["T"].values)


def test_earthkit_meteodatalab_oneway(data_dir, hostname):
    """Test conversion to and from meteodatalab."""
    from anemoi_plugins_meteoswiss.helpers import to_meteodatalab

    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    fl = ekd.from_source("file", data_dir / "iaf2025010100")
    ds = to_meteodatalab(fl)
    np.testing.assert_array_equal(
        ds["T"].values.squeeze(), fl.sel(param="T").values.squeeze()
    )


def test_meteodatalab_earthkit_oneway(data_dir, hostname):
    """Test conversion to and from meteodatalab."""
    from meteodatalab import data_source
    from meteodatalab import grib_decoder

    from anemoi_plugins_meteoswiss.helpers import from_meteodatalab

    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")
    fds = data_source.FileDataSource(datafiles=[str(data_dir / "iaf2025010100")])
    ds = grib_decoder.load(fds, {"param": ["T"]})
    fl = from_meteodatalab(ds)
    np.testing.assert_array_equal(fl.values.squeeze(), ds["T"].values.squeeze())
