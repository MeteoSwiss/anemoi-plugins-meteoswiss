import earthkit.data as ekd
import numpy as np
from meteodatalab import data_source
from meteodatalab import grib_decoder

from anemoi_plugins_meteoswiss.helpers import from_meteodatalab
from anemoi_plugins_meteoswiss.helpers import to_meteodatalab


def test_earthkit_meteodatalab_roundtrip(data_dir):
    """Test conversion to and from meteodatalab."""
    fl_original = ekd.from_source("file", data_dir / "kenda-ch1-w-ml.grib")
    ds = to_meteodatalab(fl_original)
    fl_rountrip = from_meteodatalab(ds)
    np.testing.assert_array_equal(fl_original.values, fl_rountrip.values)


def test_meteodatalab_earthkit_roundtrip(data_dir):
    """Test conversion to and from meteodatalab."""
    fds = data_source.FileDataSource(datafiles=[str(data_dir / "kenda-ch1-w-ml.grib")])
    ds_original = grib_decoder.load(fds, {"param": ["W"]})
    fl = from_meteodatalab(ds_original)
    ds_roundtrip = to_meteodatalab(fl)
    np.testing.assert_array_equal(ds_original["W"].values, ds_roundtrip["W"].values)


def test_earthkit_meteodatalab_oneway(data_dir):
    """Test conversion to and from meteodatalab."""
    fl = ekd.from_source("file", data_dir / "kenda-ch1-w-ml.grib")
    ds = to_meteodatalab(fl)
    np.testing.assert_array_equal(ds["W"].values.squeeze(), fl.values.squeeze())


def test_meteodatalab_earthkit_oneway(data_dir):
    """Test conversion to and from meteodatalab."""
    fds = data_source.FileDataSource(datafiles=[str(data_dir / "kenda-ch1-w-ml.grib")])
    ds = grib_decoder.load(fds, {"param": ["W"]})
    fl = from_meteodatalab(ds)
    np.testing.assert_array_equal(fl.values.squeeze(), ds["W"].values.squeeze())
