import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filters import filter_registry
from meteodatalab import data_source
from meteodatalab import grib_decoder
from numpy.testing import assert_array_equal

from anemoi_plugins_meteoswiss.helpers import from_meteodatalab
from anemoi_plugins_meteoswiss.transform.filters import ClipLateralBoundaries
from anemoi_plugins_meteoswiss.transform.filters import Destagger
from anemoi_plugins_meteoswiss.transform.filters import GaussianSmoother
from anemoi_plugins_meteoswiss.transform.filters import IconRemapToRegLatLon

ICONREMAP_WEIGHTS = "/store_new/mch/msopr/icon_workflow_2/iconremap-weights/icon-ch1-eps-rotlatlon.nc"

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


def test_icon_remap_to_reg_lat_lon(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    regridder = IconRemapToRegLatLon(ICONREMAP_WEIGHTS)

    assert regridder.ny == 786
    assert regridder.nx == 1170

    # Geographic lat/lon for ICON-CH1 domain (Switzerland + surroundings)
    assert regridder._latitudes.min() > 40.0
    assert regridder._latitudes.max() < 52.0
    assert regridder._longitudes.min() > 0.0
    assert regridder._longitudes.max() < 22.0

    fn = str(data_dir / "iaf2025010100")
    fieldlist = ekd.from_source("file", fn).sel(shortName="T_2M")

    result = regridder.forward(fieldlist)

    n_out = regridder.ny * regridder.nx
    assert len(result) == len(fieldlist)
    for field in result:
        values = field.to_numpy(flatten=True)
        assert values.shape == (n_out,)
        # All output points are valid for this weights file (no sentinel zeros)
        assert not np.any(np.isnan(values))
        # Physical temperature range in Kelvin
        assert values.min() > 200.0
        assert values.max() < 340.0


def test_gaussian_smoother(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    regridder = IconRemapToRegLatLon(ICONREMAP_WEIGHTS)
    smoother = GaussianSmoother(sigma=5, nx=regridder.nx, ny=regridder.ny, params=["T_2M"])

    fn = str(data_dir / "iaf2025010100")
    fs = ekd.from_source("file", fn)
    # T_2M (smoothed) + one level of W (pass-through, not in params)
    fieldlist = new_fieldlist_from_list(
        list(fs.sel(shortName="T_2M")) + [fs.sel(shortName="W")[0]]
    )

    regridded = regridder.forward(fieldlist)
    smoothed = smoother.forward(regridded)

    assert len(smoothed) == len(regridded) == 2

    t2m_raw = regridded[0].to_numpy(flatten=True)
    t2m_smo = smoothed[0].to_numpy(flatten=True)
    w_raw = regridded[1].to_numpy(flatten=True)
    w_smo = smoothed[1].to_numpy(flatten=True)

    # Smoothed T_2M must differ from raw (sigma=5 is a meaningful kernel)
    assert not np.allclose(t2m_raw, t2m_smo)
    # Smoothed values stay within the original range (conservative property)
    assert t2m_smo.min() >= t2m_raw.min() - 1e-6
    assert t2m_smo.max() <= t2m_raw.max() + 1e-6

    # W was not in params — must be bit-for-bit identical
    np.testing.assert_array_equal(w_raw, w_smo)
