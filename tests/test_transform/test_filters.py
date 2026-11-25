from pathlib import Path

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.sources.array_list import ArrayField

from anemoi_plugins_meteoswiss.transform import filters
from anemoi_plugins_meteoswiss.transform.filters import GaussianSmoother
from anemoi_plugins_meteoswiss.transform.filters import IconRemapToRegLatLon
from anemoi_plugins_meteoswiss.transform.filters import Keep
from anemoi_plugins_meteoswiss.transform.filters import ModelToPressureLevel

ICONREMAP_WEIGHTS = "/store_new/mch/msopr/icon_workflow_2/iconremap-weights/icon-ch1-eps-rotlatlon.nc"
requires_iconremap_weights = pytest.mark.skipif(
    not Path(ICONREMAP_WEIGHTS).exists(), reason=f"{ICONREMAP_WEIGHTS} not available on this machine."
)
ICON_CH1_GRID = "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/icon_grid_0001_R19B08_mch.nc"
requires_icon_ch1_grid = pytest.mark.skipif(
    not Path(ICON_CH1_GRID).exists(), reason=f"{ICON_CH1_GRID} not available on this machine."
)


def test_filter_imports():
    # Minimal test to prevent pytest exit code 5 (no tests collected) in CI.
    assert hasattr(filters, "AverageFluxToCumulativeQuantity")
    assert hasattr(filters, "AssignGrid")
    assert hasattr(filters, "GeopotentialFromHeight")


@requires_iconremap_weights
def test_icon_remap_to_reg_lat_lon(data_dir):
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
    for src_field, out_field in zip(fieldlist, result):
        src_values = src_field.to_numpy(flatten=True)
        values = out_field.to_numpy(flatten=True)
        assert values.shape == (n_out,)
        # All output points are valid for this weights file (no sentinel zeros)
        assert not np.any(np.isnan(values))
        # Conservative interpolation: output stays within the source's range.
        assert values.min() >= src_values.min() - 1e-6
        assert values.max() <= src_values.max() + 1e-6


@requires_iconremap_weights
@requires_icon_ch1_grid
def test_icon_remap_output_coordinates_match_icon_cells():
    """Output lat/lon labels must match where the regridded values actually come from.

    Interpolating the ICON cell-centre coordinates through the weights gives the true position
    of each output point, independently of the rotated-pole conversion. This catches errors such
    as the longitudes being mirrored around the pole meridian (10E for ICON-CH).
    """
    import xarray as xr

    regridder = IconRemapToRegLatLon(ICONREMAP_WEIGHTS, device="cpu")
    with xr.open_dataset(ICON_CH1_GRID) as grid:
        clon = np.degrees(grid["clon"].values.astype(np.float64))
        clat = np.degrees(grid["clat"].values.astype(np.float64))

    valid = regridder.valid_mask
    idx, wgt = regridder.indices[valid], regridder.weights[valid]
    true_lon = np.sum(clon[idx] * wgt, axis=-1)
    true_lat = np.sum(clat[idx] * wgt, axis=-1)

    np.testing.assert_allclose(regridder._longitudes[valid], true_lon, atol=1e-3)
    np.testing.assert_allclose(regridder._latitudes[valid], true_lat, atol=1e-3)


@requires_iconremap_weights
def test_gaussian_smoother(data_dir):
    regridder = IconRemapToRegLatLon(ICONREMAP_WEIGHTS)
    smoother = GaussianSmoother(sigma=5, params=["T_2M"])

    fn = str(data_dir / "iaf2025010100")
    fs = ekd.from_source("file", fn)
    # T_2M (smoothed) + one level of W (pass-through, not in params)
    fieldlist = new_fieldlist_from_list(list(fs.sel(shortName="T_2M")) + [fs.sel(shortName="W")[0]])

    regridded = list(regridder.forward(fieldlist))
    synthetic = np.zeros((regridder.ny, regridder.nx))
    synthetic[::20, ::20] = 100.0
    regridded[0] = new_field_from_numpy(synthetic.ravel(), template=regridded[0])
    regridded = new_fieldlist_from_list(regridded)

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


def test_keep(caplog):
    fieldlist = new_fieldlist_from_list([ArrayField(np.zeros(4), RawMetadata({"param": p})) for p in ["t", "q", "z"]])

    kept = Keep(param=["t", "z"]).forward(fieldlist)
    assert [f.metadata("param") for f in kept] == ["t", "z"]

    with caplog.at_level("WARNING"):
        Keep(param=["t", "missing"]).forward(fieldlist)
    assert "missing" in caplog.text


def test_pipe_or_fdb_xarray_returns_piped_value_when_present(data_dir):
    fn = str(data_dir / "iaf2025010100")
    fieldlist = ekd.from_source("file", fn).sel(shortName="T_2M")

    interpolator = ModelToPressureLevel(interpolate_levels=[500])
    da = interpolator._get_field(fieldlist, "T_2M").to_xarray()["T_2M"]
    assert da.shape == (1147980,)


def test_retrieve_observation_once_per_ref_time():
    """Repeated calls with the same ref_time retrieve observations only once."""
    from datetime import datetime
    from unittest.mock import patch

    from anemoi_plugins_meteoswiss.transform.filters import RetrieveObservation

    class _Field:
        def __init__(self, valid_time):
            self._valid_time = valid_time

        def datetime(self):
            return {"valid_time": self._valid_time}

    t0, t1 = datetime(2025, 3, 1, 0), datetime(2025, 3, 1, 6)
    filt = RetrieveObservation(obs_path="obs.parquet", jretrieve_src_path="/unused", run_mode="devt")
    with patch.object(RetrieveObservation, "_retrieve") as retrieve:
        data = [_Field(t0)]
        assert filt.forward(data) is data
        filt.forward([_Field(t0)])
        filt.forward([_Field(t1)])
    assert [c.args[0] for c in retrieve.call_args_list] == [t0, t1]
