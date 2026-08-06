import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list

from anemoi_plugins_meteoswiss.transform import filters
from anemoi_plugins_meteoswiss.transform.filters import GaussianSmoother
from anemoi_plugins_meteoswiss.transform.filters import IconRemapToRegLatLon
from anemoi_plugins_meteoswiss.transform.filters import ModelToPressureLevel
from anemoi_plugins_meteoswiss.transform.filters.vertical_interpolation import _select_one

ICONREMAP_WEIGHTS = (
    "/store_new/mch/msopr/icon_workflow_2/iconremap-weights/icon-ch1-eps-rotlatlon.nc"
)


def test_filter_imports():
    # Minimal test to prevent pytest exit code 5 (no tests collected) in CI.
    assert hasattr(filters, "AverageFluxToCumulativeQuantity")
    assert hasattr(filters, "AssignGrid")
    assert hasattr(filters, "GeopotentialFromHeight")


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
    fieldlist = ekd.from_source("file", fn).sel(shortName="2t")

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


def test_gaussian_smoother(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    regridder = IconRemapToRegLatLon(ICONREMAP_WEIGHTS)
    smoother = GaussianSmoother(sigma=5, params=["2t"])

    fn = str(data_dir / "iaf2025010100")
    fs = ekd.from_source("file", fn)
    # T_2M (smoothed) + one level of W (pass-through, not in params)
    fieldlist = new_fieldlist_from_list(
        list(fs.sel(shortName="2t")) + [fs.sel(shortName="wz")[0]]
    )

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


def test_select_one_finds_field_by_native_shortname(data_dir):
    fn = str(data_dir / "iaf2025010100")
    fieldlist = ekd.from_source("file", fn).sel(shortName="2t")

    da = _select_one(fieldlist, "2t")
    assert da.shape == (1147980,)


def test_select_one_raises_when_missing(data_dir):
    fn = str(data_dir / "iaf2025010100")
    fieldlist = ekd.from_source("file", fn).sel(shortName="2t")

    with pytest.raises(ValueError, match="auxiliary_source='pipe'"):
        _select_one(fieldlist, "NOT_IN_THIS_FILE")


# ModelToPressureLevel's auxiliary_source="pipe" mode (reads HHL/HSURF/T_2M/PS/P
# from the same input FieldList instead of FDB, for environments without FDB
# access) needs a real raw KENDA analysis file with T, U, V, W, QV, P, HHL,
# HSURF, T_2M and PS all present to test end-to-end — the checked-in
# `iaf2025010100` fixture only carries `t`/`2t`/`wz` (added for the
# IconRemapToRegLatLon/GaussianSmoother tests above) and COSMO shortNames
# can't be faked via metadata override (eccodes resolves them from the
# GRIB2 paramId via a concept table, not a free-form string) — see if a
# fuller fixture is available before deciding this needs balfrin/real data.
def test_model_to_pressure_level_pipe_mode(data_dir, hostname):
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin.")

    fn = str(data_dir / "iaf2025010100")
    fieldlist = ekd.from_source("file", fn)
    needed = {"T", "U", "V", "W", "QV", "P", "HHL", "HSURF", "T_2M", "PS"}
    present = {f.metadata("shortName") for f in fieldlist}
    missing = needed - present
    if missing:
        pytest.skip(f"iaf2025010100 fixture is missing {sorted(missing)} — need a fuller raw KENDA fixture")

    interpolator = ModelToPressureLevel(
        interpolate_levels=[500, 700, 850, 925, 1000],
        extrapolate_levels=[925, 1000],
        add_geopotential=True,
        auxiliary_source="pipe",
        surface_params=["T_2M", "PS", "HSURF"],
    )
    out = interpolator.forward(fieldlist)

    out_names = {f.metadata("shortName") for f in out}
    assert "HHL" not in out_names  # consumed, never re-emitted
    assert "P" not in out_names  # consumed, never re-emitted
    assert {"T_2M", "PS", "HSURF"} <= out_names  # passed through unchanged
    assert "FI" in out_names  # geopotential derived from HHL

    t_levels = sorted(int(f.metadata("level")) for f in out if f.metadata("shortName") == "T")
    assert t_levels == [500, 700, 850, 925, 1000]
