"""Tests for the thx-lightning source (synthetic archive, no external data)."""

import zipfile
from datetime import date
from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.spatial

from anemoi_plugins_meteoswiss.transform.sources import thx_lightning_core as core
from anemoi_plugins_meteoswiss.transform.sources.thx_lightning import ThxLightning
from anemoi_plugins_meteoswiss.transform.sources.thx_lightning import variable_name

DAY = date(2025, 8, 1)

# The original lightning-grid code used scipy's compiled cKDTree, which has no
# type stubs; fetched by name so type checkers do not flag the import.
cKDTree = getattr(scipy.spatial, "cKDTree")


def _line(
    t: datetime, lon: float, lat: float, intra: int = 0, current: float = -5.0
) -> str:
    stamp = t.strftime("%d.%m.%Y %H:%M:%S.") + f"{t.microsecond // 100_000}"
    fields = [
        f"{stamp} UTC",
        f"{lon:.3f}",
        f"{lat:.3f}",
        f"{current}",
        "1",
        "3",
        str(intra),
    ] + ["0.0"] * 9
    return "|".join(fields) + "|"


def _write_day(archive, day: date, lines: list[str], zipped: bool = True):
    stem = core.thx_file_stem(day)
    text = "\n".join(lines) + "\n"
    if zipped:
        with zipfile.ZipFile(archive / f"{stem}.prd.zip", "w") as zf:
            zf.writestr(f"{stem}.prd", text)
    else:
        (archive / f"{stem}.prd").write_text(text)


def _small_icon_grid() -> core.IconGrid:
    """0.02 deg lattice around Bern (~2 km spacing), cells are lattice points."""
    lat, lon = np.meshgrid(
        np.arange(46.5, 47.5, 0.02), np.arange(7.0, 8.0, 0.02), indexing="ij"
    )
    return core.IconGrid(lat.ravel(), lon.ravel())


def _reference_counts(grid: core.IconGrid, lon, lat, radii_km) -> np.ndarray:
    """The original per-strike loop of lightning-grid (v1-netcdf), for comparison."""
    counts = np.zeros((len(radii_km), grid.ncells), dtype=np.int32)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    xyz = np.column_stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)]
    )
    # an independent tree with cKDTree's defaults, exactly as in the original code
    tree = cKDTree(core._unit_xyz(grid.latitudes, grid.longitudes))
    nn_dist, _ = tree.query(xyz, k=1)
    xyz = xyz[nn_dist <= grid._domain_chord]
    for i, r_km in enumerate(radii_km):
        for cells in tree.query_ball_point(xyz, r=core._km_to_chord(r_km)):
            if cells:
                np.add.at(counts[i], cells, 1)
    return counts


# --- reader -----------------------------------------------------------------


def test_reader_filters_discharge_type_and_drops_malformed_lines(tmp_path):
    t = datetime(2025, 8, 1, 12, 0, 1)
    glued = _line(t, 7.4, 46.9) + _line(t, 7.5, 46.9)  # two records on one line
    _write_day(
        tmp_path,
        DAY,
        [_line(t, 7.4, 46.9, intra=0), _line(t, 7.4, 46.9, intra=1), glued],
    )

    assert len(core.read_thx_day(DAY, tmp_path, ["CG", "IC"])) == 2
    assert len(core.read_thx_day(DAY, tmp_path, ["CG"])) == 1
    assert len(core.read_thx_day(DAY, tmp_path, ["IC"])) == 1


def test_reader_drops_out_of_range_coordinates(tmp_path):
    # latitude 408.09 would wrap around to 48.09 N on the unit sphere (real case: 2024-09-08)
    t = datetime(2025, 8, 1, 12, 0, 1)
    _write_day(
        tmp_path,
        DAY,
        [_line(t, 12.331, 408.09), _line(t, 525.06, 7.58), _line(t, 7.4, 46.9)],
    )
    df = core.read_thx_day(DAY, tmp_path, ["CG", "IC"])
    assert list(df["lat"]) == [46.9]


def test_reader_drops_strikes_outside_the_files_day(tmp_path):
    _write_day(
        tmp_path,
        DAY,
        [
            _line(datetime(2025, 8, 1, 23, 59), 7.4, 46.9),
            _line(datetime(2025, 8, 2, 0, 0, 1), 7.4, 46.9),
        ],
    )
    df = core.read_thx_day(DAY, tmp_path, ["CG", "IC"])
    assert len(df) == 1


def test_reader_parses_unzipped_files_and_empty_days(tmp_path):
    _write_day(tmp_path, DAY, [_line(datetime(2025, 8, 1, 1), 7.4, 46.9)], zipped=False)
    assert len(core.read_thx_day(DAY, tmp_path, ["CG", "IC"])) == 1
    _write_day(tmp_path, DAY + timedelta(days=1), [""])
    assert core.read_thx_day(DAY + timedelta(days=1), tmp_path, ["CG", "IC"]).empty


def test_reader_fails_clearly_on_missing_or_broken_files(tmp_path):
    with pytest.raises(core.ThxArchiveError, match="No THX file"):
        core.read_thx_day(DAY, tmp_path, ["CG", "IC"])
    (tmp_path / f"{core.thx_file_stem(DAY)}.prd.zip").write_bytes(b"not a zip")
    with pytest.raises(core.ThxArchiveError, match="Cannot read"):
        core.read_thx_day(DAY, tmp_path, ["CG", "IC"])


# --- counting ---------------------------------------------------------------


def test_counts_match_the_original_per_strike_loop():
    grid = _small_icon_grid()
    rng = np.random.default_rng(0)
    lon, lat = (
        rng.uniform(6.9, 8.1, 3000),
        rng.uniform(46.4, 47.6, 3000),
    )  # some strikes outside the grid
    points, inside = grid.prepare(lon, lat)
    grid.batch_size = 257  # exercise batching
    new = grid.count(points[inside], [4.0, 8.0, 16.0])
    np.testing.assert_array_equal(
        new, _reference_counts(grid, lon, lat, [4.0, 8.0, 16.0])
    )


def test_radius_zero_counts_nearest_cell_only():
    grid = _small_icon_grid()
    points, inside = grid.prepare(np.array([7.5]), np.array([47.0]))
    counts = grid.count(points[inside], [0.0])
    assert counts.sum() == 1


# --- windows and aggregation -------------------------------------------------


@pytest.fixture
def archive(tmp_path):
    """One strike every 5 minutes at the same place on DAY, and one at 23:55 the day before."""
    lines = [
        _line(datetime(2025, 8, 1) + timedelta(minutes=5 * k), 7.5, 47.0)
        for k in range(288)
    ]
    _write_day(tmp_path, DAY, lines)
    _write_day(
        tmp_path,
        DAY - timedelta(days=1),
        [_line(datetime(2025, 7, 31, 23, 55), 7.5, 47.0)],
    )
    return core.StrikeArchive(tmp_path, ["CG", "IC"], _small_icon_grid())


def _peak(archive, valid, period, aggregation="accumulation", sub=None):
    return core.window_counts(archive, valid, period, [4.0], aggregation, sub).max()


def test_window_is_closed_at_start_and_open_at_end(archive):
    # strikes at 12:00 and 12:05 are in [12:00, 12:10); the one at 12:10 is not
    assert _peak(archive, datetime(2025, 8, 1, 12, 10), timedelta(minutes=10)) == 2


def test_midnight_window_uses_previous_day(archive):
    assert _peak(archive, datetime(2025, 8, 1, 0, 0), timedelta(minutes=10)) == 1
    assert _peak(archive, datetime(2025, 8, 1, 0, 10), timedelta(minutes=10)) == 2


def test_hour_equals_sum_of_six_ten_minute_windows(archive):
    valid = datetime(2025, 8, 1, 13, 0)
    hour = core.window_counts(archive, valid, timedelta(hours=1), [4.0, 8.0])
    tens = sum(
        core.window_counts(
            archive,
            valid - k * timedelta(minutes=10),
            timedelta(minutes=10),
            [4.0, 8.0],
        )
        for k in range(6)
    )
    np.testing.assert_array_equal(hour, tens)


def test_maximum_and_average_over_sub_periods(archive):
    valid, hour, ten = (
        datetime(2025, 8, 1, 13, 0),
        timedelta(hours=1),
        timedelta(minutes=10),
    )
    assert _peak(archive, valid, hour) == 12
    assert _peak(archive, valid, hour, "maximum", ten) == 2
    assert _peak(archive, valid, hour, "average", ten) == pytest.approx(2.0)
    assert _peak(archive, valid, hour, "maximum", timedelta(minutes=30)) == 6


# --- source -----------------------------------------------------------------


@pytest.mark.parametrize(
    "radius, period, aggregation, sub, expected",
    [
        (4, "10m", "accumulation", None, "LC_4KM_10M"),
        (16, "1h", "accumulation", None, "LC_16KM_1H"),
        (4, "1h", "maximum", "10m", "LC_4KM_1H_MAX10M"),
        (8, "6h", "average", "1h", "LC_8KM_6H_AVG1H"),
        (2.5, "30m", "accumulation", None, "LC_2.5KM_30M"),
    ],
)
def test_variable_names(radius, period, aggregation, sub, expected):
    from anemoi.utils.dates import frequency_to_timedelta

    sub = frequency_to_timedelta(sub) if sub else None
    assert (
        variable_name(radius, frequency_to_timedelta(period), aggregation, sub)
        == expected
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(aggregation="sum"), "aggregation must be one of"),
        (dict(aggregation="maximum"), "requires sub_period"),
        (dict(aggregation="maximum", sub_period="7m"), "must divide period"),
        (dict(sub_period="10m"), "only used with"),
        (dict(radius_km=[4, 4]), "distinct"),
        (dict(grid_definition={"lambert": {}}), "Unsupported grid"),
    ],
)
def test_invalid_arguments(kwargs, message):
    args = dict(grid_definition={"inca": {}}, period="1h", radius_km=4) | kwargs
    with pytest.raises(ValueError, match=message):
        ThxLightning(MagicMock(), **args)


def test_source_fields_carry_anemoi_metadata(tmp_path):
    _write_day(tmp_path, DAY, [_line(datetime(2025, 8, 1, 12, 5), 7.44, 46.95)])
    src = ThxLightning(
        MagicMock(),
        grid_definition={"inca": {}},
        period="1h",
        radius_km=[4, 16],
        archive_path=str(tmp_path),
    )
    fields = src.execute([datetime(2025, 8, 1, 13, 0)])

    assert [f.metadata("param") for f in fields] == ["LC_4KM_1H", "LC_16KM_1H"]
    f = fields[0]
    assert f.metadata(namespace="mars") == dict(
        date=20250801, time=1200, step="1h", levtype="sfc", param="LC_4KM_1H"
    )
    assert f.metadata("valid_datetime") == "2025-08-01T13:00:00"
    assert (f.metadata("startStep"), f.metadata("endStep")) == (
        timedelta(0),
        timedelta(hours=1),
    )
    assert f.metadata("typeOfStatisticalProcessing") == 1 and f.metadata("units") == "1"
    values = f.to_numpy(flatten=True)
    assert values.dtype == np.float32 and values.shape == (
        core.IncaGrid.NX * core.IncaGrid.NY,
    )
    assert np.nanmax(values) == 1 and np.nanmax(fields[1].to_numpy(flatten=True)) == 1
    assert np.nansum(fields[1].to_numpy()) > np.nansum(
        values
    )  # bigger radius, more cells


def test_cells_outside_network_bbox_are_nan(tmp_path):
    _write_day(tmp_path, DAY, [""])
    src = ThxLightning(
        MagicMock(),
        grid_definition={"inca": {}},
        period="10m",
        radius_km=4,
        archive_path=str(tmp_path),
        network_bbox=[46.0, 50.2, 0.0, 17.0],
    )
    values = src.execute([datetime(2025, 8, 1, 1, 0)])[0].to_numpy(flatten=True)
    south = src.grid.latitudes < 46.0
    assert np.isnan(values[south]).all() and (values[~south] == 0).all()


def test_grid_tree_has_ckdtree_structure():
    """KDTree(leafsize=16) builds the same tree as the original cKDTree."""
    grid = _small_icon_grid()
    fresh = cKDTree(core._unit_xyz(grid.latitudes, grid.longitudes))
    assert grid._tree.leafsize == fresh.leafsize
    np.testing.assert_array_equal(grid._tree.data, fresh.data)
    np.testing.assert_array_equal(grid._tree.indices, fresh.indices)
