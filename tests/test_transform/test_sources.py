import os
from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from anemoi_plugins_meteoswiss.transform.sources import ReaLCh1TimeStatistic
from anemoi_plugins_meteoswiss.transform.sources import ReaLCh1WindGust
from anemoi_plugins_meteoswiss.transform.sources import realch1_time_statistic as module


def test_window_covers_the_period_ending_at_the_date():
    times = module.window(datetime(2020, 1, 2, 3), timedelta(hours=6))
    assert len(times) == 36
    assert times[0] == datetime(2020, 1, 1, 21, 10)
    assert times[-1] == datetime(2020, 1, 2, 3)
    assert all(b - a == timedelta(minutes=10) for a, b in zip(times, times[1:]))


def test_window_rejects_periods_that_are_not_multiples_of_10_minutes():
    with pytest.raises(ValueError):
        module.window(datetime(2020, 1, 2, 3), timedelta(minutes=15))


@pytest.mark.parametrize(
    "valid, expected",
    [
        (datetime(2020, 1, 2, 0, 10), (date(2020, 1, 2), 10)),
        (datetime(2020, 1, 2, 1), (date(2020, 1, 2), 60)),
        (datetime(2020, 1, 2, 23, 50), (date(2020, 1, 2), 1430)),
        (
            datetime(2020, 1, 2, 0),
            (date(2020, 1, 1), 1440),
        ),  # midnight: step 24 of the previous run
    ],
)
def test_run_and_step(valid, expected):
    assert module.run_and_step(valid) == expected


@pytest.mark.parametrize(
    "minutes, expected",
    [(10, "10m"), (60, "1"), (70, "1h10m"), (1430, "23h50m"), (1440, "24")],
)
def test_format_step(minutes, expected):
    assert module.format_step(minutes) == expected


def test_requests_one_per_run_date_across_midnight():
    times = module.window(
        datetime(2020, 1, 2, 1), timedelta(hours=2)
    )  # 23:10 ... 01:00
    reqs = module.requests([500164, 500011], times)
    assert [r["date"] for r in reqs] == ["20200101", "20200102"]
    assert reqs[0]["step"] == ["23h10m", "23h20m", "23h30m", "23h40m", "23h50m", "24"]
    assert reqs[1]["step"] == ["10m", "20m", "30m", "40m", "50m", "1"]
    assert all(
        r["param"] == [500164, 500011] and r["time"] == "0000" and r["levtype"] == "sfc"
        for r in reqs
    )


class FakeField:
    def __init__(self, param_id, valid, values):
        self.param_id = param_id
        self.valid = valid
        self.values = values

    def clone(self, values, **metadata):
        out = FakeField(self.param_id, self.valid, values)
        out.metadata_overrides = metadata
        return out


def fake_retrieve(param_ids, times):
    # one value per field: minutes since 2020-01-01 00:00, negated for the second parameter
    fields = {}
    for p in param_ids:
        for t in times:
            minutes = (t - datetime(2020, 1, 1)).total_seconds() / 60
            fields[(p, t)] = FakeField(
                p, t, np.array([minutes if p == 500164 else -minutes])
            )
    return fields


@pytest.fixture
def fake_fdb(monkeypatch):
    monkeypatch.setattr(module, "_retrieve", fake_retrieve)
    monkeypatch.setattr(module, "new_field_from_grid", lambda field, grid: field)
    monkeypatch.setattr(module, "new_fieldlist_from_list", list)


def test_execute_reduces_each_window(fake_fdb):
    source = ReaLCh1TimeStatistic(
        None, param=["VMAX_10M", "T_2M"], statistic="max", period_hours=6
    )
    dates = [datetime(2020, 1, 2, 0), datetime(2020, 1, 2, 1)]
    out = source.execute(dates)
    assert [(f.param_id, f.valid) for f in out] == [
        (500164, dates[0]),
        (500011, dates[0]),
        (500164, dates[1]),
        (500011, dates[1]),
    ]
    # max of an increasing series is the last value, max of a decreasing series the first one
    assert_array_equal(out[0].values, [1440])
    assert_array_equal(out[1].values, [-(1440 - 350)])
    assert_array_equal(out[2].values, [1500])
    assert_array_equal(out[3].values, [-(1500 - 350)])
    assert out[0].metadata_overrides == {
        "level": None,
        "levelist": None,
        "levtype": "sfc",
        "stepType": "max",
        "typeOfStatisticalProcessing": 2,
        "startStep": 0,
        "endStep": 6,
    }


def test_execute_statistics(fake_fdb):
    date_ = datetime(2020, 1, 2, 3)
    expected = {"max": 1620, "min": 1570, "mean": 1595}
    for statistic, value in expected.items():
        source = ReaLCh1TimeStatistic(
            None, param=500164, statistic=statistic, period_hours=1
        )
        assert_array_equal(source.execute([date_])[0].values, [value])


def test_execute_raises_on_missing_fields(fake_fdb, monkeypatch):
    monkeypatch.setattr(module, "_retrieve", lambda param_ids, times: {})
    source = ReaLCh1TimeStatistic(None, param="VMAX_10M")
    with pytest.raises(ValueError, match="Missing field"):
        source.execute([datetime(2020, 1, 2, 3)])


def test_wind_gust_is_the_maximum_of_vmax_10m(fake_fdb):
    source = ReaLCh1WindGust(None, aggregation_period_minutes=360)
    assert source.param_ids == [500164]
    assert source.statistic == "max"
    assert source.period == timedelta(hours=6)
    with pytest.raises(ValueError):
        ReaLCh1WindGust(None, aggregation_period_minutes=90)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"param": "NOPE"},
        {"param": "T_2M", "statistic": "median"},
        {"param": "T_2M", "period_hours": 0},
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ReaLCh1TimeStatistic(None, **kwargs)


def test_against_fdb():
    if "FDB5_CONFIG_FILE" not in os.environ:
        pytest.skip("Needs the REA-L-CH1 FDB (run inside the FDB uenv).")
    import earthkit.data as ekd

    date_ = datetime(2020, 1, 2, 3)
    source = ReaLCh1TimeStatistic(
        None, param="VMAX_10M", statistic="max", period_hours=6
    )
    actual = source.execute([date_])[0]
    assert actual.datetime()["valid_time"] == date_
    assert actual.metadata("levtype") == "sfc"
    assert actual.metadata("stepType") == "max"
    assert (actual.metadata("startStep"), actual.metadata("endStep")) == (0, 6)

    expected = []
    for valid in module.window(date_, timedelta(hours=6)):
        run_date, step = module.run_and_step(valid)
        request = module.FDB_REQUEST | {
            "date": run_date.strftime("%Y%m%d"),
            "time": "0000",
            "param": 500164,
            "step": module.format_step(step),
        }
        expected.append(ekd.from_source("fdb", request, read_all=True)[0].values)
    assert_array_equal(actual.values, np.max(expected, axis=0))
