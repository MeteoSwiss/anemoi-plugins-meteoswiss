"""Tests for the rea-l-ch1-precip and rea-l-ch1-wind-gust sources, with fake FDB retrievals."""

import datetime

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.transform.fields import new_fieldlist_from_list
from grib_helpers import grib_field
from grib_helpers import time_metadata

from anemoi_plugins_meteoswiss.transform.sources import realch1_precip
from anemoi_plugins_meteoswiss.transform.sources import realch1_wind_gust
from anemoi_plugins_meteoswiss.transform.sources.realch1_precip import ReaLCh1Precip
from anemoi_plugins_meteoswiss.transform.sources.realch1_wind_gust import ReaLCh1WindGust

H = datetime.timedelta(hours=1)
DAY = datetime.datetime(2020, 1, 29)
LATITUDES = np.array([45.5, 46.0, 46.5, 47.0, 47.5])
LONGITUDES = np.array([6.0, 7.5, 8.0, 9.5, 10.0])
NPOINTS = len(LATITUDES)


class StubGrid:
    def latlon(self):
        return LATITUDES, LONGITUDES


def _increment(valid_time: datetime.datetime) -> np.ndarray:
    """Precipitation fallen in the hour ending at valid_time (the oracle)."""
    hours = (valid_time - DAY) / H
    return np.abs(np.sin(hours + np.arange(NPOINTS))) * (1 + hours % 5)


def _accumulated_from_start(date: datetime.datetime, step: int) -> np.ndarray:
    """Precipitation accumulated from the forecast start (00 UTC of date) to step."""
    return sum((_increment(date + k * H) for k in range(1, step + 1)), np.zeros(NPOINTS))


def _gust(date: datetime.datetime, step: int) -> np.ndarray:
    """10-minute maximum gust ending at the step (in hours) of the forecast started on date."""
    hours = (date - DAY) / H + step
    return np.abs(np.cos(hours * 0.7 + np.arange(NPOINTS))) * 20


def _minutes(step) -> int:
    """GRIB step in minutes ("650m" or hours as int)."""
    return int(step[:-1]) if isinstance(step, str) and step.endswith("m") else int(step) * 60


class FakeFDB:
    """Replaces ekd.from_source("fdb", request) with synthetic fields."""

    def __init__(self, make_field):
        self.make_field = make_field
        self.requests = []

    def __call__(self, name, request, **kwargs):
        assert name == "fdb"
        assert not kwargs, kwargs
        self.requests.append(request)
        field = self.make_field(datetime.datetime.strptime(request["date"], "%Y%m%d"), request["step"])
        return FakeFDBSource(field)


class FakeFDBSource:
    def __init__(self, field):
        self.field = field

    def to_fieldlist(self, read_all=False):
        assert read_all
        return new_fieldlist_from_list([self.field])


def _precip_field(date, step):
    return grib_field(
        _accumulated_from_start(date, step),
        date=int(date.strftime("%Y%m%d")),
        start_step=0,
        end_step=step,
        step_type="accum",
    )


def _gust_field(date, step):
    return grib_field(
        _gust(date, step),
        date=int(date.strftime("%Y%m%d")),
        start_step=max(step * 60 - 10, 0),
        end_step=step * 60,
        step_units="m",
        step_type="max",
        type_of_level="heightAboveGround",
        level=10,
    )


@pytest.fixture
def fake_precip_fdb(monkeypatch):
    fake = FakeFDB(_precip_field)
    monkeypatch.setattr(ekd, "from_source", fake)
    return fake


@pytest.fixture
def fake_gust_fdb(monkeypatch):
    fake = FakeFDB(_gust_field)
    monkeypatch.setattr(ekd, "from_source", fake)
    return fake


@pytest.mark.parametrize("module", [realch1_precip, realch1_wind_gust])
def test_as_fct_time_request(module):
    assert module._as_fct_time_request(DAY) == {"date": "20200128", "time": "0000", "step": 24}
    assert module._as_fct_time_request(DAY + 13 * H) == {"date": "20200129", "time": "0000", "step": 13}
    assert module._as_fct_time_request(DAY + 13 * H + datetime.timedelta(minutes=50))["step"] == 13


def test_precip_prepare_dates():
    dates = [DAY + 2 * H, DAY]
    assert realch1_precip._prepare_dates(dates, 3) == [DAY - 3 * H, DAY - 2 * H, DAY - H, DAY, DAY + 2 * H]
    with pytest.raises(ValueError, match="unique"):
        realch1_precip._prepare_dates([DAY, DAY], 1)


@pytest.mark.parametrize("period", [1, 6])
def test_precip(fake_precip_fdb, period):
    dates = [DAY + k * H for k in range(24)]
    source = ReaLCh1Precip(None, accumulation_period_hours=period)
    source.grid = StubGrid()

    result = source.execute(dates)

    assert len(fake_precip_fdb.requests) == 24 + period
    assert [f.get("time.valid_datetime") for f in result] == dates
    for date, field in zip(dates, result):
        expected = sum((_increment(date - k * H) for k in range(period)), np.zeros(NPOINTS))
        np.testing.assert_allclose(field.values, expected, rtol=1e-5, atol=1e-5)
        md = time_metadata(field)
        assert md["stepType"] == "accum"
        assert int(md["startStep"]) == int(md["endStep"]) - period, md
        assert field.get("time.step") == datetime.timedelta(hours=int(md["endStep"]))
        assert field.get("vertical.level") == 0
        np.testing.assert_allclose(field.geography.latitudes(), LATITUDES)
        np.testing.assert_allclose(field.geography.longitudes(), LONGITUDES)

    # 00 UTC comes from the previous day's +24h, 01 UTC subtracts across the day boundary
    assert time_metadata(result[0])["dataDate"] == 20200128
    assert time_metadata(result[1])["startStep"] == 1 - period


def test_precip_missing_previous_day_accumulation():
    fields = new_fieldlist_from_list([_precip_field(DAY - datetime.timedelta(days=1), 23), _precip_field(DAY, 1)])
    with pytest.raises(ValueError, match="missing previous day 24h accumulation"):
        realch1_precip._accumulation_logic(fields, 1)


def test_precip_invalid_period():
    with pytest.raises(ValueError):
        ReaLCh1Precip(None, accumulation_period_hours=25)


def test_wind_gust_prepare_dates():
    minutes = datetime.timedelta(minutes=10)
    assert realch1_wind_gust._prepare_dates(DAY, 30) == [DAY - 2 * minutes, DAY - minutes, DAY]


@pytest.mark.parametrize("hour", [0, 1, 12])
def test_wind_gust(fake_gust_fdb, hour):
    date = DAY + hour * H
    source = ReaLCh1WindGust(None, aggregation_period_minutes=60)
    source.grid = StubGrid()

    (field,) = source.execute([date])

    # the 60 minute window is retrieved at hourly resolution: the previous full hour (5 times) and the hour itself
    base, step = (DAY - datetime.timedelta(days=1), 24) if hour == 0 else (DAY, hour)
    assert [r["step"] for r in fake_gust_fdb.requests] == [step - 1] * 5 + [step]
    np.testing.assert_allclose(field.values, np.maximum(_gust(base, step - 1), _gust(base, step)), rtol=1e-5)

    md = time_metadata(field)
    assert md["stepType"] == "max"
    assert _minutes(md["startStep"]) == max((step - 1) * 60 - 10, 0)
    assert _minutes(md["endStep"]) == step * 60
    assert field.get("time.valid_datetime") == date
    assert field.get("vertical.level") is None
    np.testing.assert_allclose(field.geography.latitudes(), LATITUDES)


def test_wind_gust_invalid_period():
    with pytest.raises(ValueError):
        ReaLCh1WindGust(None, aggregation_period_minutes=5)
