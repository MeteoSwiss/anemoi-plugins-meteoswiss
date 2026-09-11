"""Unit tests for the synop-dwh source's jretrieve wrapper and station catalog.

These run anywhere — no DWH access, no jretrievedwh.py binary — because every
external call is monkeypatched. They pin the credential/prerequisite handling
(mirroring MeteoSwiss/evalml) and the station-catalog collapse logic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anemoi_plugins_meteoswiss.transform.sources.synop_dwh import jretrieve as jr
from anemoi_plugins_meteoswiss.transform.sources.synop_dwh.source import SynopDwhSource
from anemoi_plugins_meteoswiss.transform.sources.synop_dwh.stations import StationCatalog


# --- recipe date coercion (load path passes ISO strings) -------------------

def test_as_datetime_from_string_and_datetime():
    from datetime import datetime as _dt
    assert SynopDwhSource._as_datetime(None) is None
    d = _dt(2024, 1, 1)
    assert SynopDwhSource._as_datetime(d) is d
    assert SynopDwhSource._as_datetime("2024-01-01T00:00:00") == _dt(2024, 1, 1)
    # tz-aware ISO string -> naive datetime (as _fmt_time expects)
    assert SynopDwhSource._as_datetime("2024-06-15T12:00:00+00:00") == _dt(2024, 6, 15, 12)


# --- _stations_to_argv -----------------------------------------------------

def test_stations_to_argv_group():
    # evalml selector: group maps to stn_group_id (our recipes use `group: smn`).
    assert jr._stations_to_argv({"group": "smn"}) == ["-a", "stn_group_id,smn"]


def test_stations_to_argv_locations_from_list():
    assert jr._stations_to_argv({"locations": ["ARO", "KLO"]}) == [
        "-i",
        "nat_abbr,ARO,KLO",
    ]


def test_stations_to_argv_locations_from_string():
    assert jr._stations_to_argv({"locations": "ARO,KLO"}) == ["-i", "nat_abbr,ARO,KLO"]


def test_stations_to_argv_bbox_from_string():
    assert jr._stations_to_argv({"bbox": "45.8,47.8,5.9,10.5"}) == [
        "-l",
        "45.8,47.8,5.9,10.5",
    ]


def test_stations_to_argv_rejects_ambiguous():
    with pytest.raises(ValueError, match="exactly one"):
        jr._stations_to_argv({"group": "smn", "bbox": "1,2,3,4"})


# --- credentials -----------------------------------------------------------

def test_check_credentials_ok_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("JRETRIEVE_CLIENT_ID", "dummy-id")
    monkeypatch.setenv("JRETRIEVE_CLIENT_SECRET", "dummy-secret")
    assert jr._check_credentials(tmp_path) is None


def test_check_credentials_ok_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("JRETRIEVE_CLIENT_ID", raising=False)
    monkeypatch.delenv("JRETRIEVE_CLIENT_SECRET", raising=False)
    (tmp_path / ".env").write_text(
        "JRETRIEVE_CLIENT_ID=id-from-file\nJRETRIEVE_CLIENT_SECRET=secret-from-file\n"
    )
    assert jr._check_credentials(tmp_path) is None


def test_check_credentials_missing_both_no_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("JRETRIEVE_CLIENT_ID", raising=False)
    monkeypatch.delenv("JRETRIEVE_CLIENT_SECRET", raising=False)
    msg = jr._check_credentials(tmp_path)
    assert msg is not None
    assert "JRETRIEVE_CLIENT_ID" in msg and "JRETRIEVE_CLIENT_SECRET" in msg
    assert ".env file not found" in msg


def test_check_credentials_dotenv_exists_but_incomplete(monkeypatch, tmp_path):
    monkeypatch.delenv("JRETRIEVE_CLIENT_ID", raising=False)
    monkeypatch.delenv("JRETRIEVE_CLIENT_SECRET", raising=False)
    (tmp_path / ".env").write_text("JRETRIEVE_CLIENT_ID=id-from-file\n")
    msg = jr._check_credentials(tmp_path)
    assert msg is not None
    assert "JRETRIEVE_CLIENT_SECRET" in msg
    assert ".env file exists" in msg


# --- check_prerequisites ---------------------------------------------------

def test_check_prerequisites_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(jr.shutil, "which", lambda name: "/opt/bin/jretrievedwh.py")
    monkeypatch.setattr(jr, "_locate_conf_dir", lambda: tmp_path)
    monkeypatch.setenv("JRETRIEVE_CLIENT_ID", "dummy-id")
    monkeypatch.setenv("JRETRIEVE_CLIENT_SECRET", "dummy-secret")
    jr.check_prerequisites("prod")  # should not raise


def test_check_prerequisites_rejects_non_prod(monkeypatch, tmp_path):
    monkeypatch.setattr(jr.shutil, "which", lambda name: "/opt/bin/jretrievedwh.py")
    monkeypatch.setattr(jr, "_locate_conf_dir", lambda: tmp_path)
    monkeypatch.setenv("JRETRIEVE_CLIENT_ID", "dummy-id")
    monkeypatch.setenv("JRETRIEVE_CLIENT_SECRET", "dummy-secret")
    with pytest.raises(jr.JretrieveError, match="prod"):
        jr.check_prerequisites("devt")


def test_check_prerequisites_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(jr.shutil, "which", lambda name: None)
    monkeypatch.setattr(jr.os.path, "isfile", lambda p: False)
    monkeypatch.setattr(jr, "_locate_conf_dir", lambda: tmp_path)
    monkeypatch.setenv("JRETRIEVE_CLIENT_ID", "dummy-id")
    monkeypatch.setenv("JRETRIEVE_CLIENT_SECRET", "dummy-secret")
    with pytest.raises(jr.JretrieveError, match=r"\$PATH"):
        jr.check_prerequisites("prod")


def test_check_prerequisites_aggregates_all_problems(monkeypatch):
    # Binary missing AND conf missing -> both reported in one error.
    monkeypatch.setattr(jr.shutil, "which", lambda name: None)
    monkeypatch.setattr(jr.os.path, "isfile", lambda p: False)
    monkeypatch.setattr(jr, "_locate_conf_dir", lambda: None)
    with pytest.raises(jr.JretrieveError) as exc:
        jr.check_prerequisites("prod")
    msg = str(exc.value)
    assert "$PATH" in msg and "not found" in msg


def test_check_prerequisites_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(jr.shutil, "which", lambda name: "/opt/bin/jretrievedwh.py")
    monkeypatch.setattr(jr, "_locate_conf_dir", lambda: tmp_path)
    monkeypatch.delenv("JRETRIEVE_CLIENT_ID", raising=False)
    monkeypatch.delenv("JRETRIEVE_CLIENT_SECRET", raising=False)
    with pytest.raises(jr.JretrieveError, match="JRETRIEVE_CLIENT"):
        jr.check_prerequisites("prod")


# --- retry behaviour -------------------------------------------------------

def test_run_with_retry_fails_fast_on_permanent_error(monkeypatch):
    calls = {"n": 0}

    def fake_run(argv, env, timeout_s):
        calls["n"] += 1
        raise jr.JretrievePermanentError("jretrieve returned error: bad param")

    monkeypatch.setattr(jr, "_run", fake_run)
    with pytest.raises(jr.JretrievePermanentError):
        jr._run_with_retry(["x"], env={}, timeout_s=1, attempts=3)
    assert calls["n"] == 1  # not retried


# --- fetch_data argv -------------------------------------------------------

def test_fetch_data_passes_use_limitation(monkeypatch):
    captured = {}

    def fake_run_with_retry(argv, env, timeout_s):
        captured["argv"] = argv
        return "station;termin;tre200s0\n1;20260501000000;10.0\n"

    monkeypatch.setattr(jr, "_resolve_binary", lambda: "/opt/bin/jretrievedwh.py")
    monkeypatch.setattr(jr, "_build_env", lambda stage: {})
    monkeypatch.setattr(jr, "_run_with_retry", fake_run_with_retry)

    from datetime import datetime

    jr.fetch_data(
        stations={"group": "smn"},
        params=["tre200s0"],
        start=datetime(2026, 5, 1),
        end=datetime(2026, 5, 1, 1),
        increment_minutes=10,
    )
    argv = captured["argv"]
    assert "--use-limitation" in argv
    assert argv[argv.index("--use-limitation") + 1] == str(jr.DEFAULT_USE_LIMITATION)


# --- StationCatalog.from_meta ----------------------------------------------

def _sample_meta():
    return pd.DataFrame(
        {
            "station": [2, 1, 1],
            "op_since": [19900101000000, 19800101000000, 19800101000000],
            "op_till": ["", "", ""],
            "parameter": ["tre200s0", "fkl010z0", "tre200s0"],
            "latitude": [47.48, 46.79, 46.79],
            "longitude": [8.54, 9.68, 9.68],
            "elev": [426.0, 1878.0, 1878.0],
            "stn_name": ["Zurich", "Arosa", "Arosa"],
            "nat_abbr": ["KLO", "ARO", "ARO"],
        }
    )


def test_station_catalog_from_meta_collapses_and_sorts():
    cat = StationCatalog.from_meta(_sample_meta())
    assert cat.n == 2
    assert list(cat.nat_abbr) == ["ARO", "KLO"]  # sorted by nat_abbr
    assert list(cat.station_id) == [1, 2]
    np.testing.assert_allclose(cat.latitude, [46.79, 47.48])


def _meta_with_history():
    """Meta with historical relocations, mixing current (empty op_till) and
    retired (populated op_till) rows across several parameters."""
    return pd.DataFrame(
        {
            "station": [1, 1, 1, 2, 2, 3, 3],
            "op_since": [
                18920101000000,
                20120101000000,
                20120101000000,
                19700101000000,
                20200101000000,
                19500101000000,
                19900101000000,
            ],
            "op_till": [
                "19390101000000",
                "",
                "",
                "19710101000000",
                "",
                "19700101000000",
                "20000101000000",
            ],
            "parameter": [
                "fkl010z0",
                "fkl010z0",
                "tde200s0",
                "pp0qffs0",
                "rre006i0",
                "tre200s0",
                "tre200s0",
            ],
            "latitude": [45.9313, 45.9276, 45.9276, 47.10, 47.20, 40.0, 41.0],
            "longitude": [9.0198, 9.0179, 9.0179, 6.79, 6.80, 1.0, 2.0],
            "elev": [1701.0, 1600.0, 1600.0, 1060.0, 500.0, 100.0, 200.0],
            "stn_name": ["Gen", "Gen", "Gen", "Cdf", "Cdf", "Zzz", "Zzz"],
            "nat_abbr": ["GEN", "GEN", "GEN", "CDF", "CDF", "ZZZ", "ZZZ"],
        }
    )


def test_from_meta_prefers_current_priority_and_falls_back():
    cat = StationCatalog.from_meta(_meta_with_history())
    coord = {
        a: (la, lo, el)
        for a, la, lo, el in zip(cat.nat_abbr, cat.latitude, cat.longitude, cat.elevation)
    }
    # GEN: retired 1892 wind row must NOT win; current 2012 location chosen.
    assert coord["GEN"] == pytest.approx((45.9276, 9.0179, 1600.0))
    # CDF: pressure sensor retired 1971; current precip sensor location wins.
    assert coord["CDF"] == pytest.approx((47.20, 6.80, 500.0))
    # ZZZ: no current row -> fall back to the most recent op_since (1990).
    assert coord["ZZZ"] == pytest.approx((41.0, 2.0, 200.0))
