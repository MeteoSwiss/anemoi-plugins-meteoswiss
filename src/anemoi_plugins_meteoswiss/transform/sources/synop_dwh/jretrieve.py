"""Subprocess wrapper around `jretrievedwh.py`.

Mirrors the auth/env-var setup that the operational osm wrapper
(https://github.com/MeteoSwiss-APN/oprtools/blob/main/scripts/jretrievedwh)
does, then invokes the Python REST client and parses CSV output into pandas
DataFrames.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from io import StringIO
from typing import Any, Sequence

import pandas as pd

LOG = logging.getLogger(__name__)

VALID_STAGES = {"prod", "depl", "devt"}
DEFAULT_META_FIELDS: tuple[str, ...] = ("lat", "lon", "elev", "name", "nat_abbr")


class JretrieveError(RuntimeError):
    """Raised when jretrievedwh.py fails or returns malformed output."""


def _resolve_binary() -> str:
    path = shutil.which("jretrievedwh.py")
    if path is None:
        raise JretrieveError(
            "jretrievedwh.py not found in $PATH. "
            "Make sure /oprusers/osm/opr.inn/bin (or equivalent) is on your PATH."
        )
    return path


def _build_env(stage: str) -> dict[str, str]:
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage {stage!r}. Must be one of {sorted(VALID_STAGES)}.")
    opr_home = os.environ.get("OPR_HOME")
    if not opr_home:
        raise JretrieveError("OPR_HOME is not set; cannot locate jretrieve conf file.")

    conf_name = f".jretrievedwh-conf.{stage}.py"
    conf_path = os.path.join(opr_home, conf_name)
    if not os.path.isfile(conf_path):
        raise JretrieveError(f"jretrieve conf file not found: {conf_path}")
    if not os.access(conf_path, os.R_OK):
        raise JretrieveError(f"jretrieve conf file not readable: {conf_path}")

    env = os.environ.copy()
    env["JRETRIEVE_CONF_DIR"] = opr_home
    env["JRETRIEVE_CONF_NAME"] = conf_name
    return env


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def _stations_to_argv(stations: dict[str, Any]) -> list[str]:
    """Translate a `stations:` recipe dict to jretrieve CLI args.

    Exactly one of {group, locations, bbox} must be set.
    """
    keys = [k for k in ("group", "locations", "bbox") if k in stations and stations[k] is not None]
    if len(keys) != 1:
        raise ValueError(
            f"stations must specify exactly one of group/locations/bbox, got {keys}"
        )
    key = keys[0]
    val = stations[key]

    if key == "group":
        return ["-a", f"stn_group,{val}"]
    if key == "locations":
        if not isinstance(val, Sequence) or isinstance(val, str):
            raise ValueError("stations.locations must be a list of nat_abbr strings.")
        return ["-i", "nat_abbr," + ",".join(str(v) for v in val)]
    if key == "bbox":
        if len(val) != 4:
            raise ValueError("stations.bbox must be [minlat, maxlat, minlon, maxlon].")
        return ["-l", ",".join(str(v) for v in val)]
    raise AssertionError("unreachable")


def _run(argv: list[str], env: dict[str, str], timeout_s: int) -> str:
    """Run jretrieve once; return stdout text. Raises JretrieveError on failure."""
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise JretrieveError(f"jretrieve timed out after {timeout_s}s: {' '.join(argv)}") from e

    if proc.returncode != 0:
        raise JretrieveError(
            f"jretrieve exited with {proc.returncode}\n"
            f"argv: {argv}\n"
            f"stderr: {proc.stderr.strip()}\n"
            f"stdout (head): {proc.stdout[:500]}"
        )
    if proc.stdout.lstrip().startswith("ERROR"):
        raise JretrieveError(f"jretrieve returned error: {proc.stdout.strip()[:500]}")
    return proc.stdout


def _run_with_retry(argv: list[str], env: dict[str, str], timeout_s: int, attempts: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _run(argv, env=env, timeout_s=timeout_s)
        except JretrieveError as e:
            last_err = e
            if attempt == attempts:
                break
            backoff = 2**attempt
            LOG.warning("jretrieve attempt %d/%d failed (%s); retrying in %ds", attempt, attempts, e, backoff)
            time.sleep(backoff)
    assert last_err is not None
    raise last_err


def _parse_csv(csv_text: str) -> pd.DataFrame:
    csv_text = csv_text.strip()
    if not csv_text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(csv_text), sep=";")


CATALOG_TIME_RANGE_START = datetime(1900, 1, 1)
CATALOG_TIME_RANGE_END = datetime(2100, 12, 31, 23, 59)


def fetch_meta(
    *,
    stations: dict[str, Any],
    params: list[str],
    seq_type: str = "surface",
    stage: str = "prod",
    meta_fields: Sequence[str] = DEFAULT_META_FIELDS,
    timeout_s: int = 300,
) -> pd.DataFrame:
    """Fetch the station catalog (rows per station × parameter × operating period).

    Uses a fixed wide time range so the response is deterministic regardless
    of when (or where) the call runs — important for parallel workers that all
    need the same canonical station list.

    Returns a DataFrame with columns: station (int stationId), latitude, longitude,
    elev, stn_name, nat_abbr, parameter, op_since, op_till.
    """
    if not params:
        raise ValueError("params must be non-empty.")
    binary = _resolve_binary()
    env = _build_env(stage)

    argv = [
        binary,
        "-s", seq_type,
        "-n", ",".join(params),
        "-t", f"{_fmt_time(CATALOG_TIME_RANGE_START)},{_fmt_time(CATALOG_TIME_RANGE_END)}",
        "--meta-info", ",".join(meta_fields),
        "--format", "csv",
        *_stations_to_argv(stations),
    ]
    LOG.info("jretrieve meta: %s", " ".join(argv))
    text = _run_with_retry(argv, env=env, timeout_s=timeout_s)
    df = _parse_csv(text)
    if df.empty:
        raise JretrieveError("jretrieve meta-info returned no rows.")
    return df


def fetch_data(
    *,
    stations: dict[str, Any],
    params: list[str],
    start: datetime,
    end: datetime,
    increment_minutes: int,
    seq_type: str = "surface",
    stage: str = "prod",
    timeout_s: int = 600,
) -> pd.DataFrame:
    """Fetch observation data for the given selection / time range.

    Returns a DataFrame with columns: station (int), termin (str YYYYMMDDhhmmss),
    plus one column per requested parameter.
    """
    if not params:
        raise ValueError("params must be non-empty.")
    binary = _resolve_binary()
    env = _build_env(stage)

    argv = [
        binary,
        "-s", seq_type,
        "-n", ",".join(params),
        "-t", f"{_fmt_time(start)},{_fmt_time(end)},{int(increment_minutes)}",
        "--format", "csv",
        *_stations_to_argv(stations),
    ]
    LOG.info("jretrieve data: %s", " ".join(argv))
    text = _run_with_retry(argv, env=env, timeout_s=timeout_s)
    df = _parse_csv(text)
    return df
