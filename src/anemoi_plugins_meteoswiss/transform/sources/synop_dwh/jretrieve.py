"""Subprocess wrapper around `jretrievedwh.py`.

Retrieval auth follows the same model as MeteoSwiss/evalml: a committed
`.jretrievedwh-conf.prod.py` at the repo root mints a short-lived Bearer token
from OAuth client credentials (`JRETRIEVE_CLIENT_ID` / `JRETRIEVE_CLIENT_SECRET`),
supplied either in the environment or in a gitignored `.env` next to the conf.
We point `jretrievedwh.py` at that conf via `JRETRIEVE_CONF_DIR` /
`JRETRIEVE_CONF_NAME`, then invoke the REST client and parse its CSV output into
pandas DataFrames. Only the `prod` stage is supported.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

LOG = logging.getLogger(__name__)

BINARY_NAME = "jretrievedwh.py"
# Operational osm install, used when the binary isn't already on $PATH.
HARDCODED_BINARY_PATH = "/oprusers/osm/opr.inn/bin/jretrievedwh.py"
CONF_NAME = ".jretrievedwh-conf.prod.py"
DEFAULT_META_FIELDS: tuple[str, ...] = ("lat", "lon", "elev", "name", "nat_abbr")
DEFAULT_USE_LIMITATION = 40

CATALOG_TIME_RANGE_START = datetime(1900, 1, 1)
CATALOG_TIME_RANGE_END = datetime(2100, 12, 31, 23, 59)


class JretrieveError(RuntimeError):
    """Raised when jretrievedwh.py fails or returns malformed output."""


class JretrievePermanentError(JretrieveError):
    """A failure that won't improve on retry (e.g. an application-level error
    response for a bad request), so `_run_with_retry` fails fast on it."""


def _resolve_binary() -> str:
    path = shutil.which(BINARY_NAME)
    if path is not None:
        return path
    if os.path.isfile(HARDCODED_BINARY_PATH):
        return HARDCODED_BINARY_PATH
    raise JretrieveError(
        f"{BINARY_NAME} not found on $PATH or at {HARDCODED_BINARY_PATH}."
    )


def _locate_conf_dir() -> Path | None:
    """Find the directory holding the committed jretrieve conf.

    Prefer an explicit `$JRETRIEVE_CONF_DIR` if it actually contains the conf,
    otherwise walk up from the current working directory (entry points `cd` to
    the repo root before running). Returns None if it can't be found so callers
    can report a clear, aggregated error.
    """
    env_dir = os.environ.get("JRETRIEVE_CONF_DIR")
    if env_dir and (Path(env_dir) / CONF_NAME).is_file():
        return Path(env_dir)
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / CONF_NAME).is_file():
            return candidate
    return None


def _build_env(stage: str) -> dict[str, str]:
    if stage != "prod":
        raise ValueError(f"Only 'prod' stage is supported, got {stage!r}.")
    conf_dir = _locate_conf_dir()
    if conf_dir is None:
        raise JretrieveError(
            f"jretrieve conf {CONF_NAME!r} not found. Expected it at the repo "
            f"root (or set $JRETRIEVE_CONF_DIR to the directory holding it)."
        )
    env = os.environ.copy()
    env["JRETRIEVE_CONF_DIR"] = str(conf_dir)
    env["JRETRIEVE_CONF_NAME"] = CONF_NAME
    return env


def _check_credentials(conf_dir: Path) -> str | None:
    """Return a descriptive error string if jretrieve credentials are missing,
    else None. Credentials may come from the environment or a `.env` file next
    to the conf."""
    client_id = os.environ.get("JRETRIEVE_CLIENT_ID")
    client_secret = os.environ.get("JRETRIEVE_CLIENT_SECRET")

    dotenv_path = conf_dir / ".env"
    dotenv_exists = dotenv_path.is_file()

    if not (client_id and client_secret) and dotenv_exists:
        dotenv: dict[str, str] = {}
        try:
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    dotenv[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            pass
        client_id = client_id or dotenv.get("JRETRIEVE_CLIENT_ID")
        client_secret = client_secret or dotenv.get("JRETRIEVE_CLIENT_SECRET")

    if client_id and client_secret:
        return None

    missing = [
        name
        for name, val in (
            ("JRETRIEVE_CLIENT_ID", client_id),
            ("JRETRIEVE_CLIENT_SECRET", client_secret),
        )
        if not val
    ]
    lines = [
        f"Missing jretrieve credentials: {', '.join(missing)}.",
        "Credentials must be supplied in one of two ways:",
        f"  1. Set {' and '.join(missing)} as environment variables.",
        f"  2. Add them to {dotenv_path}",
    ]
    if dotenv_exists:
        lines.append(
            f"     (.env file exists but does not contain {' or '.join(missing)})"
        )
    else:
        lines.append("     (.env file not found — create it with the missing keys)")
    return "\n".join(lines)


def check_prerequisites(stage: str = "prod") -> None:
    """Fail-fast validation that the jretrievedwh environment is usable.

    Checks the binary is reachable, the conf is present, and credentials are
    available. Raises a single `JretrieveError` listing *all* problems found, so
    a misconfigured environment is reported up front rather than hours into a
    build.
    """
    problems: list[str] = []
    if stage != "prod":
        problems.append(f"Only 'prod' stage is supported, got {stage!r}.")
    try:
        _resolve_binary()
    except JretrieveError as e:
        problems.append(str(e))
    conf_dir = _locate_conf_dir()
    if conf_dir is None:
        problems.append(
            f"jretrieve conf {CONF_NAME!r} not found at the repo root "
            f"(or via $JRETRIEVE_CONF_DIR)."
        )
    else:
        cred_problem = _check_credentials(conf_dir)
        if cred_problem:
            problems.append(cred_problem)
    if problems:
        raise JretrieveError(
            "jretrievedwh prerequisites not met:\n  - " + "\n  - ".join(problems)
        )


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def _stations_to_argv(stations: dict[str, Any]) -> list[str]:
    """Translate a `stations:` recipe dict to jretrieve CLI args.

    Exactly one of {group, locations, bbox} must be set. Values may be given as
    lists or comma-separated strings.
    """
    keys = [k for k in ("group", "locations", "bbox") if stations.get(k) is not None]
    if len(keys) != 1:
        raise ValueError(
            f"stations must specify exactly one of group/locations/bbox, got {keys}"
        )
    key = keys[0]
    val = stations[key]

    if key == "group":
        return ["-a", f"stn_group,{val}"]
    if key == "locations":
        if isinstance(val, str):
            val = [v for v in val.split(",") if v]
        if not isinstance(val, Sequence):
            raise ValueError("stations.locations must be a list of nat_abbr strings.")
        return ["-i", "nat_abbr," + ",".join(str(v) for v in val)]
    if key == "bbox":
        if isinstance(val, str):
            val = [v for v in val.split(",") if v]
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
        # Application-level error for the request as posed — retrying the same
        # argv won't help, so surface it immediately.
        raise JretrievePermanentError(
            f"jretrieve returned error: {proc.stdout.strip()[:500]}"
        )
    return proc.stdout


def _run_with_retry(argv: list[str], env: dict[str, str], timeout_s: int, attempts: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _run(argv, env=env, timeout_s=timeout_s)
        except JretrievePermanentError:
            raise  # no point retrying a bad request
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
    use_limitation: int = DEFAULT_USE_LIMITATION,
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
        "--use-limitation", str(use_limitation),
        "--format", "csv",
        *_stations_to_argv(stations),
    ]
    LOG.info("jretrieve data: %s", " ".join(argv))
    text = _run_with_retry(argv, env=env, timeout_s=timeout_s)
    df = _parse_csv(text)
    return df
