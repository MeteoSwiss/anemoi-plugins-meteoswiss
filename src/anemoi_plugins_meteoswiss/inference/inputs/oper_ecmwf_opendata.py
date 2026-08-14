"""anemoi-inference input: ECMWF Open Data with run-availability resilience.

The built-in ``opendata`` input uses the checkpoint's training-time MARS
step/time verbatim — an archive storage artifact, not a real forecast lead
time — which is wrong against ECMWF Open Data and has no fallback if a run
isn't published yet. This input treats every date as a target valid time,
finds the latest published run via ``ecmwf.opendata.Client.latest()``, and
walks back up to ``STORED_RUNS`` runs to hit it exactly.

It also translates the checkpoint's KENDA/COSMO parameter names (``T``,
``U``, ...) to ECMWF's own (``t``, ``u``, ...) per variable, since requesting
COSMO names against ECMWF Open Data fails outright. And it clears
``ECCODES_DEFINITION_PATH`` for the retrieve+regrid call only, so the
operational image's COSMO eccodes override doesn't crash MIR on genuine
ECMWF GRIB2 (see ``_without_eccodes_definition_path_override``).

Only pulled in via the ``oper-inference`` extra.

Usage
-----

::

    input:
      cutout:
        - global:
            oper-ecmwf-opendata:
              allow_forecast_fallback: false   # optional, default shown here
              constant: false                  # optional, default shown here -- set for orography/lsm/... sources
              cache_dir: /path/to/cache        # optional, default: none (always re-download)
"""

import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from typing import Any

import earthkit.data as ekd
from anemoi.inference.inputs import input_registry
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import Date
from anemoi.plugins.ecmwf.inference.opendata.opendata import OpenDataInputPlugin
from anemoi.plugins.ecmwf.inference.opendata.opendata import retrieve as _retrieve_opendata
from anemoi.transform.fields import NewMetadataField
from ecmwf.opendata import Client as _EcmwfOpenDataClient

LOG = logging.getLogger(__name__)

# ECMWF Open Data facts, not deployment config -- a wrong override would
# silently compute a different, wrong valid time in `_resolve_init_time_and_lead_hours`.
FREQUENCY_H = 6  # hours between runs
STEP_H = 3  # step granularity; 0/3/6h exist, 1/2h don't (verified via HEAD requests)
MAX_LEAD_TIME_H = 144  # max published forecast step
STORED_RUNS = 12  # runs to walk back through before giving up


@contextmanager
def _without_eccodes_definition_path_override():
    """Temporarily clear ``ECCODES_DEFINITION_PATH`` so MIR's own eccodes is unaffected (see module docstring)."""
    original = os.environ.pop("ECCODES_DEFINITION_PATH", None)
    try:
        yield
    finally:
        if original is not None:
            os.environ["ECCODES_DEFINITION_PATH"] = original


@contextmanager
def _with_cache_dir(cache_dir: str | None):
    """Persist ``ekd.from_source("ecmwf-open-data", ...)`` downloads under ``cache_dir`` for the
    duration of the wrapped call, restoring earthkit-data's ambient cache config on exit. A no-op
    if ``cache_dir`` is unset -- downloads then go through earthkit-data's default (uncached) policy."""
    if not cache_dir:
        yield
        return
    with ekd.config.temporary(
        {
            "cache-policy": "user",
            "user-cache-directory": cache_dir,
            "maximum-cache-disk-usage": "98%",
        }
    ):
        yield


def _latest_published_run(**params: Any) -> datetime:
    """Return the most recent published run, HEAD-checked via ``Client.latest()``."""
    return _EcmwfOpenDataClient().latest(**params)


def _param_translation_from_variables_metadata(metadata: Metadata, variables: list[str]) -> dict[str, str]:
    """Map each variable's COSMO ``mars`` param (e.g. ``T``) to its ECMWF name (e.g. ``t``)."""
    mapping = {}
    for variable in variables:
        mars = metadata.variables_metadata.get(variable, {}).get("mars", {})
        cosmo_name = mars.get("param")
        if cosmo_name is None:
            continue
        # Pressure-level variable names are "{ecmwf_param}_{level}" (e.g. "t_500");
        # single-level ones (msl, 2t, 10u, ...) have no such suffix.
        mapping[cosmo_name] = re.compile(r"_\d+$").sub("", variable)
    return mapping


def _drop_null_levelist(request: dict) -> dict:
    """Remove a ``levelist`` key that's ``None`` (surface fields have none).

    ``ecmwf.opendata.Client`` stringifies every request value via ``str(x)`` before matching
    it against the remote index, turning a Python ``None`` into the literal ``"None"``; the
    index has no ``levelist`` entry at all for surface fields, so nothing matches and the
    request fails with "Cannot find index entries".
    """
    levelist = request.get("levelist")
    if levelist is None or (isinstance(levelist, (list, tuple, set)) and all(v is None for v in levelist)):
        request = {k: v for k, v in request.items() if k != "levelist"}
    return request


def _cosmo_to_ecmwf_request_param(request: dict, cosmo_to_ecmwf: dict[str, str]) -> dict:
    """Translate the outgoing MARS request's ``param`` from COSMO names to ECMWF ones."""
    param = request.get("param")
    if param is None:
        return request
    if isinstance(param, (list, tuple, set)):
        request["param"] = [cosmo_to_ecmwf.get(p, p) for p in param]
    else:
        request["param"] = cosmo_to_ecmwf.get(param, param)
    return request


def _cosmo_to_ecmwf_field_param(fields: ekd.FieldList, cosmo_to_ecmwf: dict[str, str]) -> ekd.FieldList:
    """Same COSMO -> ECMWF mapping, applied to retrieved fields whose ``param`` still decoded
    as the COSMO name (plus ``GH`` -> ``gh``)."""
    translation = {**cosmo_to_ecmwf, "GH": "gh"}
    return ekd.SimpleFieldList(
        [
            field
            if (true_param := translation.get(field.metadata("param", default=None))) is None
            else NewMetadataField(field, param=true_param)
            for field in fields
        ]
    )


@input_registry.register("oper-ecmwf-opendata")
class OperEcmwfOpenDataInput(OpenDataInputPlugin):
    """ECMWF Open Data input with run-availability fallback; no analysis product, so step 0 is the closest equivalent."""

    def __init__(
        self,
        context,
        metadata,
        *,
        allow_forecast_fallback: bool = False,
        constant: bool = False,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """If ``allow_forecast_fallback`` is ``False`` (default), only step-0 fields are requested; an off-grid target raises.

        ``constant``, if ``True``, always fetches step-0 fields from the latest published run

        ``cache_dir``, if set, persists downloaded ECMWF Open Data fields to disk (via earthkit-data's
        own cache) so later runs don't re-download them."""
        super().__init__(context, metadata, **kwargs)
        self.allow_forecast_fallback = allow_forecast_fallback
        self.constant = constant
        self.cache_dir = cache_dir

    def _resolve_init_time_and_lead_hours(self, target: datetime, latest_init_time: datetime) -> tuple[datetime, int]:
        """Compute ``(init_time, lead_hours)`` reaching ``target`` exactly; raises rather than rounding when off-grid."""
        hour = timedelta(hours=1)
        offset = latest_init_time - target
        if offset % hour:
            raise ValueError(f"{target} is not hour-aligned; ECMWF Open Data only publishes on whole hours")
        hours_ahead = offset // hour
        n_back = -(-max(hours_ahead, 0) // FREQUENCY_H)  # ceiling division
        if n_back > STORED_RUNS:
            raise ValueError(f"{target} is older than the {STORED_RUNS} stored ECMWF Open Data runs")
        init_time = latest_init_time - timedelta(hours=FREQUENCY_H * n_back)
        lead_hours = (target - init_time) // hour
        if lead_hours < 0:
            raise ValueError(f"{target} is more recent than the latest published run {latest_init_time}")
        if lead_hours % STEP_H != 0:
            raise ValueError(
                f"{target} is {lead_hours}h after run {init_time}, not a multiple of the {STEP_H}h step "
                "ECMWF Open Data publishes"
            )
        if lead_hours > MAX_LEAD_TIME_H:
            raise ValueError(
                f"step {lead_hours}h for {target} exceeds ECMWF Open Data's {MAX_LEAD_TIME_H}h max lead time"
            )
        if lead_hours != 0 and not self.allow_forecast_fallback:
            raise ValueError(
                f"{target} requires a {lead_hours}h forecast step from run {init_time}, but "
                "allow_forecast_fallback=False restricts this input to step-0 (analysis-equivalent) fields"
            )
        return init_time, lead_hours

    def retrieve(self, variables: list[str], dates: list[Date]) -> Any:
        """Retrieve data for the given variables at the given target valid times."""
        guaranteed_init_time = _latest_published_run(type="fc")
        cosmo_to_ecmwf = _param_translation_from_variables_metadata(self.metadata, variables)

        kwargs = self.kwargs.copy()
        kwargs.setdefault("grid", self.metadata.grid)
        kwargs.setdefault("area", self.metadata.area)

        result = ekd.FieldList()
        for target in dates:
            if self.constant:
                init_time, lead_hours = guaranteed_init_time, 0
            else:
                init_time, lead_hours = self._resolve_init_time_and_lead_hours(target, guaranteed_init_time)
            LOG.info(
                "oper-ecmwf-opendata: %s -> run %s step %dh",
                target,
                init_time,
                lead_hours,
            )

            requests = self.metadata.mars_requests(
                variables=variables,
                dates=[init_time],
                use_grib_paramid=False,
                type="fc",
                patch_request=lambda r, lead_hours=lead_hours: _cosmo_to_ecmwf_request_param(
                    _drop_null_levelist({**r, "step": lead_hours}), cosmo_to_ecmwf
                ),
            )
            if not requests:
                raise ValueError(f"No requests for {variables} ({target})")

            with (
                _without_eccodes_definition_path_override(),
                _with_cache_dir(self.cache_dir),
            ):
                batch = _retrieve_opendata(requests, patch=self.patch_data_request, **kwargs)

            if self.constant:
                overrides = {
                    "date": int(target.strftime("%Y%m%d")),
                    "time": int(target.strftime("%H%M")),
                    "step": 0,
                }
                batch = ekd.SimpleFieldList(
                    [field.clone(metadata=field.metadata().override(**overrides)) for field in batch]
                )

            result += batch

        return _cosmo_to_ecmwf_field_param(result, cosmo_to_ecmwf)
