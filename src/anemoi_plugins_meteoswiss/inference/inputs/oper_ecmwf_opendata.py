"""anemoi-inference input: ECMWF Open Data with run-availability resilience.

``anemoi-plugins-ecmwf-inference``'s built-in ``opendata`` input requests
data at the exact target valid time using whatever ``step``/``time`` the
checkpoint's training-time MARS provenance (``variables_metadata``) happens
to carry — which, for a model trained on an FDB-archived reanalysis product,
is an artifact of that archive's own ``(date, time, step)`` storage
convention, not a real forecast lead time. Used verbatim against ECMWF Open
Data (which *does* interpret ``step`` as forecast lead time), that would
request a value hours away from the intended one, and has no fallback if the
run exactly at the target time isn't published yet.

This input corrects both: every requested date is treated as a *target valid
time*, and the ``(run, step)`` pair actually requested is computed by
starting from the most recent run confirmed published — via
``ecmwf.opendata.Client.latest()``, which HEAD-checks the real file index
rather than assuming a fixed publication delay — and walking back through up
to ``stored_runs`` older runs if needed to reach the target. Regridding and
the gh<->z geopotential swap are inherited unchanged from
``OpenDataInputPlugin``.

The checkpoint's ``variables_metadata`` also carries KENDA/COSMO parameter
names (``T``, ``U``, ``QV``, ``FI``, ``OMEGA``, ...) rather than ECMWF Open
Data's own (``t``, ``u``, ``q``, ``z``/``gh``, ``w``) — requesting those
verbatim gets rejected outright (``Cannot find index entries matching ...
'param': ['FI', 'QV', 'T', 'U', 'V']``). This is deliberate on the checkpoint
side: the *same* ``variables_metadata.mars`` also drives GRIB output encoding
(``variable.grib_keys`` in ``anemoi-inference``/``anemoi-transform`` reads it
directly), and the operational output templates
(``templates_index_icon.yaml`` etc.) are keyed by these same COSMO/KENDA
names — so ``patch_metadata`` must keep patching them in, for output's sake,
even though that's the wrong convention for this input's own retrieval.

This input fixes that on its own side without any extra config: for each
requested variable (e.g. ``t_500``), it already knows two things —
``variables_metadata[variable]["mars"]["param"]`` (the COSMO name, ``T``) and
the variable's own name (``t_500``, which *is* the ECMWF-side name, by the
same convention the checkpoint's variable schema uses everywhere) — so it
pairs the two directly, per retrieval, with nothing to configure or keep in
sync.

``ECCODES_DEFINITION_PATH``
---------------------------

The operational image sets ``ECCODES_DEFINITION_PATH`` to MeteoSwiss's COSMO
resources so the KENDA/``lam_0`` side decodes GRIB1 COSMO fields correctly.
That override replaces the *core* ``grib2/section.1.def`` (parsed for every
GRIB2 message, not just COSMO-specific ones) to hardcode
``tablesVersion = 33`` (a hidden ``tablesVersionMTG2Switch`` constant)
instead of reading it from the message. Applied to genuine ECMWF Open Data
GRIB2 messages, that desyncs downstream template/table resolution and MIR's
own bundled eccodes (a separate native library from the one used for KENDA
reading) crashes with an unrecoverable C++ abort (``eckit::SeriousBug``,
``codes_get_long(... "7777" ...)`` — a corrupted-looking message, really a
symptom of the wrong tables version cascading into unrelated template
concepts) instead of a catchable Python exception.

This input clears ``ECCODES_DEFINITION_PATH`` for the duration of the
retrieve+regrid call only (see ``_without_eccodes_definition_path_override``)
so MIR's own eccodes falls back to its bundled standard definitions, while
KENDA-side reading (elsewhere in the same run, through the *other* eccodes
library) is unaffected.

Optional dependency
--------------------

This module (and the ``anemoi-inference``/``anemoi-plugins-ecmwf-inference``
packages it needs) is only pulled in via the ``oper-ecmwf-opendata`` extra
(``pip install anemoi-plugins-meteoswiss[oper-ecmwf-opendata]``) — those
bring in ``mir-python``/``eckit``/``atlas`` and are unnecessary for consumers
that only use this package's ``anemoi-transform`` filters (e.g. dataset
creation).

Usage
-----

::

    input:
      cutout:
        - global:
            oper-ecmwf-opendata:
              frequency_h: 6        # optional, defaults shown here
              step_h: 3
              max_lead_time_h: 144
              stored_runs: 12
"""

import logging
import math
import os
import re
from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from typing import Any

from anemoi.inference.inputs import input_registry
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import Date
from anemoi.plugins.ecmwf.inference.opendata.opendata import OpenDataInputPlugin
from anemoi.plugins.ecmwf.inference.opendata.opendata import retrieve as _retrieve_opendata
from ecmwf.opendata import Client as _EcmwfOpenDataClient

LOG = logging.getLogger(__name__)


@contextmanager
def _without_eccodes_definition_path_override():
    """Temporarily clear ``ECCODES_DEFINITION_PATH`` (see module docstring).

    MIR's own bundled eccodes is a separate native library from the one used
    for KENDA/GRIB1 reading elsewhere in the same process, so this only
    affects the call it wraps — not unrelated eccodes usage in the same run.
    """
    original = os.environ.pop("ECCODES_DEFINITION_PATH", None)
    try:
        yield
    finally:
        if original is not None:
            os.environ["ECCODES_DEFINITION_PATH"] = original


def _latest_published_run(**params: Any) -> datetime:
    """Ask the real ECMWF Open Data catalog for the most recent published run.

    ``Client.latest()`` HEAD-checks the actual file index (walking back in
    6-hourly steps, up to 2 days) rather than assuming a fixed publication
    delay — ``source``/``model`` default to ``"ecmwf"``/``"ifs"``, matching
    what ``OpenDataInputPlugin`` itself retrieves from.
    """
    return _EcmwfOpenDataClient().latest(**params)


def _param_translation_from_variables_metadata(metadata: Metadata, variables: list[str]) -> dict[str, str]:
    """Build a KENDA/COSMO-name -> ECMWF-name table for the given variables.

    For each variable, ``variables_metadata[variable]["mars"]["param"]`` is
    the KENDA/COSMO name (e.g. ``T``); the variable's own name with its level
    suffix stripped (e.g. ``t_500`` -> ``t``) is the ECMWF Open Data one.
    """
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


def _translate_params(request: dict, param_translation: dict[str, str]) -> dict:
    """Map ``request["param"]`` from KENDA/COSMO names to ECMWF Open Data ones."""
    param = request.get("param")
    if param is None:
        return request
    if isinstance(param, (list, tuple, set)):
        request["param"] = [param_translation.get(p, p) for p in param]
    else:
        request["param"] = param_translation.get(param, param)
    return request


@input_registry.register("oper-ecmwf-opendata")
class OperEcmwfOpenDataInput(OpenDataInputPlugin):
    """ECMWF Open Data input with run-availability fallback."""

    def __init__(
        self,
        context,
        metadata,
        *,
        frequency_h: int = 6,
        step_h: int = 3,
        max_lead_time_h: int = 144,
        stored_runs: int = 12,
        **kwargs: Any,
    ) -> None:
        """Initialise the OperEcmwfOpenDataInput.

        Parameters
        ----------
        frequency_h : int
            Hours between ECMWF Open Data runs.
        step_h : int
            Step granularity available in ECMWF Open Data.
        max_lead_time_h : int
            Maximum forecast step available in ECMWF Open Data.
        stored_runs : int
            Number of past runs to try walking back through before giving up.
        """
        super().__init__(context, metadata, **kwargs)
        self.frequency_h = frequency_h
        self.step_h = step_h
        self.max_lead_time_h = max_lead_time_h
        self.stored_runs = stored_runs

    def _run_and_step(self, target: datetime, latest_run: datetime) -> tuple[datetime, int]:
        """Compute ``(run, step_h)`` so the forecast from ``run`` reaches ``target``.

        Walks back through stored runs (``frequency_h`` apart, up to
        ``stored_runs``) to find the most recent one whose lead time (rounded
        to the nearest ``step_h`` boundary) covers ``target``.
        """
        hours_ahead = round((latest_run - target).total_seconds() / 3600)
        n_back = math.ceil(max(hours_ahead, 0) / self.frequency_h)
        if n_back > self.stored_runs:
            raise ValueError(f"{target} is older than the {self.stored_runs} stored open data runs")
        run = latest_run - timedelta(hours=self.frequency_h * n_back)
        step = round((target - run).total_seconds() / 3600 / self.step_h) * self.step_h
        if step > self.max_lead_time_h:
            raise ValueError(f"step {step}h exceeds open data max lead time for {target}")
        if step < 0:
            raise ValueError(f"computed a negative step ({step}h) for {target} against run {run}")
        return run, step

    def retrieve(self, variables: list[str], dates: list[Date]) -> Any:
        """Retrieve data for the given variables at the given target valid times."""
        guaranteed_run = _latest_published_run(type="fc")
        param_translation = _param_translation_from_variables_metadata(self.metadata, variables)

        kwargs = self.kwargs.copy()
        kwargs.setdefault("grid", self.metadata.grid)
        kwargs.setdefault("area", self.metadata.area)

        result = None
        for target in dates:
            run, step = self._run_and_step(target, guaranteed_run)
            LOG.info("oper-ecmwf-opendata: %s -> run %s step %dh", target, run, step)

            requests = self.metadata.mars_requests(
                variables=variables,
                dates=[run],
                use_grib_paramid=False,
                type="fc",
                patch_request=lambda r, step=step: _translate_params({**r, "step": step}, param_translation),
            )
            if not requests:
                raise ValueError(f"No requests for {variables} ({target})")

            with _without_eccodes_definition_path_override():
                batch = _retrieve_opendata(requests, patch=self.patch_data_request, **kwargs)
            result = batch if result is None else result + batch

        return result
