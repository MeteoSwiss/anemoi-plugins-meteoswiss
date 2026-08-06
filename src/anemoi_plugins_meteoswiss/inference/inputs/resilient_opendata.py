"""anemoi-inference input: ECMWF Open Data with delivery-delay resilience.

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
starting from the most recent run assumed already published
(``delivery_delay_h`` in the past) and walking back through up to
``stored_runs`` older runs if needed to reach the target — mirroring
runml-preprocessor's
``EcmwfOpenDataSource._guaranteed_run``/``_run_and_step``. Regridding and the
gh<->z geopotential swap are inherited unchanged from ``OpenDataInputPlugin``.

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

Usage
-----

::

    input:
      cutout:
        - global:
            resilient-opendata:
              delivery_delay_h: 7   # optional, defaults shown here
              frequency_h: 6
              step_h: 3
              max_lead_time_h: 144
              stored_runs: 12
"""

import logging
import math
import re
from datetime import datetime
from datetime import timedelta
from typing import Any

from anemoi.inference.inputs import input_registry
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import Date
from anemoi.plugins.ecmwf.inference.opendata.opendata import OpenDataInputPlugin
from anemoi.plugins.ecmwf.inference.opendata.opendata import retrieve as _retrieve_opendata

LOG = logging.getLogger(__name__)

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


@input_registry.register("resilient-opendata")
class ResilientOpenDataInput(OpenDataInputPlugin):
    """ECMWF Open Data input with delivery-delay/run-availability fallback."""

    def __init__(
        self,
        context,
        metadata,
        *,
        frequency_h: int = 6,
        step_h: int = 3,
        max_lead_time_h: int = 144,
        stored_runs: int = 12,
        delivery_delay_h: int = 7,
        **kwargs: Any,
    ) -> None:
        """Initialise the ResilientOpenDataInput.

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
        delivery_delay_h : int
            Assumed publication delay, used to pick the run considered
            "guaranteed published" as the walk-back's starting point.
        """
        super().__init__(context, metadata, **kwargs)
        self.frequency_h = frequency_h
        self.step_h = step_h
        self.max_lead_time_h = max_lead_time_h
        self.stored_runs = stored_runs
        self.delivery_delay_h = delivery_delay_h

    def _guaranteed_run(self, now: datetime | None = None) -> datetime:
        """Latest run boundary guaranteed published given the delivery delay."""
        available_since = (now or datetime.utcnow()) - timedelta(hours=self.delivery_delay_h)
        run_hour = (available_since.hour // self.frequency_h) * self.frequency_h
        return available_since.replace(hour=run_hour, minute=0, second=0, microsecond=0)

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
        guaranteed_run = self._guaranteed_run()
        param_translation = _param_translation_from_variables_metadata(self.metadata, variables)

        kwargs = self.kwargs.copy()
        kwargs.setdefault("grid", self.metadata.grid)
        kwargs.setdefault("area", self.metadata.area)

        result = None
        for target in dates:
            run, step = self._run_and_step(target, guaranteed_run)
            LOG.info("resilient-opendata: %s -> run %s step %dh", target, run, step)

            requests = self.metadata.mars_requests(
                variables=variables,
                dates=[run],
                use_grib_paramid=False,
                type="fc",
                patch_request=lambda r, step=step: _translate_params({**r, "step": step}, param_translation),
            )
            if not requests:
                raise ValueError(f"No requests for {variables} ({target})")

            batch = _retrieve_opendata(requests, patch=self.patch_data_request, **kwargs)
            result = batch if result is None else result + batch

        return result
