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
``EcmwfOpenDataSource._guaranteed_run``/``_run_and_step``. Everything else
(regridding, geopotential-height conversion, parameter renaming) is inherited
unchanged from ``OpenDataInputPlugin``.

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
from datetime import datetime
from datetime import timedelta
from typing import Any

from anemoi.inference.inputs import input_registry
from anemoi.inference.types import Date
from anemoi.plugins.ecmwf.inference.opendata.opendata import OpenDataInputPlugin
from anemoi.plugins.ecmwf.inference.opendata.opendata import retrieve as _retrieve_opendata

LOG = logging.getLogger(__name__)


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
                patch_request=lambda r, step=step: {**r, "step": step},
            )
            if not requests:
                raise ValueError(f"No requests for {variables} ({target})")

            batch = _retrieve_opendata(requests, patch=self.patch_data_request, **kwargs)
            result = batch if result is None else result + batch

        return result
