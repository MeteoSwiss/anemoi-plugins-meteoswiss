"""anemoi-inference output that emits a zero-valued field at step=0 for
variables accumulated from the start of the forecast (see the
``accumulate_from_start_of_forecast`` post-processor), by cloning the GRIB
metadata of a real reference message rather than re-deriving it.

Why this exists
----------------
``Accumulate`` only ever updates fields already present in the state, so an
accumulated variable (e.g. ``tp``) is never part of the initial state and no
step-0 message is written. Synthesising it through the normal
``write_step``/``grib_keys()`` path forces a zero-length accumulation window
(``startStep == endStep == 0``) to be encoded from scratch for a variable
that has never appeared in the state before, which is fragile to get exactly
right for downstream consumers (see anemoi-inference PR #546, discarded for
this reason).

Instead, this output clones an existing, known-good step-0 GRIB message (the
operational reference file for this stream) and overrides only the
run-specific keys (``date``/``time``/``step``). Every other key -- edition,
packing, table version, product definition template, quantile/percentile
keys, etc. -- is inherited byte-for-byte from the template, via
``GribWriter.write()`` / ``encode_message()``.

Usage
-----
Register as a sibling output next to the "real" grib output, once per output
stream::

    output:
      tee:
      - grib:
          path: ninjo_icon-1e_ctrl_..._alp_{step}.grb2
          write_initial_state: false
          ...
      - zero-step-from-template:
          path: ninjo_icon-1e_ctrl_..._alp_000.grb2
          template_path: /opr/osm/inn/wd/*_640/grib/ninjo_icon-1e_ctrl_*_alp_000.grb2
          accumulations: [tp]   # optional, defaults to metadata.accumulations
"""

import glob
import logging
from functools import cached_property
from typing import Any

import earthkit.data as ekd
import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.outputs import output_registry
from anemoi.inference.outputs.gribfile import GribFileOutput
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)


@output_registry.register("zero-step-from-template")
class ZeroStepFromTemplate(GribFileOutput):
    """Write a zero-valued, step=0 GRIB message for accumulated variables,
    cloning metadata from a reference GRIB file. Writes nothing at any other
    step."""

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        template_path: str,
        accumulations: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the output.

        Parameters
        ----------
        template_path:
            Glob to the reference GRIB file(s) to clone step=0 messages
            from, e.g. the operational reference file for this stream. Must
            contain one message per variable in ``accumulations`` (matched
            by anemoi variable name / GRIB ``param``).
        accumulations:
            Anemoi variable names to emit a zero field for. Defaults to
            ``metadata.accumulations`` (the variables the checkpoint marks
            as accumulations), mirroring the ``accumulate_from_start_of_forecast``
            post-processor's own default.
        """
        super().__init__(context, metadata, **kwargs)
        self.template_path = template_path
        self.accumulations = accumulations if accumulations is not None else list(metadata.accumulations)
        LOG.info(
            "[zero-step-from-template] init: template_path=%s accumulations=%s",
            self.template_path,
            self.accumulations,
        )

    @cached_property
    def template_index(self) -> dict[str, ekd.Field]:
        """The reference GRIB file(s), indexed by anemoi variable name."""
        files = sorted(glob.glob(self.template_path))
        if not files:
            raise FileNotFoundError(
                f"zero-step-from-template: no template file(s) match {self.template_path!r}"
            )
        index: dict[str, ekd.Field] = {}
        for f in ekd.from_source("file", files):
            name = f.metadata("param")
            index[name] = f
        LOG.info(
            "[zero-step-from-template] indexed %d template field(s) from %d file(s): %s",
            len(index),
            len(files),
            sorted(index),
        )
        return index

    def write_initial_state(self, state: State) -> None:
        """Write a zero-valued message for each configured accumulation
        variable not already present in the initial state."""
        if not self.write_step_zero:
            return

        date = state["date"]

        for name in self.accumulations:
            if name in state["fields"]:
                # Already part of the initial conditions, nothing to synthesize.
                continue

            template = self.template_index.get(name)
            if template is None:
                raise KeyError(
                    f"zero-step-from-template: no template field for {name!r} in "
                    f"{self.template_path!r} (available: {sorted(self.template_index)})"
                )

            values = np.zeros(template.shape, dtype=float)
            self.write_message(
                values,
                template=template,
                date=int(date.strftime("%Y%m%d")),
                time=date.hour * 100 + date.minute,
                step=0,
            )

    def write_step(self, state: State) -> None:
        """No-op: this output only ever emits the step=0 message(s)."""
