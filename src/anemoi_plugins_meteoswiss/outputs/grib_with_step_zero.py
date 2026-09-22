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


@output_registry.register("grib-with-step-zero")
class GribWithStepZero(GribFileOutput):
    """A drop-in replacement for the "grib" output that additionally writes a
    zero-valued, step=0 GRIB message for accumulated variables, cloning
    metadata from a reference GRIB file.

    Accumulated variables (e.g. total precipitation) have no meaningful value
    before the forecast starts accumulating, so the "grib" output never emits
    a step=0 message for them. Synthesizing one generically is unreliable:
    the standard step encoding logic (``encode_time_processing``) has no
    ``previous_step`` to work from at step=0 and does not produce a correct
    zero-length accumulation window (see
    https://github.com/ecmwf/anemoi-inference/issues/545). Instead, this
    class clones a real, correctly-encoded step=0 reference message per
    variable, writing it directly (bypassing that generic step-encoding
    logic entirely) alongside all the other fields written normally.

    Because the extra message must land in the very same GRIB file as
    everything else this output writes (not a separate file), this has to be
    the output that owns that file, rather than a sibling output pointed at
    the same path.

    Usage
    -----
    Replace the "grib" output with this one::

        output:
            grib-with-step-zero:
                path: outputs/forecaster/{dateTime}_{step:03}.grib
                encoding:
                typeOfGeneratingProcess: 2
                templates:
                samples: resources/templates/templates_index_icon.yaml
                post_processors:
                - extract_from_state: lam_0
                - accumulate_from_start_of_forecast:
                    accumulations: [tp]
                step_zero_template: /path/to/reference/*_000.grib2   # a real step-0 reference message per accumulated var
    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        step_zero_template: str,
        step_zero_accumulations: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the output.

        Parameters
        ----------
        step_zero_template:
            Glob to the reference GRIB file(s) to clone step=0 messages
            from, e.g. the operational reference file for this stream. Must
            contain one message per variable in ``step_zero_accumulations``
            (matched by anemoi variable name / GRIB ``param``), on the same
            grid as this run (checked against the shape of the state and
            rejected otherwise). Distinct from the base output's own
            ``templates:`` param, which resolves templates for every other
            (non step-0-synthesized) message.
        step_zero_accumulations:
            Anemoi variable names to emit a zero field for. Optional: when
            omitted, defaults to ``metadata.accumulations`` (every variable
            the checkpoint itself marks as an accumulation), mirroring the
            ``accumulate_from_start_of_forecast`` post-processor's own
            default. Only set this explicitly to emit a step=0 message for a
            subset of the checkpoint's accumulated variables.
        """
        super().__init__(context, metadata, **kwargs)
        self.step_zero_template = step_zero_template
        self.step_zero_accumulations = (
            step_zero_accumulations if step_zero_accumulations is not None else list(metadata.accumulations)
        )
        LOG.info(
            "[grib-with-step-zero] init: step_zero_template=%s step_zero_accumulations=%s",
            self.step_zero_template,
            self.step_zero_accumulations,
        )

    @cached_property
    def step_zero_template_index(self) -> dict[str, ekd.Field]:
        """The reference GRIB file(s), indexed by GRIB param name."""
        files = sorted(glob.glob(self.step_zero_template))
        if not files:
            raise FileNotFoundError(f"grib-with-step-zero: no template file(s) match {self.step_zero_template!r}")
        index: dict[str, ekd.Field] = {}
        for f in ekd.from_source("file", files):
            name = f.metadata("param")
            index[name] = f
        LOG.info(
            "[grib-with-step-zero] indexed %d template field(s) from %d file(s): %s",
            len(index),
            len(files),
            sorted(index),
        )
        return index

    def _mars_param(self, name: str) -> str:
        """Resolve an anemoi variable name to its GRIB param (mars shortName)."""
        mars = self.metadata.variables_metadata.get(name, {}).get("mars", {})
        return mars.get("param", name)

    def write_initial_state(self, state: State) -> None:
        """Write the initial state as usual, then additionally emit a
        zero-valued message for each configured accumulation variable not
        already present in it."""
        super().write_initial_state(state)

        if not self.write_step_zero:
            return

        self._write_zero_step_messages(state)

    def _write_zero_step_messages(self, state: State) -> None:
        """Write a zero-valued message for each configured accumulation
        variable not already present in the initial state."""
        date = state["date"]

        for name in self.step_zero_accumulations:
            if name in state["fields"]:
                # Already part of the initial conditions, nothing to synthesize.
                continue

            param = self._mars_param(name)
            template = self.step_zero_template_index.get(param)
            if template is None:
                raise KeyError(
                    f"grib-with-step-zero: no template field for {name!r} "
                    f"(param {param!r}) in {self.step_zero_template!r} "
                    f"(available: {sorted(self.step_zero_template_index)})"
                )

            expected_shape = (len(state["latitudes"]),)
            if template.shape != expected_shape:
                raise ValueError(
                    f"grib-with-step-zero: template field for {name!r} (param {param!r}) "
                    f"in {self.step_zero_template!r} has shape {template.shape}, but this "
                    f"run's state has {expected_shape[0]} grid points. The template file "
                    "likely comes from a different domain or resolution."
                )

            values = np.zeros(template.shape, dtype=float)
            self.write_message(
                values,
                template=template,
                date=int(date.strftime("%Y%m%d")),
                time=date.hour * 100 + date.minute,
                step=0,
            )
