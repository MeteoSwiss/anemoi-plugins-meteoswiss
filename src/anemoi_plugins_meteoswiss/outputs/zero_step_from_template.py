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
    step.

    Usage
    -----
    Register as a sibling output next to the "real" grib output, once per output
    stream::

        output:
            tee:
                outputs:
                - grib:
                    path: outputs/forecaster/{dateTime}_{step:03}.grib
                    encoding:
                    typeOfGeneratingProcess: 2
                    templates:
                    samples: resources/templates/templates_index_icon.yaml
                    post_processors:
                    - extract_from_state: lam_0
                    - accumulate_from_start_of_forecast:
                        accumulations: [tp]
                - zero-step-from-template:
                    path: outputs/forecaster/{dateTime}_000.grib
                    template_path: /path/to/reference/*_000.grib2   # a real step-0 reference message per accumulated var
                    accumulations: [tp]
                    write_initial_state: true
    """

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
