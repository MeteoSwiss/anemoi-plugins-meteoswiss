"""anemoi-transform filter that, at the overlap lead times of a forecaster +
temporal-downscaler run, replaces the downscaler's prognostic fields with the
parent forecaster's values.

Usage
-----
Register as an anemoi-inference output post-processor, once per output stream::

    output:
      tee:
      - grib:
          post_processors:
          - extract_mask: {...}
          - forward_transform_filter:
              copy-prognostic-from-forecaster:
                forecaster_path: forecaster/20*        # this stream's forecaster GRIB
                common_leadtime: 6h                    # forecaster stride (a duration)
                params_to_keep: [tp]                   # Alternatively, params_to_copy: [2t, sp, ...]. Supports glob pattenrs like t_*.
                namer: *id001                          # same namer used to read the input (optional,not needed if only using ifs names)

The filter is called once per output step. At a step whose lead time is a
positive multiple of ``common_leadtime`` it loads (once, cached) the forecaster
GRIB, matches fields by anemoi variable name + lead time, and swaps the values
in; other steps pass through untouched.

Notes
-----
- ``forecaster_path`` is a plain glob resolved from the inference working
  directory, exactly like the input source paths, the filter makes no
  assumption about any particular directory layout.
- The value copy assumes the downscaler and forecaster share the same grid for a
  given stream If this is failing, double check the level and position at which the postprocessor is applied
  (e.g in case of regriddings, make sure it's applied when the grids still align).
"""

import fnmatch
import glob
import logging
import math
from datetime import timedelta
from functools import cached_property

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)


def _parse_duration(value) -> timedelta:
    """Parse a duration like ``6h``, ``30m``/``30min``, ``1d`` into a timedelta."""
    s = str(value).strip().lower()

    suffix_unit_mapping = (
        ("min", "minutes"),
        ("h", "hours"),
        ("m", "minutes"),
        ("d", "days"),
    )

    for suffix, unit in suffix_unit_mapping:
        if s.endswith(suffix):
            return timedelta(**{unit: float(s[: -len(suffix)])})
    raise ValueError(
        f"common_leadtime must be a duration with a unit (e.g. '6h', '30min', '1d'), got {value!r}. "
        f"Accepted suffixes are: {', '.join(suffix for suffix, _ in suffix_unit_mapping)}"
    )


def _namer_map(namer: dict | None) -> dict[str, str]:
    """Build ``{grib shortName -> anemoi-name template}`` from a config ``namer``
    block (``{'rules': [[{'shortName': 'T'}, 't_{level}'], ...]}``)."""
    if not namer:
        return {}
    rules = namer.get("rules", []) if isinstance(namer, dict) else []
    return {
        rule[0]["shortName"]: rule[1]
        for rule in rules
        if isinstance(rule, list) and len(rule) == 2 and "shortName" in rule[0]
    }


def _anemoi_name(short_name: str, level: int, type_of_level: str, namer: dict) -> str:
    """Map a forecaster GRIB field to its anemoi variable name.
    The anemoi variable is the variable "understood" by the model (usually IFS names, which can differ from the GRIB names).

    ICON names (LAM stream) go through the ``namer`` (``T`` -> ``t_{level}``,
    ``T_2M`` -> ``2t``, ...). IFS names (global stream) are not in the namer, so
    they fall back to the IFS convention: surface names are already anemoi names
    (``2t``, ``sp``, ``tp``), pressure-level names become ``{shortName}_{level}``
    (``t`` at 500 -> ``t_500``). ICON keys are upper-case and IFS lower-case, so
    the two never collide.
    """
    template = namer.get(short_name)
    if template is not None:
        return template.format(level=level) if "{level}" in template else template
    return f"{short_name}_{level}" if type_of_level == "isobaricInhPa" else short_name


def _leadtime(field: ekd.Field) -> timedelta | None:
    """Lead time as a timedelta (unit-agnostic match key), or None if unknown.

    Downscaler post-processor fields give ``step`` as a timedelta directly;
    forecaster GRIB fields give an integer ``step``, so derive it from the
    datetimes (``valid_time - base_time``) instead.
    """
    step = field.metadata("step", default=None)
    if isinstance(step, timedelta):
        return step
    try:
        dt = field.datetime()
        return dt["valid_time"] - dt["base_time"]
    except Exception:
        return None


def _num_values(field: ekd.Field) -> int:
    """Number of grid points, via ``.shape`` — a base Field property present on
    both field kinds. (The GRIB ``numberOfValues`` key seem to exist only on the
    GRIB-backed forecaster fields, not the in-memory downscaler fields.)"""
    return int(math.prod(field.shape))


class CopyPrognosticFromForecaster(Filter):
    """Swap prognostic fields for the forecaster's values at the overlap steps."""

    def __init__(
        self,
        *,
        forecaster_path: str,
        common_leadtime: str,
        namer: dict | None = None,
        params_to_copy: list[str] | None = None,
        params_to_keep: list[str] | None = None,
    ):
        """Initialise the filter.

        Parameters
        ----------
        forecaster_path:
            Glob to this stream's forecaster GRIB (as seen from the inference
            working directory), e.g. ``forecaster/20*`` (regional) or
            ``forecaster/ifs*`` (global).
        common_leadtime:
            The forecaster stride as a duration string (e.g. ``6h``). Output
            steps whose lead time is a nonzero multiple of this are the overlap
            steps that get patched.
        namer:
            The config ``namer`` block used to read the forecaster input
            (ICON shortName -> anemoi-name). Pass the same one (e.g. ``*id001``).
            Optional for a pure-IFS stream, where both namings match.
        params_to_copy / params_to_keep:
            Exactly one must be given (anemoi names, glob patterns allowed).
            ``params_to_copy`` = swap only these; ``params_to_keep`` = swap all
            except these (usually the short list, e.g. diagnostics like ``tp``).
            These need to be in anemoi names (``2t``, ``sp``, ``tp``, ``t_500``, ...), not COSMOS/ICON shortNames.
        """
        if (params_to_copy is None) == (params_to_keep is None):
            raise ValueError("Provide exactly one of params_to_copy or params_to_keep.")

        self.forecaster_path = forecaster_path
        self.common_leadtime = _parse_duration(common_leadtime)
        self.params_to_copy = list(params_to_copy) if params_to_copy else None
        self.params_to_keep = list(params_to_keep) if params_to_keep else None
        self._namer = _namer_map(namer)
        LOG.info(
            "[copy-prognostic-from-forecaster] init: forecaster_path=%s common_leadtime=%s copy=%s keep=%s",
            self.forecaster_path,
            self.common_leadtime,
            self.params_to_copy,
            self.params_to_keep,
        )
        super().__init__()

    def _should_copy(self, name: str) -> bool:
        # Glob-match the anemoi variable name against the patterns so users can
        # write e.g. `t_*`/`q_*` instead of every level.
        if self.params_to_copy is not None:
            return any(fnmatch.fnmatchcase(name, p) for p in self.params_to_copy)
        return not any(fnmatch.fnmatchcase(name, p) for p in self.params_to_keep)

    @cached_property
    def forecaster_index(self) -> dict[tuple[str, timedelta], ekd.Field]:
        """The forecaster GRIB indexed by ``(anemoi name, lead time)`` — all lead
        times of this stream/reftime. Loaded lazily on first access and cached
        for the rest of the run (only the overlap steps ever touch it)."""
        files = sorted(glob.glob(self.forecaster_path))
        if not files:
            raise FileNotFoundError(
                f"copy-prognostic-from-forecaster: no forecaster files match "
                f"{self.forecaster_path!r} (cwd={glob.os.getcwd()})"
            )
        index: dict[tuple[str, timedelta], ekd.Field] = {}
        for f in ekd.from_source("file", files):
            name = _anemoi_name(
                f.metadata("shortName"),
                f.metadata("level"),
                f.metadata("typeOfLevel"),
                self._namer,
            )
            lead = _leadtime(f)
            if lead is not None:
                index[(name, lead)] = f
        LOG.info(
            "[copy-prognostic-from-forecaster] indexed %d forecaster fields from %d file(s)",
            len(index),
            len(files),
        )
        return index

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        out = []
        copied = []
        leads_copied: set[timedelta] = set()
        for field in data:  # loop through each variable and look for prognostics to copy over.
            name = field.metadata("param")
            lead = _leadtime(field)
            # Patch a field only if it is a requested prognostic AND sits at an
            # overlap step (lead a positive multiple of the stride)
            if (
                not self._should_copy(name)
                or lead is None
                or lead <= timedelta(0)
                or lead % self.common_leadtime != timedelta(0)
            ):
                out.append(field)
                continue

            # among all forecaster fields (cached), find the one with the same name and lead time.
            match = self.forecaster_index.get((name, lead))
            if match is None:
                LOG.warning(
                    "[copy-prognostic-from-forecaster] no forecaster field for %s "
                    "at lead %s; keeping downscaler value.",
                    name,
                    lead,
                )
                out.append(field)
                continue

            n_ds = _num_values(field)
            n_fc = _num_values(match)
            if n_ds != n_fc:
                raise ValueError(
                    f"copy-prognostic-from-forecaster: grid mismatch for {name} "
                    f"at lead {lead} — downscaler has {n_ds} values, forecaster "
                    f"has {n_fc}. This filter assumes a shared grid (temporal downscaler)."
                    f"Possible cause: the postprocessor is applied at a level where the incoming stream and the data in `forecaster_path` have different number of grid points."
                )
            out.append(new_field_from_numpy(match.to_numpy(), template=field))
            copied.append(name)
            leads_copied.add(lead)

        if copied:
            LOG.info(
                "[copy-prognostic-from-forecaster] lead(s) %s: copied %d field(s): %s",
                sorted(str(x) for x in leads_copied),
                len(copied),
                sorted(copied),
            )
        return new_fieldlist_from_list(out)
