"""Collection of anemoi plugins used at MeteoSwiss."""

from pathlib import Path

import eccodes  # type: ignore
import eccodes_cosmo_resources


def _use_cosmo_grib_definitions() -> None:
    """Make eccodes resolve COSMO/KENDA-CH1 GRIB2 shortNames (``T_2M``, ``HSURF``, ...) correctly,
    process-wide, from this point on.

    eccodes only honors its GRIB definitions path for the *first* GRIB decode done in a process --
    changing it afterward (be it ``ECCODES_DEFINITION_PATH`` or the live
    ``codes_set_definitions_path`` API) has no effect on concept tables it already cached, and
    forcing a reload (``codes_context_delete``) crashes the process if any GRIB handle from
    before the wipe is still alive (verified). So this can't be toggled per input; it has to run
    once, here, at import time, before anything in the process has decoded a single GRIB message.
    """
    restore = eccodes.codes_definition_path()
    # `restore` can itself be a colon-joined list of paths (eccodes always appends
    # "/MEMFS/definitions"), so validate each component separately rather than treating the
    # whole string as one path.
    paths = (
        eccodes_cosmo_resources.get_definitions_path(),
        *(Path(p) for p in restore.split(":")),
    )
    for path in paths:
        if not path.exists() and not str(path).startswith("/MEMFS"):
            raise RuntimeError(f"{path} does not exist")
    eccodes.codes_set_definitions_path(":".join(map(str, paths)))


_use_cosmo_grib_definitions()
