"""Titanlib-free variant of CleanObservation.

Subclasses CleanObservation and overrides only _spatial_fns to route spatial
QC calls through the pure-Python implementations in
clean_observation_tests_standalone.py.

DWH_flag, hard_test, and plateau_test do not call titanlib internally and
are reused from clean_observation_tests.py (imported after titanlib is
stubbed at module level).

QC configuration is loaded from clean_observation_config.yaml by
CleanObservation (_load_qc_config) when clean_observation is imported.

Usage (via run_clean_observation.py --standalone):
    python run_clean_observation.py --ref-time 202406011200 --standalone
"""
import sys
from pathlib import Path
from types import ModuleType as _MT

_FILTERS_DIR = str(Path(__file__).parent)
if _FILTERS_DIR not in sys.path:
    sys.path.insert(0, _FILTERS_DIR)


def _stub(name: str) -> _MT:
    """Register a dummy module under *name* if absent; return it."""
    if name not in sys.modules:
        sys.modules[name] = _MT(name)
    return sys.modules[name]


# Must be stubbed before clean_observation_tests is imported.
_stub("titanlib")

try:
    from anemoi.transform.filter import Filter as _Filter  # noqa: F401
except ImportError:
    class _Filter:
        def __init__(self):
            pass
    _stub("anemoi.transform.filter").Filter = _Filter
    sys.modules.setdefault("anemoi", _MT("anemoi"))
    sys.modules.setdefault("anemoi.transform", _MT("anemoi.transform"))

try:
    import earthkit.data as _ekd
    if not hasattr(_ekd, "FieldList"):
        _ekd.FieldList = object
except ImportError:
    _ekd = _stub("earthkit.data")
    _ekd.FieldList = object
    sys.modules.setdefault("earthkit", _MT("earthkit"))


from clean_observation import CleanObservation  # noqa: E402
from clean_observation_tests_standalone import (  # noqa: E402
    buddy_check_py as _buddy_check,
    first_guess_test_py as _first_guess_test,
    isolation_check_py as _isolation_check,
    sct_dual_py as _spacial_ct_dual,
    sct_resistant_py as _spacial_ct_resistant,
)


class CleanObservationStandalone(CleanObservation):
    """Titanlib-free variant of CleanObservation.

    Inherits all I/O, model-interpolation, blacklisting, and test-dispatch
    logic from CleanObservation.  Only ``_spatial_fns`` is overridden to
    supply the pure-Python spatial QC functions from
    clean_observation_tests_standalone.py instead of the titanlib wrappers.

    Parameters
    ----------
    Same as CleanObservation (obs_path_in, obs_path_out, model_grib_path,
    model_interp).
    """

    @classmethod
    def _spatial_fns(cls):
        return {
            "buddy_check":          _buddy_check,
            "first_guess_test":     _first_guess_test,
            "isolation_check":      _isolation_check,
            "spacial_ct_dual":      _spacial_ct_dual,
            "spacial_ct_resistant": _spacial_ct_resistant,
        }
