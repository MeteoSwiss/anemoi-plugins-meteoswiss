"""Plugins for anemoi-transform.

Filters are imported lazily, on first attribute access, so that each entry point only
imports its own module: a filter whose dependencies are missing or broken (e.g. an
optional extra that is not installed) does not break the other filters.
"""

import importlib
from typing import Any

_FILTERS = {
    "AverageFluxToCumulativeQuantity": "time_processing",
    "AssignGrid": "grid",
    "CleanObservation": "clean_observation",
    "CopyPrognosticFromForecaster": "copy_prognostic_from_forecaster",
    "ModelToPressureLevel": "vertical_interpolation",
    "GeopotentialFromHeight": "geopotential_from_height",
    "NudgeTowardObservation": "nudging",
    "RetrieveObservation": "retrieve_observation",
    "IconRemapToRegLatLon": "iconremap",
    "GaussianSmoother": "smoothing",
    "SurfaceDiagnostics": "surface_diagnostics",
    "Keep": "keep",
}

__all__ = list(_FILTERS)


def __getattr__(name: str) -> Any:
    if name not in _FILTERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_FILTERS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
