"""Shared helpers for the operational ICON-CH1 grid: cell geometry and cropping.

Used by both the ``grid-nearest`` filter and the ``icon-static`` source so they
produce an identical, identically-ordered cell axis (a prerequisite for anemoi to
align them on the same grid when joined).
"""

from __future__ import annotations

import numpy as np
from anemoi.transform.grids import grid_registry
from anemoi.transform.spatial import cropping_mask

# Operational ICON-CH1 (R19B08) grid and its external-parameter (extpar) file.
# The grid file carries only geometry (clat/clon/topology); static fields such as
# the land-sea mask live in the extpar file, in the same cell order.
ICON_GRID_PATH = (
    "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/"
    "icon_grid_0001_R19B08_mch.nc"
)
ICON_EXTPAR_PATH = (
    "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/"
    "external_parameter_icon_grid_0001_R19B08_mch.nc"
)

# Bounding box around Switzerland, [west, east, south, north] in degrees. Matches
# CH_EXTENT in scripts/plot_station_map.py. Cropping to this box reduces the
# ~1.15M-cell ICON-CH1 computational domain to ~90k cells over Switzerland.
CH_EXTENT: list[float] = [5.7, 10.6, 45.7, 47.95]


def icon_latlon(grid_definition: dict) -> tuple[np.ndarray, np.ndarray]:
    """Full ICON cell-centre latitudes/longitudes (degrees) for a grid config.

    ``grid_definition`` is the usual ``{"icon": {"path": ...}}`` mapping resolved
    by ``grid_registry``.
    """
    lat, lon = grid_registry.from_config(grid_definition).latlon()
    return np.asarray(lat), np.asarray(lon)


def crop_mask(lat: np.ndarray, lon: np.ndarray, crop: list[float] | None) -> np.ndarray:
    """Boolean mask of cells inside ``crop`` = ``[west, east, south, north]``."""
    if crop is None:
        return np.ones(len(lat), dtype=bool)
    west, east, south, north = crop
    return cropping_mask(lat, lon, north=north, west=west, south=south, east=east)


def cropped_icon_grid(
    grid_definition: dict, crop: list[float] | None = CH_EXTENT
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lat, lon, mask)`` for the ICON cells kept by ``crop``.

    ``lat``/``lon`` (degrees) are the kept cell centres; ``mask`` is the boolean
    selection over the full grid, so other per-cell arrays read from the extpar
    file can be subset in the same order.
    """
    lat, lon = icon_latlon(grid_definition)
    mask = crop_mask(lat, lon, crop)
    return lat[mask], lon[mask], mask
