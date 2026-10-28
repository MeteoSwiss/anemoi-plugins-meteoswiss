"""Project point observations onto the (cropped) ICON-CH1 grid by nearest neighbour."""

from __future__ import annotations

import logging

import numpy as np
from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.spatial import latlon_to_xyz
from scipy.spatial import KDTree

from ..icon_grid import CH_EXTENT
from ..icon_grid import cropped_icon_grid

LOG = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


class GridNearest(Filter):
    """Scatter each input field's point values onto their nearest ICON-CH1 cell.

    Input is a gridded FieldList with one value per station (as produced by the
    ``synop-dwh`` source); output is a FieldList on the cropped ICON grid, with
    every cell that has no nearby station left as NaN. Each field keeps its
    ``param``/``valid_datetime`` metadata.

    If two stations map to the same cell, the last valid value wins. Stations
    whose nearest cell is further than ``max_distance_km`` are dropped.
    """

    def __init__(
        self,
        grid: dict,
        crop: list[float] | None = CH_EXTENT,
        max_distance_km: float = 5.0,
    ):
        super().__init__()
        self.grid_definition = grid
        self.crop = crop
        self.latitudes, self.longitudes, _ = cropped_icon_grid(grid, crop)
        self.n_cells = int(self.latitudes.size)
        x, y, z = latlon_to_xyz(self.latitudes, self.longitudes)
        self.tree = KDTree(np.stack([x, y, z], axis=-1))
        # KDTree works on the unit sphere, so compare against the chord length
        # corresponding to the requested great-circle distance.
        self.max_distance = max_distance_km / EARTH_RADIUS_KM
        LOG.info(
            "grid-nearest: target grid %d cells (crop=%s), max_distance=%.1f km",
            self.n_cells,
            self.crop,
            max_distance_km,
        )

    def forward(self, data):
        result = []
        for field in data:
            lat, lon = field.grid_points()
            values = np.asarray(field.to_numpy()).reshape(-1)

            x, y, z = latlon_to_xyz(np.asarray(lat), np.asarray(lon))
            distances, indices = self.tree.query(np.stack([x, y, z], axis=-1))

            gridded = np.full(self.n_cells, np.nan, dtype=np.float32)
            keep = np.isfinite(values) & (distances <= self.max_distance)
            gridded[indices[keep]] = values[keep]

            result.append(
                new_field_from_numpy(
                    gridded,
                    template=new_field_from_latitudes_longitudes(
                        field, self.latitudes, self.longitudes
                    ),
                )
            )
        return new_fieldlist_from_list(result)
