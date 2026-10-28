"""Static ICON-CH1 grid fields (land-sea mask, orography) from the extpar file."""

from __future__ import annotations

import logging

import earthkit.data as ekd
import numpy as np
import pandas as pd
import xarray as xr
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources.xarray_support import XarrayFieldList

from ..icon_grid import CH_EXTENT
from ..icon_grid import ICON_EXTPAR_PATH
from ..icon_grid import cropped_icon_grid

LOG = logging.getLogger(__name__)

# Recipe param name -> variable name in the ICON external-parameter (extpar) file.
# Names follow ECMWF/anemoi conventions where there is a clear analog (sdor, slor),
# so these fields line up with ERA5-derived orography descriptors.
EXTPAR_VARIABLES: dict[str, str] = {
    "lsm": "FR_LAND",  # land fraction in [0, 1]
    "orography": "topography_c",  # geometric height of the surface above sea level [m]
    "sdor": "SSO_STDH",  # std. dev. of subgrid-scale orography [m] (ECMWF sdor)
    "slor": "SSO_SIGMA",  # slope of subgrid-scale orography (ECMWF slor)
    "sso_theta": "SSO_THETA",  # angle of subgrid-scale orography [rad]
    "sso_gamma": "SSO_GAMMA",  # anisotropy of subgrid-scale orography
    "z0": "Z0",  # surface roughness length
}


class IconStatic(Source):
    """Time-invariant ICON-CH1 grid fields broadcast over the requested dates.

    Reads per-cell static fields (e.g. ``lsm`` from extpar's ``FR_LAND``) and
    emits them on the same cropped grid as the gridded observations, so the two
    share an identical cell axis when joined. Suitable as the ``forcings``
    template, since it is a full-grid field.
    """

    emoji = "🗺️"

    def __init__(
        self,
        context,
        *,
        grid: dict,
        param: list[str],
        crop: list[float] | None = CH_EXTENT,
        extpar_path: str = ICON_EXTPAR_PATH,
    ):
        super().__init__(context)
        if not param:
            raise ValueError("param must be a non-empty list.")
        unknown = [p for p in param if p not in EXTPAR_VARIABLES]
        if unknown:
            raise ValueError(
                f"Unknown icon-static param(s): {unknown}. "
                f"Known: {sorted(EXTPAR_VARIABLES)}"
            )
        self.param = list(param)
        self.grid_definition = grid
        self.crop = crop
        self.extpar_path = extpar_path
        self.latitudes, self.longitudes, self.mask = cropped_icon_grid(grid, crop)

    def _values(self) -> dict[str, np.ndarray]:
        """Read and crop each requested static field from the extpar file."""
        out: dict[str, np.ndarray] = {}
        with xr.open_dataset(self.extpar_path) as ds:
            for p in self.param:
                arr = np.asarray(ds[EXTPAR_VARIABLES[p]].values).reshape(-1)
                if arr.size != self.mask.size:
                    raise ValueError(
                        f"extpar variable {EXTPAR_VARIABLES[p]!r} has {arr.size} "
                        f"cells, expected {self.mask.size} to match the grid."
                    )
                out[p] = arr[self.mask].astype(np.float32)
        return out

    def execute(self, dates) -> ekd.FieldList:
        date_list = sorted(set(getattr(dates, "dates", dates)))
        if not date_list:
            return ekd.from_source("empty")

        values = self._values()
        n_t = len(date_list)
        time_index = pd.DatetimeIndex(date_list)

        coords = {
            "time": ("time", time_index.values.astype("datetime64[ns]")),
            "cell": ("cell", np.arange(self.latitudes.size)),
            "latitude": ("cell", self.latitudes),
            "longitude": ("cell", self.longitudes),
        }
        data_vars = {
            p: (("time", "cell"), np.tile(v, (n_t, 1))) for p, v in values.items()
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds["latitude"].attrs.update(units="degrees_north", standard_name="latitude")
        ds["longitude"].attrs.update(units="degrees_east", standard_name="longitude")
        ds["time"].attrs.update(standard_name="time")
        return XarrayFieldList.from_xarray(ds)
