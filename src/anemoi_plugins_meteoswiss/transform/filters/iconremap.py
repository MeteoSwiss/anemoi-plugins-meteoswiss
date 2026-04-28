import logging

import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)


def _rot_to_geo(
    rlon_deg: np.ndarray,
    rlat_deg: np.ndarray,
    pollon_deg: float,
    pollat_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rotated lat-lon to geographic lat-lon (CF convention).

    Uses arctan2 for correct quadrant handling in all cases.
    """
    pollat = np.radians(pollat_deg)
    rlon = np.radians(rlon_deg)
    rlat = np.radians(rlat_deg)

    sin_lat = np.sin(pollat) * np.sin(rlat) + np.cos(pollat) * np.cos(rlat) * np.cos(rlon)
    lat = np.degrees(np.arcsin(np.clip(sin_lat, -1.0, 1.0)))

    dlon_rad = np.arctan2(
        np.cos(rlat) * np.sin(rlon),
        np.cos(pollat) * np.sin(rlat) - np.sin(pollat) * np.cos(rlat) * np.cos(rlon),
    )
    lon = np.degrees(dlon_rad) + pollon_deg
    return lat, ((lon + 180) % 360) - 180


class IconRemapToRegLatLon(Filter):
    """Regrid ICON-CH fields to a regular lat-lon grid using iconremap RBF weights.

    The order of input points must match the ICON cell ordering used when computing
    the weights with iconremap. Output coordinates are converted from rotated to
    geographic (unrotated) lat-lon.

    Weights can be negative (RBF characteristic); the result is clipped to the
    [vmin, vmax] range of each stencil to ensure conservative interpolation.
    Output points where any stencil index equals 0 (out-of-domain sentinel used
    by some iconremap weight files) are set to NaN.

    Parameters
    ----------
    weights_file:
        Path to an iconremap NetCDF weights file containing ``rbf_B_glbidx``,
        ``rbf_B_wgt``, and grid attributes (nx, ny, xmin, dx, ymin, dy,
        north_pole_lon, north_pole_lat).
    """

    def __init__(self, weights_file: str):
        coeffs = xr.open_dataset(weights_file)
        self.nx = int(coeffs.attrs["nx"])
        self.ny = int(coeffs.attrs["ny"])

        self.indices = coeffs["rbf_B_glbidx"].values  # (ny*nx, n_stencil), 0-based
        self.weights = coeffs["rbf_B_wgt"].values  # (ny*nx, n_stencil), may be negative

        # Index 0 is the out-of-domain sentinel in general iconremap weight files
        self.valid_mask = np.all(self.indices != 0, axis=-1)  # (ny*nx,)

        np_lon = float(coeffs.attrs["north_pole_lon"])
        np_lat = float(coeffs.attrs["north_pole_lat"])
        xmin = float(coeffs.attrs["xmin"])
        dx = float(coeffs.attrs["dx"])
        ymin = float(coeffs.attrs["ymin"])
        dy = float(coeffs.attrs["dy"])
        coeffs.close()

        rlon = xmin + np.arange(self.nx) * dx
        rlat = ymin + np.arange(self.ny) * dy
        rlon_2d, rlat_2d = np.meshgrid(rlon, rlat)  # (ny, nx) row-major

        self._latitudes, self._longitudes = _rot_to_geo(
            rlon_2d.ravel(), rlat_2d.ravel(), np_lon, np_lat
        )

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        result = []
        for field in data:
            values = field.to_numpy(flatten=True)  # (n_source,)
            stencil = values[self.indices]  # (n_out, n_stencil)
            vmin = stencil.min(axis=-1)
            vmax = stencil.max(axis=-1)
            regridded = np.sum(stencil * self.weights, axis=-1)
            regridded = np.clip(regridded, vmin, vmax)
            regridded[~self.valid_mask] = np.nan
            result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(regridded, template=field),
                    latitudes=self._latitudes,
                    longitudes=self._longitudes,
                )
            )
        return new_fieldlist_from_list(result)
