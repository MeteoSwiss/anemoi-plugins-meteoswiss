import logging

import earthkit.data as ekd
import numpy as np
import torch
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
        np.sin(pollat) * np.cos(rlat) * np.cos(rlon) - np.cos(pollat) * np.sin(rlat),
    )
    lon = np.degrees(dlon_rad) + pollon_deg - 180.0
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

    def __init__(self, weights_file: str, device: str | None = None, fields_per_chunk: int = 16):
        coeffs = xr.open_dataset(weights_file)
        self.nx = int(coeffs.attrs["nx"])
        self.ny = int(coeffs.attrs["ny"])

        self.indices = coeffs["rbf_B_glbidx"].values  # (ny*nx, n_stencil), 0-based
        self.weights = coeffs["rbf_B_wgt"].values  # (ny*nx, n_stencil), may be negative

        # Index 0 is the out-of-domain sentinel in general iconremap weight files
        self.valid_mask = np.all(self.indices != 0, axis=-1)  # (ny*nx,)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fields_per_chunk = fields_per_chunk
        self._idx_t = torch.as_tensor(self.indices, dtype=torch.int64, device=self.device)
        self._wgt_t = torch.as_tensor(self.weights, dtype=torch.float32, device=self.device)
        self._invalid_t = torch.as_tensor(~self.valid_mask, dtype=torch.bool, device=self.device)

        LOG.info("Using new on-GPU iconremap implementation for regridding. Device is %s.", self.device)
        
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

        self._latitudes, self._longitudes = _rot_to_geo(rlon_2d.ravel(), rlat_2d.ravel(), np_lon, np_lat)

    @torch.no_grad()
    def _remap_batch(self, values: np.ndarray) -> np.ndarray:
        """Regrid a batch of fields using iconremap weights.

        Parameters
        ----------
        batch:
            Input fields as a 2D array of shape (n_fields, ny*nx).
        """
        v = torch.from_numpy(np.ascontiguousarray(values)).to(self.device, non_blocking=True)  # (n_fields, ny*nx)
        stencil = v[:, self._idx_t]  # (n_fields, n_out, n_stencil)
        out = (stencil * self._wgt_t).sum(dim=-1)  # (n_fields, n_out)
        out = torch.minimum(torch.maximum(out, stencil.amin(-1)), stencil.amax(-1))  # clip to [vmin, vmax]
        out[:, self._invalid_t] = float("nan")
        return out.cpu().numpy()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        fields = list(data)
        result = []
        for i in range(0, len(fields), self.fields_per_chunk):
            chunk = fields[i : i + self.fields_per_chunk]
            regridded = self._remap_batch(np.stack([f.to_numpy(flatten=True) for f in chunk], axis=0))  # (n_chunk, ny*nx)
            for field, r in zip(chunk, regridded):
                result.append(
                    new_field_from_latitudes_longitudes(
                        new_field_from_numpy(r.reshape(self.ny, self.nx), template=field),
                        latitudes=self._latitudes,
                        longitudes=self._longitudes,
                    )
                )
        return new_fieldlist_from_list(result)