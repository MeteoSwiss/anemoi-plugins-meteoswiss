import earthkit.data as ekd
import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from scipy.ndimage import gaussian_filter


class GaussianSmoother(Filter):
    """Smooth selected fields on a regular lat-lon grid with a Gaussian kernel.

    NaN values are handled via the weighted-normalisation trick: both the data
    and a binary validity mask are convolved, then the ratio is taken so that
    NaN cells do not contaminate their neighbours.

    Parameters
    ----------
    sigma:
        Standard deviation of the Gaussian kernel in grid cells.
    params:
        Short names of the parameters to smooth. If omitted, all fields are smoothed.
    """

    def __init__(self, sigma: float, params: list[str] | None = None):
        self.sigma = sigma
        self.params = set(params) if params is not None else None

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        result = []
        for field in data:
            if (
                self.params is not None
                and field.metadata("shortName") not in self.params
            ):
                result.append(field)
                continue
            values = field.to_numpy(flatten=True).reshape(field.shape)
            result.append(
                new_field_from_numpy(
                    _smooth(values, self.sigma).ravel(), template=field
                )
            )
        return new_fieldlist_from_list(result)


def _smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing on a 2D array."""
    nan_mask = np.isnan(grid)
    if not nan_mask.any():
        return gaussian_filter(grid, sigma=sigma)
    filled = np.where(nan_mask, 0.0, grid)
    weight = np.where(nan_mask, 0.0, 1.0)
    smoothed = gaussian_filter(filled, sigma=sigma)
    norm = gaussian_filter(weight, sigma=sigma)
    return np.where(norm > 0, smoothed / norm, np.nan)
