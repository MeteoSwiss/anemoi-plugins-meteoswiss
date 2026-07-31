import fnmatch

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
        Names (or glob patterns, e.g. ``z_*``) of the parameters to smooth.
        If omitted, all fields are smoothed.
    """

    def __init__(self, sigma: float, params: list[str] | None = None):
        self.sigma = sigma
        self.params = list(params) if params is not None else None

    def matches(self, name: str | None) -> bool:
        if self.params is None:
            return True
        if name is None:
            return False
        return any(fnmatch.fnmatch(name, pat) for pat in self.params)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        return new_fieldlist_from_list(
            [
                smooth(x, self.sigma) if self.matches(x.metadata("param")) else x
                for x in data
            ]
        )


def smooth(field: ekd.Field, sigma: float) -> ekd.Field:
    """NaN-aware Gaussian smoothing of a single field."""
    values = field.to_numpy(flatten=True).reshape(field.shape)
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        smoothed = gaussian_filter(values, sigma=sigma)
    else:
        filled = np.where(nan_mask, 0.0, values)
        weight = np.where(nan_mask, 0.0, 1.0)
        smoothed = gaussian_filter(filled, sigma=sigma)
        norm = gaussian_filter(weight, sigma=sigma)
        smoothed = np.where(norm > 0, smoothed / norm, np.nan)
    return new_field_from_numpy(smoothed.ravel(), template=field)
