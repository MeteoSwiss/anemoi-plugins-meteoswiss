import logging
from pathlib import Path

import earthkit.data as ekd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)


class NudgeTowardObservation(Filter):
    """A filter that nudges the forecast toward observations"""

    def __init__(self, nugget: float, path_to_observation: str):
        """Initialize the filter.

        Parameters
        ----------
        nugget : float
            Tuning weight that controls how strongly observations influence the model correction.
        observation_path : str
            Path to the observation data used for nudging.
        """
        self.nugget = nugget
        self.path_to_observation = Path(path_to_observation)
        LOG.info(
            "Initialised nudging filter with nugget=%.4f, observations from '%s'",
            self.nugget,
            self.path_to_observation,
        )
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        LOG.info(
            "Applying nudging toward observations (nugget=%.4f, observations='%s')",
            self.nugget,
            self.path_to_observation,
        )
        for field in data:
            # TODO: apply nudging to some fields
            pass
        return data
