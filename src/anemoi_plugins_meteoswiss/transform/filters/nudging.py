import logging
from pathlib import Path

import earthkit.data as ekd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)


class NudgeTowardObservation(Filter):
    """A filter that nudges the forecast toward observations"""

    def __init__(self, nugget: float, observation_path: str):
        """Initialize the filter.

        Parameters
        ----------
        nugget : float
            Tuning weight that controls how strongly observations influence the model correction.
        observation_path : str
            Path to the observation data used for nudging.
        """
        self.nugget = nugget
        self.observation_path = Path(observation_path)
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        LOG.info("NudgeTowardObservation filter active (nugget=%s, observation_path=%s)", self.nugget, self.observation_path)
        print(f"NudgeTowardObservation filter active (nugget={self.nugget}, observation_path={self.observation_path})", flush=True)
        for field in data:
            # TODO: apply nudging to some fields
            pass
        return data
