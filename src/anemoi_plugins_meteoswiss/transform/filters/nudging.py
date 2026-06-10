import earthkit.data as ekd
from anemoi.transform.filter import Filter


class NudgeTowardObservation(Filter):
    """A filter that nudges the forecast toward observations"""
    
    def __init__(self, nugget: float):
        """Initialize the filter.

        Parameters
        ----------
        nugget:
            tuning weight that controls how strongly observations influence the model correction.
        """
        self.nugget = nugget
        super().__init__()
    
    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        print(f"NudgeObservation filter active (nugget={self.nugget})")
        for field in data:
            # TODO: apply nudging to some fields
            pass
        return data
