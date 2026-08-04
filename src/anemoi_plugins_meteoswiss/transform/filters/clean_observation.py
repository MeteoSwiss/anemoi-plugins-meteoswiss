import logging
from pathlib import Path

import earthkit.data as ekd
import pandas as pd
from anemoi.transform.filter import Filter

LOG = logging.getLogger(__name__)


class CleanObservation(Filter):
    """Clean pre-fetched station observations and write the result to disk.

    Reads a Parquet file produced by RetrieveObservation, applies quality-
    control cleaning (placeholder — no-op for now), and writes the cleaned
    DataFrame to a new Parquet file for use by NudgeTowardObservation.
    The forecast fields are passed through unchanged.

    Parameters
    ----------
    obs_path_in : str
        Path to the raw observation Parquet file written by RetrieveObservation.
    obs_path_out : str
        Path where the cleaned observation Parquet file will be written.
    """

    def __init__(self, obs_path_in: str, obs_path_out: str):
        self.obs_path_in = Path(obs_path_in)
        self.obs_path_out = Path(obs_path_out)
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Read, clean, and write observations; return forecast fields unchanged.

        Parameters
        ----------
        data : ekd.FieldList
            Forecast fields (passed through unchanged).

        Returns
        -------
        ekd.FieldList
            The input data, unchanged.
        """
        if not self.obs_path_in.exists():
            raise FileNotFoundError(f"Observation file not found: {self.obs_path_in}")

        df = pd.read_parquet(self.obs_path_in)
        LOG.info("Loaded %d stations from %s", len(df), self.obs_path_in)

        df = self._clean(df)

        df.to_parquet(self.obs_path_out)
        LOG.info("Saved %d cleaned stations to %s", len(df), self.obs_path_out)

        return data

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality-control cleaning to station observations.

        Parameters
        ----------
        df : pd.DataFrame
            Raw station observations with columns for each variable plus
            ``latitude`` and ``longitude``, in SI units.

        Returns
        -------
        pd.DataFrame
            Cleaned observations (currently a no-op placeholder).
        """
        
        LOG.info("The data cleaning logic should be implemented here. Currently, this is a no-op placeholder.")
        
        return df