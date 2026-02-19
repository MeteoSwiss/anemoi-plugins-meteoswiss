"""Test configuration."""

import os
import sys
from pathlib import Path

import pytest

eccodes_definitions = (
    Path(sys.prefix) / "share" / "eccodes-cosmo-resources" / "definitions"
)
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definitions)


@pytest.fixture
def data_dir() -> Path:
    """Path to the test data directory."""
    # TODO: tests use a template file with an empty data section
    # we have to set up proper tests with real data
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    return out
