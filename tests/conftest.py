"""Test configuration."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def hostname():
    return os.environ.get("CLUSTER_NAME") or os.uname().nodename


@pytest.fixture
def data_dir() -> Path:
    """Path to the test data directory."""
    # TODO: tests use a template file with an empty data section
    # we have to set up proper tests with real data
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    return out
