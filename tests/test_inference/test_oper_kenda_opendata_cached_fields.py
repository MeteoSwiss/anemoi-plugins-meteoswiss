"""Offline tests for ``oper_kenda_opendata._cached_fields``."""

import pytest

pytest.importorskip("anemoi.inference.inputs.mars")

from anemoi_plugins_meteoswiss.inference.inputs.oper_kenda_opendata import _cached_fields


@pytest.mark.parametrize("cache_dir", [None, "cache"])
def test_nothing_to_download(tmp_path, cache_dir):
    """A request for grid constants only leaves no hourly assets to fetch, with or without a cache."""
    cache_dir = cache_dir and str(tmp_path / cache_dir)
    assert len(_cached_fields(cache_dir, "raw", {})) == 0
