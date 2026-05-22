from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from anemoi_plugins_meteoswiss.transform.sources import SynopDwhSource
from anemoi_plugins_meteoswiss.transform.sources.synop_dwh.stations import StationCatalog


def _meta_df(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "station", "op_since", "op_till", "parameter",
            "latitude", "longitude", "elev", "stn_name", "nat_abbr",
        ],
    )


def test_station_catalog_dedupes_per_station_and_sorts_by_nat_abbr():
    # Two stations, two params each, plus a stale historical row for station 75.
    # The exact-row-wins detail isn't important; what matters is (a) one row per
    # station, (b) sorted by nat_abbr ascending, (c) deterministic across input
    # row orderings — this is the parallel-safety contract.
    rows = [
        # station, op_since,         op_till,           parameter, lat,        lon,      elev, name,                nat_abbr
        [75,  "20091209153000", None,             "tre200s0", 47.541065, 7.583556, 316, "Basel / Binningen", "BAS"],
        [75,  "20091209153000", None,             "prestas0", 47.541145, 7.583525, 316, "Basel / Binningen", "BAS"],
        [75,  "19810101000000", "20091209152900", "tre200s0", 47.541222, 7.582753, 316, "Basel / Binningen", "BAS"],
        [78,  "20060803110000", None,             "tre200s0", 46.990746, 7.463997, 553, "Bern / Zollikofen", "BER"],
        [78,  "20060803110000", None,             "prestas0", 46.990746, 7.464061, 553, "Bern / Zollikofen", "BER"],
    ]
    cat = StationCatalog.from_meta(_meta_df(rows))

    assert cat.n == 2
    assert list(cat.nat_abbr) == ["BAS", "BER"]
    assert list(cat.station_id) == [75, 78]
    np.testing.assert_array_equal(cat.elevation, [316.0, 553.0])

    # Determinism under input shuffling: a parallel worker may receive rows in
    # a different order from the wire, but the catalog must come out identical.
    shuffled = StationCatalog.from_meta(_meta_df(list(reversed(rows))))
    np.testing.assert_array_equal(shuffled.station_id, cat.station_id)
    np.testing.assert_allclose(shuffled.latitude, cat.latitude)
    np.testing.assert_allclose(shuffled.longitude, cat.longitude)


def test_df_to_xarray_aligns_to_canonical_and_fills_missing_with_nan():
    catalog = StationCatalog(
        nat_abbr=np.array(["BAS", "BER", "GVE"], dtype=object),
        station_id=np.array([75, 78, 58], dtype=np.int64),
        latitude=np.array([47.5, 47.0, 46.2]),
        longitude=np.array([7.6, 7.4, 6.1]),
        elevation=np.array([316.0, 553.0, 412.0]),
        name=np.array(["Basel", "Bern", "Geneva"], dtype=object),
    )

    # Recipe wants 3 timestamps × 3 stations × 2 params, but DWH only returned:
    #   - BAS at all 3 timestamps for both params
    #   - BER at the middle timestamp only, prestas0 only
    #   - GVE: nothing
    #   - One row whose stationId is unknown to the catalog (must be ignored)
    data = pd.DataFrame({
        "station":  [75,                75,                75,                78,                999],
        "termin":   ["20240101000000",  "20240101001000",  "20240101002000",  "20240101001000",  "20240101000000"],
        "tre200s0": [6.4,               6.5,               6.6,               np.nan,            42.0],
        "prestas0": [974.7,             974.6,             974.5,             948.3,             100.0],
    })

    src = SynopDwhSource(
        context=None,
        param=["tre200s0", "prestas0"],
        stations={"locations": ["BAS", "BER", "GVE"]},
    )
    dates = [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10), datetime(2024, 1, 1, 0, 20)]
    ds = src._df_to_xarray(data, dates, catalog)

    # Shape & dim names
    assert ds["tre200s0"].dims == ("time", "station")
    assert ds["tre200s0"].shape == (3, 3)
    # Station axis is in canonical (BAS, BER, GVE) order
    assert list(ds["station"].values) == ["BAS", "BER", "GVE"]
    np.testing.assert_allclose(ds["latitude"].values, [47.5, 47.0, 46.2])

    # BAS has values at all 3 timesteps
    np.testing.assert_allclose(ds["tre200s0"].sel(station="BAS").values, [6.4, 6.5, 6.6])
    # BER only had prestas0 at the middle timestep, NaN elsewhere
    np.testing.assert_allclose(
        ds["prestas0"].sel(station="BER").values,
        [np.nan, 948.3, np.nan],
    )
    # GVE returned nothing → all NaN
    assert np.isnan(ds["tre200s0"].sel(station="GVE").values).all()
    assert np.isnan(ds["prestas0"].sel(station="GVE").values).all()
    # The bogus stationId 999 row must not pollute the cube
    assert not (ds["tre200s0"].values == 42.0).any()


def test_df_to_xarray_empty_response_yields_all_nan():
    catalog = StationCatalog(
        nat_abbr=np.array(["BAS"], dtype=object),
        station_id=np.array([75], dtype=np.int64),
        latitude=np.array([47.5]),
        longitude=np.array([7.6]),
        elevation=np.array([316.0]),
        name=np.array(["Basel"], dtype=object),
    )
    src = SynopDwhSource(
        context=None,
        param=["tre200s0"],
        stations={"locations": ["BAS"]},
    )
    dates = [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 10)]
    ds = src._df_to_xarray(pd.DataFrame(), dates, catalog)

    assert ds["tre200s0"].shape == (2, 1)
    assert np.isnan(ds["tre200s0"].values).all()


def test_jretrieve_argv_translation_for_each_station_selection_mode():
    """The three station selection modes produce the expected jretrieve flags."""
    from anemoi_plugins_meteoswiss.transform.sources.synop_dwh.jretrieve import (
        _stations_to_argv,
    )

    assert _stations_to_argv({"group": "smn"}) == ["-a", "stn_group,smn"]
    assert _stations_to_argv({"locations": ["BAS", "BER"]}) == ["-i", "nat_abbr,BAS,BER"]
    assert _stations_to_argv({"bbox": [45.8, 47.9, 5.9, 10.5]}) == [
        "-l", "45.8,47.9,5.9,10.5"
    ]
    with pytest.raises(ValueError, match="exactly one"):
        _stations_to_argv({"group": "smn", "locations": ["BAS"]})
    with pytest.raises(ValueError, match="exactly one"):
        _stations_to_argv({})


def test_synop_source_execute_balfrin(hostname):
    """End-to-end smoke against real DWH — Balfrin-only since it shells out to jretrievedwh.py."""
    if not hostname.startswith("balfrin"):
        pytest.skip("Only runs on Balfrin (needs jretrievedwh.py + DWH access).")
    src = SynopDwhSource(
        context=None,
        param=["tre200s0", "prestas0"],
        stations={"locations": ["BAS", "BER"]},
        increment_minutes=10,
    )
    fl = src.execute([
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 10),
    ])
    # Expect 2 params × 2 stations-as-cells × 2 timesteps. earthkit's XarrayFieldList
    # flattens (time, var) so we just check it produced a non-empty FieldList where
    # each field has 2 grid points (one per station).
    assert len(fl) > 0
    for f in fl:
        assert f.values.size == 2  # 2 stations
