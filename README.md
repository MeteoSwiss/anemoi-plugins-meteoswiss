# anemoi-plugins-meteoswiss

Collection of [anemoi plugins](https://anemoi.readthedocs.io/projects/plugins/en/latest/index.html#) used at MeteoSwiss.

## Sources (`anemoi.datasets.create.sources`)

| YAML key | Class | What it does |
|---|---|---|
| `rea-l-ch1-precip` | `ReaLCh1Precip` | REA-L-CH1 precipitation accumulation from FDB |
| `rea-l-ch1-wind-gust` | `ReaLCh1WindGust` | REA-L-CH1 wind gust max from FDB |
| `synop-dwh` | `SynopDwhSource` | MeteoSwiss synop station observations from DWH |

### `synop-dwh`

Retrieves Swiss synoptic measurement-station observations from the MeteoSwiss
Data Warehouse (DWH) via the `jretrievedwh.py` REST client and exposes them to
the gridded anemoi-datasets creator as one cell per station.

**Requirements**

- `jretrievedwh.py` must be on `$PATH`.
- `$OPR_HOME` must be set; auth uses `$OPR_HOME/.jretrievedwh-conf.<stage>.py`.

On Balfrin both are set up by `source ~osm/.opr_setup_dir`.

**Recipe usage**

```yaml
input:
  join:
    - synop-dwh:
        stage: prod                       # prod | depl | devt
        seq_type: surface                 # jretrieve --seq-type
        increment_minutes: 10             # native cadence; matches recipe frequency
        param:
          - tre200s0                      # 2 m air temperature
          - tde200s0                      # 2 m dew point
          - prestas0                      # station pressure (QFE)
          - fkl010z0                      # 10-min mean wind speed
          - dkl010z0                      # 10-min mean wind direction
          - rre150z0                      # 10-min precipitation
        stations:
          # exactly one of the three:
          group: smn                                  # by station group name
          # locations: [BAS, BER, GVE, ZRH]           # by nat_abbr list
          # bbox: [45.8, 47.9, 5.9, 10.5]             # [minlat, maxlat, minlon, maxlon]
        timeout: 600                       # subprocess timeout in seconds
```

**Behaviour**

- Resolves a deterministic canonical station catalog via a fixed-time-range
  meta-info call to DWH, cached per process. This makes parallel anemoi-datasets
  `load --part i/N` workers all agree on the cell axis.
- Each `execute(dates)` call issues one batched `jretrievedwh.py` invocation for
  the date group, parses the CSV response, reindexes to the canonical station
  order, and fills `(date, station, param)` gaps with `np.nan`.
- Output is an `xarray.Dataset` with dims `(time, station)`, coords
  `latitude`/`longitude` indexed by station, wrapped via
  `XarrayFieldList.from_xarray`. The gridded creator picks up `station` as the
  cell dimension and writes per-station `latitudes`/`longitudes` zarr arrays.

## Tests

```bash
pytest tests/
```

Most tests are pure unit tests (no network / no Balfrin dependency). A few are
gated to Balfrin via the `hostname` fixture in `tests/conftest.py` — they
require either eccodes/meteodatalab data files or DWH access, and are skipped
elsewhere. Examples of Balfrin-only tests:

- `tests/test_helpers.py` — earthkit ↔ meteodatalab roundtrips against a
  GRIB fixture.
- `tests/test_transform/test_filters.py` — `Destagger` and
  `ClipLateralBoundaries` against real ICON data.
- `tests/test_transform/test_synop_dwh.py::test_synop_source_execute_balfrin`
  — end-to-end `SynopDwhSource.execute(...)` against real DWH.
