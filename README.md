# anemoi-plugins-meteoswiss

Collection of [anemoi plugins](https://anemoi.readthedocs.io/projects/plugins/en/latest/index.html#) used at MeteoSwiss.

## Sources (`anemoi.datasets.create.sources`)

| YAML key | Class | What it does |
|---|---|---|
| `rea-l-ch1-precip` | `ReaLCh1Precip` | REA-L-CH1 precipitation accumulation from FDB |
| `rea-l-ch1-wind-gust` | `ReaLCh1WindGust` | REA-L-CH1 wind gust max from FDB |
| `synop-dwh` | `SynopDwhSource` | MeteoSwiss synop station observations from DWH |
| `icon-static` | `IconStatic` | Time-invariant ICON-CH1 grid fields (e.g. `lsm`) from the extpar file |

## Filters (`anemoi.transform.filters`)

| YAML key | Class | What it does |
|---|---|---|
| `grid-nearest` | `GridNearest` | Nearest-neighbour project point observations onto the (cropped) ICON-CH1 grid |

(plus `assign-grid`, `destagger`, `clip-lateral-boundaries`,
`model-to-pressure-level`, `omega-from-w`, `geopotential-from-height`,
`average-flux-to-cumulative-quantity`.)

### `synop-dwh`

Retrieves Swiss synoptic measurement-station observations from the MeteoSwiss
Data Warehouse (DWH) via the `jretrievedwh.py` REST client and exposes them to
the gridded anemoi-datasets creator as one cell per station.

**Requirements**

- `jretrievedwh.py` must be reachable — on `$PATH`, or the wrapper falls back to
  `/oprusers/osm/opr.inn/bin/jretrievedwh.py`. On Balfrin, `source
  ~osm/.opr_setup_dir` puts it on `$PATH`.
- Auth uses a committed `.jretrievedwh-conf.prod.py` that mints a Bearer token
  from OAuth client credentials `JRETRIEVE_CLIENT_ID` / `JRETRIEVE_CLIENT_SECRET`
  (env vars or a `.env` next to the conf). Only the `prod` stage is supported.

**Recipe usage**

```yaml
input:
  join:
    - synop-dwh:
        stage: prod                       # prod only
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
- Set `include_altitude: true` to add a time-invariant `stalt` variable (station
  altitude from the catalog). It rides through `grid-nearest` and lands on each
  station's cell — useful as a per-cell altitude feature.

### `grid-nearest` and `icon-static` (observations on the ICON-CH1 grid)

To build a dataset *on the ICON-CH1 grid* (instead of one cell per station),
pipe the `synop-dwh` source through the `grid-nearest` filter and join a static
land-sea mask + forcings computed on the same grid. The crop
(`[west, east, south, north]`, default `CH_EXTENT` ≈ Switzerland) reduces the
~1.15M-cell ICON-CH1 domain to ~90k cells; cells without a nearby station are
`NaN`.

```yaml
input:
  join:
    - pipe:
        - synop-dwh: { param: [tre200s0, ...], include_altitude: true, stations: {...} }
        - grid-nearest:
            grid: { icon: { path: /scratch/.../icon_grid_0001_R19B08_mch.nc } }
            crop: [5.7, 10.6, 45.7, 47.95]   # [W, E, S, N]; omit for the full grid
            max_distance_km: 5               # drop stations with no cell this close
    - icon-static:
        grid: { icon: { path: /scratch/.../icon_grid_0001_R19B08_mch.nc } }
        crop: [5.7, 10.6, 45.7, 47.95]       # identical crop → identical cell axis
        param: [lsm, orography, sdor, slor]  # see EXTPAR_VARIABLES for the full set
    - pipe:
        - forcings:
            template: ${input.join.1.icon-static}   # a full-grid field as template
            param: [cos_latitude, ..., cos_solar_zenith_angle]
        - rename:
            param: { cos_solar_zenith_angle: cos_sza }
```

`grid-nearest` and `icon-static` share `transform/icon_grid.py` (`cropped_icon_grid`)
so they resolve the same, identically-ordered cell axis — a prerequisite for the
join. `icon-static` reads static fields from the ICON *extpar* file (the grid file
itself carries only geometry). See
[config/surface-dwh-smn-icon-ch1-2024-10m-v0.1.yaml](../../config/surface-dwh-smn-icon-ch1-2024-10m-v0.1.yaml).

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
