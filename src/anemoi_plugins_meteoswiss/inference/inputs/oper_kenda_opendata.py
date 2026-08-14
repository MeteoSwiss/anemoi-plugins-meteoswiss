"""anemoi-inference input: KENDA-CH1 analysis via the MeteoSwiss STAC open-data API.

Only pulled in via the ``oper-inference`` extra.

Usage
-----

::

    input:
      cutout:
        - lam_0:
            oper-kenda-opendata:
              cache_dir: /path/to/cache   # optional, default: none (always re-download)
"""

import logging
import os
from collections.abc import Iterable
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import earthkit.data as ekd
import requests
from anemoi.inference.inputs import input_registry
from anemoi.inference.inputs.mars import MarsInput
from anemoi.inference.types import Date
from earthkit.data.utils.dates import to_datetime

LOG = logging.getLogger(__name__)

STAC_BASE_URL = "https://data.geo.admin.ch/api/stac/v1"
STAC_COLLECTION_ID_KENDA = "ch.meteoschweiz.ogd-analysis-kenda-ch1"
STAC_COLLECTION_ID_ICON = "ch.meteoschweiz.ogd-forecasting-icon-ch1"
STAC_PAGE_LIMIT = 100


def _stac_items(
    valid_time: datetime, *, limit: int = STAC_PAGE_LIMIT
) -> Iterator[dict]:
    """Yield every STAC item for the KENDA-CH1 collection valid at ``valid_time``, following pagination."""
    url = f"{STAC_BASE_URL}/collections/{STAC_COLLECTION_ID_KENDA}/items"
    params = {"datetime": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": limit}
    while url:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        yield from page.get("features", [])
        next_hrefs = [
            link["href"] for link in page.get("links", []) if link.get("rel") == "next"
        ]
        url, params = (next_hrefs[0], None) if next_hrefs else (None, None)


def _asset_href(item: dict) -> str:
    """Return a STAC item's sole asset href; raises if that invariant doesn't hold."""
    assets = list(item.get("assets", {}).values())
    if len(assets) != 1:
        raise ValueError(
            f"Expected exactly one asset on KENDA-CH1 item {item.get('id')!r}, got {len(assets)}"
        )
    return assets[0]["href"]


def _hrefs_by_variable(items: Iterable[dict]) -> dict[str, str]:
    """Map each item's ``forecast:variable`` to its asset href."""
    return {
        item["properties"]["forecast:variable"]: _asset_href(item) for item in items
    }


def _as_list(value: Any) -> list[Any]:
    """Normalise a ``mars_requests()`` request value (scalar or list/tuple/set) to a list."""
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _collection_static_assets() -> dict[str, str]:
    """Map each KENDA-CH1 static GRIB2 asset's filename to its (presigned) href.

    These are collection-level assets (grid constants), not hourly items --
    fetched from the collection endpoint, not ``/items``.
    """
    response = requests.get(
        f"{STAC_BASE_URL}/collections/{STAC_COLLECTION_ID_ICON}", timeout=30
    )
    response.raise_for_status()
    return {
        name: asset["href"]
        for name, asset in response.json().get("assets", {}).items()
        if asset.get("type") == "application/grib"
    }


def _download_to_path(href: str, path: str) -> None:
    """Stream ``href`` to ``path``, writing to a temp file first so an interrupted download can't
    leave a corrupt/truncated file that looks like a valid cache entry."""
    tmp_path = f"{path}.part"
    with requests.get(href, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=1024 * 1024))
    os.replace(tmp_path, path)


def _cached_fields(
    cache_dir: str | None, subdir: str, hrefs_by_filename: dict[str, str]
) -> ekd.FieldList:
    """Fetch ``hrefs_by_filename``'s assets, reusing whatever's already at ``cache_dir/subdir`` and
    persisting the rest there; downloads straight to memory without touching disk if ``cache_dir`` is unset."""
    if not cache_dir:
        return ekd.from_source("url", list(hrefs_by_filename.values()))

    directory = os.path.join(cache_dir, subdir)
    os.makedirs(directory, exist_ok=True)
    paths = []
    for filename, href in hrefs_by_filename.items():
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            _download_to_path(href, path)
        paths.append(path)

    return ekd.from_source("file", paths)


def _cached_constants_fields(cache_dir: str | None) -> ekd.FieldList:
    """Fetch KENDA-CH1's static grid constants, reusing an on-disk cache if ``cache_dir`` is set."""
    return _cached_fields(cache_dir, "constants", _collection_static_assets())


@input_registry.register("oper-kenda-opendata")
class OperKendaOpenDataInput(MarsInput):
    """KENDA-CH1 analysis input via the MeteoSwiss STAC open-data API; analysis-only, no forecast-step fallback."""

    def __init__(
        self,
        context,
        metadata,
        *,
        cache_dir: str | None = None,
        extra_variables: Iterable[str] = (),
        skip_variables: Iterable[str] = (),
        **kwargs: Any,
    ) -> None:
        """``cache_dir``, if set, persists downloaded analysis fields and grid-constants assets to
        disk so later runs don't re-download them.

        ``extra_variables`` are KENDA-CH1 variable names to fetch on every call regardless of
        whether the checkpoint requests them as prognostic variables in their own right -- e.g.
        fields ``ModelToPressureLevel`` needs as fixed, time-varying auxiliary inputs. Its
        *constant* auxiliaries (HSURF, HHL) are the same two already covered by ``_constants()``.

        ``skip_variables``
        """
        super().__init__(context, metadata, **kwargs)
        self.cache_dir = cache_dir
        self.extra_variables = set(extra_variables)
        self.skip_variables = set(skip_variables)
        self._constant_fields: ekd.FieldList | None = None

    def _constants(self, *, names: Iterable[str] | None = None) -> ekd.FieldList:
        """Fetch KENDA-CH1's static grid constants (``HHL``, ``HSURF``, ``FR_LAND``, ...), downloading
        at most once per instance. Their ``date``/``time``/``step`` are whatever the collection
        asset itself carries -- callers that need a specific valid time must override it themselves.

        ``names``, if given, restricts the result to constants whose ``shortName`` is in it --
        pass the caller's actually-requested variables so unrequested constants aren't retrieved."""
        if self._constant_fields is None:
            self._constant_fields = _cached_constants_fields(self.cache_dir)
        fields = self._constant_fields
        if names is not None:
            names = set(names)
            fields = [field for field in fields if field.metadata("shortName") in names]
        return ekd.SimpleFieldList(fields)

    def _constant_variable_names(self) -> set[str]:
        """Names of all static grid constants this input can serve (whether or not requested this call).

        These never show up in the hourly STAC items (they're collection-level assets, fetched
        separately -- see ``_collection_static_assets``), so if a checkpoint variable happens to
        share a constant's name it must be excluded from the raw hourly fetch in ``retrieve()``;
        otherwise ``hrefs_by_variable[v]`` raises a ``KeyError`` for it."""
        return {field.metadata("shortName") for field in self._constants()}

    def retrieve(self, variables: list[str], dates: list[Date]) -> Any:
        """Retrieve KENDA-CH1 analysis fields (plus static constants) for the given target valid times."""
        result = ekd.FieldList()

        constant_variable_names = self._constant_variable_names()

        for date in dates:
            requests_ = self.metadata.mars_requests(
                variables=variables,
                dates=[date],  # one date at a time -> r["date"]/r["time"] stay scalars
                use_grib_paramid=False,
                patch_request=self.patch_data_request,
            )
            if not requests_:
                raise ValueError(f"No requests for {variables} ({date})")
            requested_variables = {
                p for r in requests_ for p in _as_list(r.get("param", []))
            }
            if requested_variables & self.skip_variables:
                LOG.info(
                    "oper-kenda-opendata: skipping the following variables(s): %s",
                    sorted(requested_variables & self.skip_variables),
                )
            variables_to_provide = (
                requested_variables | self.extra_variables
            ) - self.skip_variables

            constants_to_fetch = sorted(variables_to_provide & constant_variable_names)
            variables_to_fetch = sorted(variables_to_provide - constant_variable_names)

            valid_time = to_datetime(date)

            hrefs_by_variable = _hrefs_by_variable(_stac_items(valid_time))

            LOG.info(
                "oper-kenda-opendata: fetching %d variable(s): %s for valid_datetime=%s",
                len(variables_to_fetch) + len(constants_to_fetch),
                variables_to_fetch + constants_to_fetch,
                valid_time,
            )

            result += _cached_fields(
                self.cache_dir,
                os.path.join("raw", valid_time.strftime("%Y%m%dT%H%M%S")),
                {f"{v}.grib2": hrefs_by_variable[v] for v in variables_to_fetch},
            )

            overrides = {
                "date": int(valid_time.strftime("%Y%m%d")),
                "time": int(valid_time.strftime("%H%M")),
                "step": 0,
            }
            result += ekd.SimpleFieldList(
                [
                    field.clone(metadata=field.metadata().override(**overrides))
                    for field in self._constants(names=constants_to_fetch)
                ]
            )

        return result.to_fieldlist()
