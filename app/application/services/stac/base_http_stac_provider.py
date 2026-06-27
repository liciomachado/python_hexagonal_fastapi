from abc import ABC
from datetime import date, datetime, timezone

import httpx

from app.core.http_client import get_shared_http_client
from app.application.services.stac.satellite_collection import (
    DEFAULT_SATELLITE_COLLECTION,
    SatelliteCollection,
)
from app.application.services.stac.stac_provider_port import StacProviderPort
from app.application.services.stac.stac_types import (
    StacGatewayTimeoutError,
    StacProviderName,
    StacSearchError,
    StacSearchResult,
    features_to_items,
)


class BaseHttpStacProvider(StacProviderPort, ABC):
    TIMEOUT_SECONDS = 30

    def __init__(self, search_url: str, provider_name: StacProviderName):
        self._search_url = search_url
        self._provider_name = provider_name

    @property
    def name(self) -> StacProviderName:
        return self._provider_name

    async def search_items_by_day(
        self,
        geojson_geom: dict,
        day: date,
        max_items: int,
        collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> StacSearchResult:
        payload = {
            "collections": [collection.value],
            "intersects": geojson_geom,
            "datetime": f"{day.isoformat()}T00:00:00Z/{day.isoformat()}T23:59:59Z",
            "limit": max_items,
        }
        features = await self._post_search(payload)
        return StacSearchResult(
            items=features_to_items(features),
            provider=self._provider_name,
            collection=collection,
        )

    async def search_items_by_range(
        self,
        geojson_geom: dict,
        start_date: datetime,
        end_date: datetime,
        limit: int,
        collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> StacSearchResult:
        start_date, end_date = self._ensure_utc(start_date), self._ensure_utc(end_date)
        payload = {
            "collections": [collection.value],
            "intersects": geojson_geom,
            "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
            "limit": limit,
        }
        features = await self._post_search(payload)
        return StacSearchResult(
            items=features_to_items(features),
            provider=self._provider_name,
            collection=collection,
        )

    async def health_check(self) -> tuple[bool, int | None, str]:
        payload = {"collections": [DEFAULT_SATELLITE_COLLECTION.value], "limit": 1}
        try:
            client = await get_shared_http_client(timeout=self.TIMEOUT_SECONDS)
            response = await client.post(self._search_url, json=payload)
            if response.status_code == 200:
                count = len(response.json().get("features", []))
                return True, response.status_code, f"Busca STAC executada com sucesso ({count} item(s))"
            message = response.text.strip() or f"HTTP {response.status_code}"
            return False, response.status_code, message
        except httpx.TimeoutException:
            return False, None, "Timeout ao conectar com a API STAC"
        except httpx.RequestError as exc:
            return False, None, f"Erro de conexão: {exc}"

    async def _post_search(self, payload: dict) -> list[dict]:
        try:
            client = await get_shared_http_client(timeout=self.TIMEOUT_SECONDS)
            response = await client.post(self._search_url, json=payload)
        except httpx.TimeoutException as exc:
            raise StacGatewayTimeoutError(
                "Timeout ao conectar com a API STAC",
                provider=self._provider_name,
            ) from exc
        except httpx.RequestError as exc:
            raise StacSearchError(str(exc), provider=self._provider_name) from exc

        if response.status_code == 504:
            message = response.text.strip() or "504 Gateway Timeout"
            raise StacGatewayTimeoutError(message, provider=self._provider_name)

        if response.status_code >= 400:
            message = response.text.strip() or f"HTTP {response.status_code}"
            raise StacSearchError(message, status_code=response.status_code, provider=self._provider_name)

        return response.json().get("features", [])

    def _ensure_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
