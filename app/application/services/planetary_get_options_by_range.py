from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, List
from shapely import wkt
from shapely.geometry import mapping
from collections import defaultdict

from app.application.services.dtos.planetary_images_filter_response import PlanetaryImageFilterResponse
from app.application.services.stac.preferred_provider import PreferredProvider
from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.application.services.stac.stac_types import StacProviderName


class PlanetaryGetOptionImagesByRangeServicePort(ABC):
    @abstractmethod
    async def search_images(
        self,
        geometry: str,
        start_date: datetime,
        end_date: datetime,
        preferred_provider: PreferredProvider | None = None,
    ) -> List[PlanetaryImageFilterResponse]:
        pass


class PlanetaryGetOptionImagesByRangeService(PlanetaryGetOptionImagesByRangeServicePort):
    BASE_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

    def __init__(self, stac_facade: StacResilientFacade):
        self._stac_facade = stac_facade

    async def search_images(
        self,
        geometry: str,
        start_date: datetime,
        end_date: datetime,
        preferred_provider: PreferredProvider | None = None,
    ) -> List[PlanetaryImageFilterResponse]:
        shapely_geom = wkt.loads(geometry)
        geojson_geom = mapping(shapely_geom)
        start_date, end_date = self.adjustDates(start_date, end_date)
        provider_enum = StacProviderName(preferred_provider) if preferred_provider else None

        search_result = await self._stac_facade.search_items_by_range(
            geojson_geom=geojson_geom,
            start_date=start_date,
            end_date=end_date,
            limit=100,
            preferred_provider=provider_enum,
        )

        features = [
            {
                "id": item.id,
                "properties": item.properties,
                "geometry": item.geometry,
                "assets": {key: asset.to_dict() for key, asset in item.assets.items()},
            }
            for item in search_result.items
        ]
        return self.mapAndGroupResult(features)

    def mapAndGroupResult(self, features) -> List[PlanetaryImageFilterResponse]:
        responses = [
            PlanetaryImageFilterResponse(
                id=feature["id"],
                datetime=datetime.fromisoformat(feature["properties"]["datetime"]),
                cloud_cover=feature["properties"].get("eo:cloud_cover"),
                geometry=feature["geometry"],
                assets=feature["assets"],
            )
            for feature in features
        ]

        grouped = defaultdict(list)
        for item in responses:
            grouped[item.datetime].append(item)

        result = []
        for _, items in grouped.items():
            filtered = [i for i in items if i.cloud_cover is not None]
            if filtered:
                best = min(filtered, key=lambda x: x.cloud_cover)
            else:
                best = items[0]
            result.append(best)
        return result

    def adjustDates(self, start_date: datetime, end_date: datetime) -> tuple[datetime, datetime]:
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        return (start_date, end_date)
