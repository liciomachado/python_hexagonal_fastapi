import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

import pystac
from shapely import wkt
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from app.application.services.dtos.planetary_images_filter_response import PlanetaryImageFilterResponse
from app.application.services.geometry_bounds import compute_cloud_cover_geom_bounds
from app.application.services.geometry_cloud_cover_service import GeometryCloudCoverService
from app.application.services.stac.preferred_provider import PreferredProvider
from app.application.services.stac.satellite_collection import DEFAULT_SATELLITE_COLLECTION, SatelliteCollection
from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.application.services.stac.stac_types import StacProviderName
from app.core.config import Config


class PlanetaryGetOptionImagesByRangeServicePort(ABC):
    @abstractmethod
    async def search_images(
        self,
        geometry: str,
        start_date: datetime,
        end_date: datetime,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> List[PlanetaryImageFilterResponse]:
        pass


class PlanetaryGetOptionImagesByRangeService(PlanetaryGetOptionImagesByRangeServicePort):

    def __init__(
        self,
        stac_facade: StacResilientFacade,
        cloud_cover_service: GeometryCloudCoverService | None = None,
    ):
        self._stac_facade = stac_facade
        self._cloud_cover_service = cloud_cover_service or GeometryCloudCoverService(stac_facade)

    async def search_images(
        self,
        geometry: str,
        start_date: datetime,
        end_date: datetime,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> List[PlanetaryImageFilterResponse]:
        geom, geom_bounds = self._parse_geometry(geometry)
        geojson_geom = mapping(geom)
        start_date, end_date = self.adjustDates(start_date, end_date)
        provider_enum = StacProviderName(preferred_provider) if preferred_provider else None

        search_result = await self._stac_facade.search_items_by_range(
            geojson_geom=geojson_geom,
            start_date=start_date,
            end_date=end_date,
            limit=100,
            preferred_provider=provider_enum,
            collection=satellite_collection,
        )

        items_by_id = {item.id: item for item in search_result.items}
        features = [
            {
                "id": item.id,
                "properties": item.properties,
                "geometry": item.geometry,
                "assets": {key: asset.to_dict() for key, asset in item.assets.items()},
            }
            for item in search_result.items
        ]
        grouped = self.mapAndGroupResult(features)
        return await self._enrich_with_geometry_cloud_cover(
            grouped,
            items_by_id,
            geom,
            geom_bounds,
            search_result.provider,
            search_result.collection,
        )

    async def _enrich_with_geometry_cloud_cover(
        self,
        items: List[PlanetaryImageFilterResponse],
        items_by_id: dict[str, pystac.Item],
        geom: BaseGeometry,
        geom_bounds: tuple[float, float, float, float],
        provider: StacProviderName,
        collection: SatelliteCollection,
    ) -> List[PlanetaryImageFilterResponse]:
        if not items:
            return items

        semaphore = asyncio.Semaphore(Config.SCL_CONCURRENT_READS)

        async def enrich(item: PlanetaryImageFilterResponse) -> None:
            stac_item = items_by_id.get(item.id)
            if stac_item is None:
                return
            async with semaphore:
                item.cloud_cover_geometry = await asyncio.to_thread(
                    self._cloud_cover_service.compute_cloud_percentual_over_geometry,
                    stac_item,
                    geom,
                    geom_bounds,
                    provider,
                    collection,
                )

        await asyncio.gather(*(enrich(item) for item in items))
        return items

    def _parse_geometry(self, geometry: str) -> tuple[BaseGeometry, tuple[float, float, float, float]]:
        geom = wkt.loads(geometry)
        return geom, compute_cloud_cover_geom_bounds(geom)

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
