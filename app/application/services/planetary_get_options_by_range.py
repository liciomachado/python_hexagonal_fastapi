from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, List
from shapely import wkt
from shapely.geometry import mapping
import httpx

from app.application.services.dtos.planetary_images_filter_response import PlanetaryImageFilterResponse
from collections import defaultdict

class PlanetaryGetOptionImagesByRangeServicePort(ABC):
    @abstractmethod
    async def search_images(self, geometry: str, start_date: datetime, end_date: datetime) -> List[PlanetaryImageFilterResponse]:
        pass

class PlanetaryGetOptionImagesByRangeService(PlanetaryGetOptionImagesByRangeServicePort):
    BASE_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

    async def search_images(self, geometry: str, start_date: datetime, end_date: datetime) -> List[PlanetaryImageFilterResponse]:
        # Converte WKT para GeoJSON
        shapely_geom = wkt.loads(geometry)
        geojson_geom = mapping(shapely_geom)

        start_date, end_date = self.adjustDates(start_date, end_date)

        payload = {
            "collections": ["sentinel-2-l2a"],
            "intersects": geojson_geom,
            "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
            "limit": 100
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.BASE_URL, json=payload)
            response.raise_for_status()
            features = response.json().get("features", [])

        result = self.mapAndGroupResult(features)
        return result

    def mapAndGroupResult(self, features) -> List[PlanetaryImageFilterResponse]:
        responses = [
            PlanetaryImageFilterResponse(
            id=feature["id"],
            datetime=datetime.fromisoformat(feature["properties"]["datetime"]),
            cloud_cover=feature["properties"].get("eo:cloud_cover"),
            geometry=feature["geometry"],
            assets=feature["assets"]
            )
            for feature in features
        ]

        # Group by datetime and select item with lowest cloud_cover
        grouped = defaultdict(list)
        for item in responses:
            grouped[item.datetime].append(item)

        result = []
        for dt, items in grouped.items():
            # Filter out items with None cloud_cover, then select min
            filtered = [i for i in items if i.cloud_cover is not None]
            if filtered:
                best = min(filtered, key=lambda x: x.cloud_cover)
            else:
                best = items[0]  
            result.append(best)
        return result

    def adjustDates(self, start_date: datetime, end_date: datetime) -> tuple[datetime, datetime]:
        # Ensure datetime objects are timezone-aware (UTC)
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        return (start_date, end_date)