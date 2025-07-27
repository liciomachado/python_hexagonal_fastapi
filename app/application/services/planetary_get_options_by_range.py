from abc import ABC, abstractmethod
import datetime
from typing import Any, List
from shapely import wkt
from shapely.geometry import mapping
import httpx

from app.application.services.dtos.planetary_images_filter_response import PlanetaryImageFilterResponse

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
        
        return [
            PlanetaryImageFilterResponse(
                id=feature["id"],
                datetime=datetime.datetime.fromisoformat(feature["properties"]["datetime"]),
                cloud_cover=feature["properties"].get("eo:cloud_cover"),
                geometry=feature["geometry"],
                assets=feature["assets"]
            )
            for feature in features
        ]
    
