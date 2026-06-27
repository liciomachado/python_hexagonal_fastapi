from enum import Enum
from typing import Any

import pystac

from app.application.services.stac.satellite_collection import (
    DEFAULT_SATELLITE_COLLECTION,
    SatelliteCollection,
)


class StacProviderName(str, Enum):
    PLANETARY = "planetary"
    EARTH_SEARCH = "earth_search"


SENTINEL2_L2A_COLLECTION = SatelliteCollection.SENTINEL2_L2A.value
LANDSAT_C2_L2_COLLECTION = SatelliteCollection.LANDSAT_C2_L2.value

BAND_ASSET_ALIASES: dict[str, list[str]] = {
    "B02": ["B02", "blue"],
    "B03": ["B03", "green"],
    "B04": ["B04", "red"],
    "B08": ["B08", "nir", "nir08"],
    "B11": ["B11", "swir16", "swir"],
    "SCL": ["SCL", "scl"],
    "QA_PIXEL": ["QA_PIXEL", "qa_pixel"],
}


class StacSearchError(Exception):
    def __init__(self, message: str, status_code: int | None = None, provider: StacProviderName | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider


class StacGatewayTimeoutError(StacSearchError):
    def __init__(self, message: str, provider: StacProviderName):
        super().__init__(message, status_code=504, provider=provider)


class StacSearchResult:
    def __init__(
        self,
        items: list[pystac.Item],
        provider: StacProviderName,
        collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ):
        self.items = items
        self.provider = provider
        self.collection = collection


def resolve_band_href(item: pystac.Item, band_key: str) -> str:
    aliases = BAND_ASSET_ALIASES.get(band_key, [band_key])
    for alias in aliases:
        asset = item.assets.get(alias)
        if asset is not None and asset.href:
            return asset.href
    raise KeyError(band_key)


def resolve_collection_from_item(item: pystac.Item) -> SatelliteCollection:
    collection_id = item.collection
    if collection_id == LANDSAT_C2_L2_COLLECTION:
        return SatelliteCollection.LANDSAT_C2_L2
    return SatelliteCollection.SENTINEL2_L2A


def features_to_items(features: list[dict[str, Any]]) -> list[pystac.Item]:
    return [pystac.Item.from_dict(feature) for feature in features]
