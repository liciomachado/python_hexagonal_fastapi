from enum import Enum
from typing import Any

import pystac


class StacProviderName(str, Enum):
    PLANETARY = "planetary"
    EARTH_SEARCH = "earth_search"


SENTINEL2_L2A_COLLECTION = "sentinel-2-l2a"

BAND_ASSET_ALIASES: dict[str, list[str]] = {
    "B02": ["B02", "blue"],
    "B03": ["B03", "green"],
    "B04": ["B04", "red"],
    "B08": ["B08", "nir"],
    "B11": ["B11", "swir16", "swir"],
    "SCL": ["SCL", "scl"],
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
    def __init__(self, items: list[pystac.Item], provider: StacProviderName):
        self.items = items
        self.provider = provider


def resolve_band_href(item: pystac.Item, band_key: str) -> str:
    aliases = BAND_ASSET_ALIASES.get(band_key, [band_key])
    for alias in aliases:
        asset = item.assets.get(alias)
        if asset is not None and asset.href:
            return asset.href
    raise KeyError(band_key)


def features_to_items(features: list[dict[str, Any]]) -> list[pystac.Item]:
    return [pystac.Item.from_dict(feature) for feature in features]
