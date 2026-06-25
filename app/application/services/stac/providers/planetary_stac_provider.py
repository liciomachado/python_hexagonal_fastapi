from planetary_computer import sign

from app.application.services.stac.base_http_stac_provider import BaseHttpStacProvider
from app.application.services.stac.stac_types import StacProviderName

PLANETARY_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


class PlanetaryStacProvider(BaseHttpStacProvider):
    def __init__(self, search_url: str = PLANETARY_SEARCH_URL):
        super().__init__(search_url=search_url, provider_name=StacProviderName.PLANETARY)

    def sign_asset_url(self, href: str) -> str:
        return sign(href)
