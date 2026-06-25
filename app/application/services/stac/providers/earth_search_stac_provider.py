from app.application.services.stac.base_http_stac_provider import BaseHttpStacProvider
from app.application.services.stac.stac_types import StacProviderName

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"


class EarthSearchStacProvider(BaseHttpStacProvider):
    def __init__(self, search_url: str = EARTH_SEARCH_URL):
        super().__init__(search_url=search_url, provider_name=StacProviderName.EARTH_SEARCH)

    def sign_asset_url(self, href: str) -> str:
        return href
