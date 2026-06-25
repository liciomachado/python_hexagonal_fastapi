from abc import ABC, abstractmethod
from datetime import date, datetime

from app.application.services.stac.stac_types import StacProviderName, StacSearchResult


class StacProviderPort(ABC):
    @property
    @abstractmethod
    def name(self) -> StacProviderName:
        pass

    @abstractmethod
    async def search_items_by_day(
        self,
        geojson_geom: dict,
        day: date,
        max_items: int,
    ) -> StacSearchResult:
        pass

    @abstractmethod
    async def search_items_by_range(
        self,
        geojson_geom: dict,
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> StacSearchResult:
        pass

    @abstractmethod
    def sign_asset_url(self, href: str) -> str:
        pass

    @abstractmethod
    async def health_check(self) -> tuple[bool, int | None, str]:
        pass
