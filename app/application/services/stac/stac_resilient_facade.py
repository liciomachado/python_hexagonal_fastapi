from datetime import date, datetime
import logging

from app.application.services.resilience.circuit_breaker import CircuitBreaker
from app.application.services.stac.providers.earth_search_stac_provider import EarthSearchStacProvider
from app.application.services.stac.providers.planetary_stac_provider import PlanetaryStacProvider
from app.application.services.stac.stac_provider_port import StacProviderPort
from app.application.services.stac.stac_types import (
    StacGatewayTimeoutError,
    StacProviderName,
    StacSearchError,
    StacSearchResult,
)


logger = logging.getLogger("app.stac")


class StacResilientFacade:
    def __init__(
        self,
        planetary_provider: PlanetaryStacProvider,
        earth_search_provider: EarthSearchStacProvider,
        circuit_breaker: CircuitBreaker,
    ):
        self._planetary = planetary_provider
        self._earth_search = earth_search_provider
        self._breaker = circuit_breaker

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._breaker

    async def search_items_by_day(
        self,
        geojson_geom: dict,
        day: date,
        max_items: int,
        preferred_provider: StacProviderName | None = None,
    ) -> StacSearchResult:
        return await self._execute_with_fallback(
            preferred_provider=preferred_provider,
            operation=lambda provider: provider.search_items_by_day(geojson_geom, day, max_items),
        )

    async def search_items_by_range(
        self,
        geojson_geom: dict,
        start_date: datetime,
        end_date: datetime,
        limit: int,
        preferred_provider: StacProviderName | None = None,
    ) -> StacSearchResult:
        return await self._execute_with_fallback(
            preferred_provider=preferred_provider,
            operation=lambda provider: provider.search_items_by_range(
                geojson_geom, start_date, end_date, limit
            ),
        )

    def sign_asset_url(self, href: str, provider: StacProviderName) -> str:
        return self._get_provider(provider).sign_asset_url(href)

    async def health_check_planetary(self) -> tuple[bool, int | None, str, str]:
        healthy, status_code, message = await self._planetary.health_check()
        return healthy, status_code, message, self._planetary._search_url

    async def health_check_earth_search(self) -> tuple[bool, int | None, str, str]:
        healthy, status_code, message = await self._earth_search.health_check()
        return healthy, status_code, message, self._earth_search._search_url

    async def _execute_with_fallback(
        self,
        preferred_provider: StacProviderName | None,
        operation,
    ) -> StacSearchResult:
        if preferred_provider == StacProviderName.EARTH_SEARCH:
            logger.info(
                "STAC routing: earth_search (preferred_provider=earth_search)"
            )
            return await operation(self._earth_search)

        if self._breaker.is_open():
            logger.info(
                "STAC routing: earth_search (circuit_breaker=open, opened_until=%s)",
                self._breaker.opened_until(),
            )
            return await operation(self._earth_search)

        logger.info(
            "STAC routing: planetary (preferred_provider=%s, circuit_breaker=%s)",
            preferred_provider or "planetary",
            self._breaker.state(),
        )
        try:
            return await operation(self._planetary)
        except StacGatewayTimeoutError:
            logger.warning("STAC routing: planetary failed with 504, opening breaker and falling back to earth_search")
            self._breaker.open()
            return await operation(self._earth_search)
        except StacSearchError as exc:
            if exc.status_code == 504:
                logger.warning("STAC routing: planetary failed with 504, opening breaker and falling back to earth_search")
                self._breaker.open()
                return await operation(self._earth_search)
            raise

    def _get_provider(self, provider: StacProviderName) -> StacProviderPort:
        if provider == StacProviderName.EARTH_SEARCH:
            return self._earth_search
        return self._planetary
