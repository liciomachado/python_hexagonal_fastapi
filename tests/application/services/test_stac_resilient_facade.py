import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from app.application.services.resilience.circuit_breaker import CircuitBreaker
from app.application.services.stac.providers.earth_search_stac_provider import EarthSearchStacProvider
from app.application.services.stac.providers.planetary_stac_provider import PlanetaryStacProvider
from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.application.services.stac.stac_types import StacGatewayTimeoutError, StacProviderName, StacSearchResult


class StacResilientFacadeTests(unittest.IsolatedAsyncioTestCase):
    def _build_facade(self) -> tuple[StacResilientFacade, MagicMock, MagicMock, CircuitBreaker]:
        planetary = MagicMock(spec=PlanetaryStacProvider)
        planetary.name = StacProviderName.PLANETARY
        earth = MagicMock(spec=EarthSearchStacProvider)
        earth.name = StacProviderName.EARTH_SEARCH
        breaker = CircuitBreaker(open_seconds=300)
        facade = StacResilientFacade(planetary, earth, breaker)
        return facade, planetary, earth, breaker

    async def test_uses_planetary_by_default(self):
        facade, planetary, earth, _ = self._build_facade()
        expected = StacSearchResult(items=[], provider=StacProviderName.PLANETARY)
        planetary.search_items_by_day = AsyncMock(return_value=expected)

        result = await facade.search_items_by_day({}, date(2024, 6, 1), 1)

        self.assertEqual(result.provider, StacProviderName.PLANETARY)
        planetary.search_items_by_day.assert_awaited_once()
        earth.search_items_by_day.assert_not_called()

    async def test_fallback_on_504_and_opens_breaker(self):
        facade, planetary, earth, breaker = self._build_facade()
        planetary.search_items_by_day = AsyncMock(
            side_effect=StacGatewayTimeoutError("504", provider=StacProviderName.PLANETARY)
        )
        expected = StacSearchResult(items=[], provider=StacProviderName.EARTH_SEARCH)
        earth.search_items_by_day = AsyncMock(return_value=expected)

        result = await facade.search_items_by_day({}, date(2024, 6, 1), 1)

        self.assertEqual(result.provider, StacProviderName.EARTH_SEARCH)
        self.assertTrue(breaker.is_open())

    async def test_uses_earth_search_when_breaker_open(self):
        facade, planetary, earth, breaker = self._build_facade()
        breaker.open()
        expected = StacSearchResult(items=[], provider=StacProviderName.EARTH_SEARCH)
        earth.search_items_by_day = AsyncMock(return_value=expected)

        result = await facade.search_items_by_day({}, date(2024, 6, 1), 1)

        self.assertEqual(result.provider, StacProviderName.EARTH_SEARCH)
        planetary.search_items_by_day.assert_not_called()

    async def test_preferred_earth_search_skips_planetary(self):
        facade, planetary, earth, breaker = self._build_facade()
        expected = StacSearchResult(items=[], provider=StacProviderName.EARTH_SEARCH)
        earth.search_items_by_day = AsyncMock(return_value=expected)

        result = await facade.search_items_by_day(
            {},
            date(2024, 6, 1),
            1,
            preferred_provider=StacProviderName.EARTH_SEARCH,
        )

        self.assertEqual(result.provider, StacProviderName.EARTH_SEARCH)
        planetary.search_items_by_day.assert_not_called()
        self.assertFalse(breaker.is_open())


if __name__ == "__main__":
    unittest.main()
