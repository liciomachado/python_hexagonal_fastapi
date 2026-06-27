import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.application.services.stac.base_http_stac_provider import BaseHttpStacProvider
from app.application.services.stac.satellite_collection import (
    DEFAULT_SATELLITE_COLLECTION,
    SatelliteCollection,
)
from app.application.services.stac.stac_types import StacProviderName, StacSearchResult


class _TestStacProvider(BaseHttpStacProvider):
    def sign_asset_url(self, href: str) -> str:
        return href


class BaseHttpStacProviderCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_items_by_day_uses_sentinel_by_default(self):
        provider = _TestStacProvider("https://example.com/search", StacProviderName.PLANETARY)
        with patch.object(provider, "_post_search", new_callable=AsyncMock, return_value=[]) as mock_post:
            await provider.search_items_by_day({}, date(2024, 6, 1), 10)
            mock_post.assert_awaited_once()
            payload = mock_post.await_args.args[0]
            self.assertEqual(payload["collections"], [DEFAULT_SATELLITE_COLLECTION.value])

    async def test_search_items_by_day_uses_landsat_when_specified(self):
        provider = _TestStacProvider("https://example.com/search", StacProviderName.PLANETARY)
        with patch.object(provider, "_post_search", new_callable=AsyncMock, return_value=[]) as mock_post:
            result = await provider.search_items_by_day(
                {}, date(2007, 6, 1), 10, collection=SatelliteCollection.LANDSAT_C2_L2
            )
            payload = mock_post.await_args.args[0]
            self.assertEqual(payload["collections"], ["landsat-c2-l2"])
            self.assertEqual(result.collection, SatelliteCollection.LANDSAT_C2_L2)

    async def test_search_items_by_range_uses_landsat_collection(self):
        provider = _TestStacProvider("https://example.com/search", StacProviderName.PLANETARY)
        from datetime import datetime, timezone

        start = datetime(2007, 1, 1, tzinfo=timezone.utc)
        end = datetime(2007, 12, 31, tzinfo=timezone.utc)
        with patch.object(provider, "_post_search", new_callable=AsyncMock, return_value=[]) as mock_post:
            await provider.search_items_by_range(
                {}, start, end, 100, collection=SatelliteCollection.LANDSAT_C2_L2
            )
            payload = mock_post.await_args.args[0]
            self.assertEqual(payload["collections"], ["landsat-c2-l2"])


if __name__ == "__main__":
    unittest.main()
