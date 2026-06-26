import unittest
from unittest.mock import AsyncMock

from app.infraestructure.cache.redis_cache_service import RedisCacheService


class RedisCacheServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_set_and_get(self):
        cache = RedisCacheService("redis://localhost:6379/0")
        mock_client = AsyncMock()
        mock_client.get.return_value = "cached-value"
        cache._client = mock_client

        await cache.set("key", "value", ttl_seconds=60)
        result = await cache.get("key")

        mock_client.set.assert_awaited_once_with("key", "value", ex=60)
        mock_client.get.assert_awaited_once_with("key")
        self.assertEqual(result, "cached-value")

    async def test_delete(self):
        cache = RedisCacheService("redis://localhost:6379/0")
        mock_client = AsyncMock()
        cache._client = mock_client

        await cache.delete("key")
        mock_client.delete.assert_awaited_once_with("key")


if __name__ == "__main__":
    unittest.main()
