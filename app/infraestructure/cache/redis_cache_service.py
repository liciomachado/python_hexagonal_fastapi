import logging

import redis.asyncio as redis

from app.domain.ports.cache_port import CachePort

logger = logging.getLogger("app.cache")


class RedisCacheService(CachePort):
    """Implementação Redis com métodos GET/SET reutilizáveis por qualquer fluxo."""

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("RedisCacheService não conectado. Chame connect() no startup.")
        return self._client

    async def get(self, key: str) -> str | None:
        client = self._require_client()
        value = await client.get(key)
        if value is not None:
            logger.debug("Cache HIT key=%s", key)
        return value

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        client = self._require_client()
        if ttl_seconds is not None:
            await client.set(key, value, ex=ttl_seconds)
        else:
            await client.set(key, value)
        logger.debug("Cache SET key=%s ttl=%s", key, ttl_seconds)

    async def delete(self, key: str) -> None:
        client = self._require_client()
        await client.delete(key)
