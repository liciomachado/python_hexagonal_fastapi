from app.infraestructure.cache.redis_cache_service import RedisCacheService
from app.core.config import Config

_cache_service_instance: RedisCacheService | None = None


def get_cache_service() -> RedisCacheService:
    global _cache_service_instance
    if _cache_service_instance is None:
        _cache_service_instance = RedisCacheService(redis_url=Config.REDIS_URL)
    return _cache_service_instance
