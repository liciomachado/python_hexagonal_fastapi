from app.application.services.stac.providers.earth_search_stac_provider import EarthSearchStacProvider
from app.application.services.stac.providers.planetary_stac_provider import PlanetaryStacProvider
from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.application.services.resilience.circuit_breaker import CircuitBreaker
from app.core.config import Config


_stac_facade_instance: StacResilientFacade | None = None


def get_stac_facade() -> StacResilientFacade:
    global _stac_facade_instance
    if _stac_facade_instance is None:
        _stac_facade_instance = StacResilientFacade(
            planetary_provider=PlanetaryStacProvider(),
            earth_search_provider=EarthSearchStacProvider(search_url=Config.STAC_EARTHSEARCH_URL),
            circuit_breaker=CircuitBreaker(open_seconds=Config.STAC_BREAKER_OPEN_SECONDS),
        )
    return _stac_facade_instance
