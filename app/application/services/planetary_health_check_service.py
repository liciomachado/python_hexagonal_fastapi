from abc import ABC, abstractmethod

from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.core.utils.result import AppError, Result


class ProviderHealthStatus:
    def __init__(self, healthy: bool, status_code: int | None, message: str, url: str):
        self.healthy = healthy
        self.status_code = status_code
        self.message = message
        self.url = url


class CircuitBreakerStatus:
    def __init__(self, state: str, opened_until: str | None):
        self.state = state
        self.opened_until = opened_until


class PlanetaryHealthCheckResponse:
    def __init__(
        self,
        healthy: bool,
        planetary: ProviderHealthStatus,
        earth_search: ProviderHealthStatus,
        circuit_breaker: CircuitBreakerStatus,
    ):
        self.healthy = healthy
        self.planetary = planetary
        self.earth_search = earth_search
        self.circuit_breaker = circuit_breaker


class PlanetaryHealthCheckServicePort(ABC):
    @abstractmethod
    async def check(self) -> Result[PlanetaryHealthCheckResponse, AppError]:
        pass


class PlanetaryHealthCheckService(PlanetaryHealthCheckServicePort):
    def __init__(self, stac_facade: StacResilientFacade):
        self._stac_facade = stac_facade

    async def check(self) -> Result[PlanetaryHealthCheckResponse, AppError]:
        planetary_healthy, planetary_status, planetary_message, planetary_url = (
            await self._stac_facade.health_check_planetary()
        )
        earth_healthy, earth_status, earth_message, earth_url = (
            await self._stac_facade.health_check_earth_search()
        )
        breaker = self._stac_facade.circuit_breaker
        opened_until = breaker.opened_until()
        breaker_status = CircuitBreakerStatus(
            state=breaker.state(),
            opened_until=opened_until.isoformat() if opened_until else None,
        )
        overall_healthy = planetary_healthy or earth_healthy
        return Result.Ok(
            PlanetaryHealthCheckResponse(
                healthy=overall_healthy,
                planetary=ProviderHealthStatus(
                    healthy=planetary_healthy,
                    status_code=planetary_status,
                    message=planetary_message,
                    url=planetary_url,
                ),
                earth_search=ProviderHealthStatus(
                    healthy=earth_healthy,
                    status_code=earth_status,
                    message=earth_message,
                    url=earth_url,
                ),
                circuit_breaker=breaker_status,
            )
        )
