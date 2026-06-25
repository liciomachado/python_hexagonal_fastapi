from pydantic import BaseModel

from app.application.services.planetary_health_check_service import PlanetaryHealthCheckServicePort
from app.core.utils.result import AppError, Result


class ProviderHealthStatusResponse(BaseModel):
    healthy: bool
    status_code: int | None
    message: str
    url: str


class CircuitBreakerStatusResponse(BaseModel):
    state: str
    opened_until: str | None


class CheckPlanetaryComputerHealthResponse(BaseModel):
    healthy: bool
    planetary: ProviderHealthStatusResponse
    earth_search: ProviderHealthStatusResponse
    circuit_breaker: CircuitBreakerStatusResponse


class CheckPlanetaryComputerHealthUseCase:
    def __init__(self, planetary_health_check_service: PlanetaryHealthCheckServicePort):
        self.planetary_health_check_service = planetary_health_check_service

    async def execute(self) -> Result[CheckPlanetaryComputerHealthResponse, AppError]:
        response = await self.planetary_health_check_service.check()

        if response.is_err():
            return Result.Err(response.error())

        health = response.value()
        return Result.Ok(
            CheckPlanetaryComputerHealthResponse(
                healthy=health.healthy,
                planetary=ProviderHealthStatusResponse(
                    healthy=health.planetary.healthy,
                    status_code=health.planetary.status_code,
                    message=health.planetary.message,
                    url=health.planetary.url,
                ),
                earth_search=ProviderHealthStatusResponse(
                    healthy=health.earth_search.healthy,
                    status_code=health.earth_search.status_code,
                    message=health.earth_search.message,
                    url=health.earth_search.url,
                ),
                circuit_breaker=CircuitBreakerStatusResponse(
                    state=health.circuit_breaker.state,
                    opened_until=health.circuit_breaker.opened_until,
                ),
            )
        )
