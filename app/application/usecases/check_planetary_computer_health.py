from pydantic import BaseModel

from app.application.services.planetary_health_check_service import PlanetaryHealthCheckServicePort
from app.core.utils.result import AppError, Result


class CheckPlanetaryComputerHealthResponse(BaseModel):
    healthy: bool
    status_code: int | None
    message: str
    url: str


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
                status_code=health.status_code,
                message=health.message,
                url=health.url,
            )
        )
