from abc import ABC, abstractmethod

import httpx

from app.application.services.planetary_get_options_by_range import PlanetaryGetOptionImagesByRangeService
from app.core.utils.result import AppError, Result


class PlanetaryHealthCheckResponse:
    def __init__(self, healthy: bool, status_code: int | None, message: str, url: str):
        self.healthy = healthy
        self.status_code = status_code
        self.message = message
        self.url = url


class PlanetaryHealthCheckServicePort(ABC):
    @abstractmethod
    async def check(self) -> Result[PlanetaryHealthCheckResponse, AppError]:
        pass


class PlanetaryHealthCheckService(PlanetaryHealthCheckServicePort):
    SEARCH_URL = PlanetaryGetOptionImagesByRangeService.BASE_URL
    TIMEOUT_SECONDS = 30
    HEALTH_CHECK_PAYLOAD = {
        "collections": ["sentinel-2-l2a"],
        "limit": 1,
    }

    async def check(self) -> Result[PlanetaryHealthCheckResponse, AppError]:
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT_SECONDS) as client:
                response = await client.post(self.SEARCH_URL, json=self.HEALTH_CHECK_PAYLOAD)

            if response.status_code == 200:
                features_count = len(response.json().get("features", []))
                return Result.Ok(
                    PlanetaryHealthCheckResponse(
                        healthy=True,
                        status_code=response.status_code,
                        message=f"Busca STAC executada com sucesso ({features_count} item(s) retornado(s))",
                        url=self.SEARCH_URL,
                    )
                )

            message = response.text.strip() or f"HTTP {response.status_code}"
            return Result.Ok(
                PlanetaryHealthCheckResponse(
                    healthy=False,
                    status_code=response.status_code,
                    message=message,
                    url=self.SEARCH_URL,
                )
            )
        except httpx.TimeoutException:
            return Result.Ok(
                PlanetaryHealthCheckResponse(
                    healthy=False,
                    status_code=None,
                    message="Timeout ao conectar com a API do Planetary Computer",
                    url=self.SEARCH_URL,
                )
            )
        except httpx.RequestError as exc:
            return Result.Ok(
                PlanetaryHealthCheckResponse(
                    healthy=False,
                    status_code=None,
                    message=f"Erro de conexão: {exc}",
                    url=self.SEARCH_URL,
                )
            )
