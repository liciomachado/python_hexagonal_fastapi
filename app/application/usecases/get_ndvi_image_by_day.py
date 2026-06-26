from datetime import date

from pydantic import BaseModel

from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.application.services.stac.preferred_provider import PreferredProvider
from app.core.utils.result import AppError, Result


class GetNdviImageByDayRequest(BaseModel):
    day: date
    cloud_percentual: float
    geometry: str
    generate_image: bool = True
    preferred_provider: PreferredProvider | None = None

class GetNdviImageByDayResponse(BaseModel):
    day: date
    cloud_percentual: float
    image_url: str | None
    ndvi_mean: float | None
    ndvi_min: float | None
    ndvi_max: float | None
    sat_image_id: str

class GetNdviImageByDayUseCase:
    def __init__(self, planetary_visual_image_service: PlanetaryVisualImageServicePort):
        self.planetary_visual_image_service = planetary_visual_image_service

    async def execute(self, request: GetNdviImageByDayRequest) -> Result[GetNdviImageByDayResponse, AppError]:
        response = await self.planetary_visual_image_service.get_ndvi_image(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry,
            generate_image=request.generate_image,
            preferred_provider=request.preferred_provider,
        )

        if response.is_err():
            return Result.Err(response.error())
        response = response.value()
        return Result.Ok(GetNdviImageByDayResponse(
            day=response.day,
            cloud_percentual=response.cloud_percentual,
            image_url=response.image_url,
            ndvi_mean=response.ndvi_mean,
            ndvi_min=response.ndvi_min,
            ndvi_max=response.ndvi_max,
            sat_image_id=response.sat_image_id
        ))
