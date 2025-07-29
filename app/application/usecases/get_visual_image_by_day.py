from pydantic import BaseModel
from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.core.utils.result import AppError, Result
from datetime import date

class GetVisualImageByDayRequest(BaseModel):
    day: date
    cloud_percentual: float
    geometry: str

class GetVisualImageByDayResponse(BaseModel):
    day: date
    cloud_percentual: float
    base64image: str

class GetVisualImageByDayUseCase:
    def __init__(self, planetary_visual_image_service: PlanetaryVisualImageServicePort):
        self.planetary_visual_image_service = planetary_visual_image_service

    async def execute(self, request: GetVisualImageByDayRequest) -> Result[GetVisualImageByDayResponse, AppError]:
        response = await self.planetary_visual_image_service.get_visual_image(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry
        )
        if response.is_err():
            return Result.Err(response.error)
        response = response.value()
        return Result.Ok(GetVisualImageByDayResponse(
            day=response.day,
            cloud_percentual=response.cloud_percentual,
            base64image=response.base64image
        ))
    