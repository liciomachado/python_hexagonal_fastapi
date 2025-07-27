from pydantic import BaseModel
from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.core.utils.result import AppError, Result

class GetVisualImageByDayRequest(BaseModel):
    day: str
    cloud_percentual: float
    geometry: str

class GetVisualImageByDayResponse(BaseModel):
    #day: str
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
        return Result.Ok(GetVisualImageByDayResponse(
            #day=response.day,
            cloud_percentual=response.cloud_percentual,
            base64image=response.base64image
        ))
    