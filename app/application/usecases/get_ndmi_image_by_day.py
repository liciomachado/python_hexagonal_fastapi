
from datetime import date

from pydantic import BaseModel

from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.core.utils.result import AppError, Result


class GetNdmiImageByDayRequest(BaseModel):
    day: date
    cloud_percentual: float
    geometry: str
    generate_image: bool = True

class GetNdmiImageByDayResponse(BaseModel):
    day: date
    cloud_percentual: float
    base64image: str | None
    ndmi_mean: float | None
    ndmi_min: float | None
    ndmi_max: float | None
    sat_image_id: str

class GetNdmiImageByDayUseCase:
    def __init__(self, planetary_visual_image_service: PlanetaryVisualImageServicePort):
        self.planetary_visual_image_service = planetary_visual_image_service

    async def execute(self, request: GetNdmiImageByDayRequest) -> Result[GetNdmiImageByDayResponse, AppError]:
        response = await self.planetary_visual_image_service.get_ndmi_image(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry,
            generate_image=request.generate_image
        )

        if response.is_err():
            return Result.Err(response.error())
        response = response.value()
        return Result.Ok(GetNdmiImageByDayResponse(
            day=response.day,
            cloud_percentual=response.cloud_percentual,
            base64image=response.base64image,
            ndmi_mean=response.ndvi_mean,
            ndmi_min=response.ndvi_min,
            ndmi_max=response.ndvi_max,
            sat_image_id=response.sat_image_id
        ))