from datetime import date

from pydantic import BaseModel

from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.application.services.stac.preferred_provider import PreferredProvider
from app.core.utils.result import AppError, Result
from .get_visual_image_by_day import GetVisualImageByDayResponse
from .get_ndvi_image_by_day import GetNdviImageByDayResponse
from .get_ndmi_image_by_day import GetNdmiImageByDayResponse


class GetAllImagesByDayRequest(BaseModel):
    day: date
    cloud_percentual: float
    geometry: str
    generate_image: bool = True
    preferred_provider: PreferredProvider | None = None


class GetAllImagesByDayResponse(BaseModel):
    visual: GetVisualImageByDayResponse
    ndvi: GetNdviImageByDayResponse
    ndmi: GetNdmiImageByDayResponse


class GetAllImagesByDayUseCase:
    def __init__(self, planetary_visual_image_service: PlanetaryVisualImageServicePort):
        self.planetary_visual_image_service = planetary_visual_image_service

    async def execute(self, request: GetAllImagesByDayRequest) -> Result[GetAllImagesByDayResponse, AppError]:
        result = await self.planetary_visual_image_service.get_all_images_by_day(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry,
            generate_image=request.generate_image,
            preferred_provider=request.preferred_provider,
        )

        if result.is_err():
            return Result.Err(result.error())

        payload = result.value()
        return Result.Ok(
            GetAllImagesByDayResponse(
                visual=GetVisualImageByDayResponse(
                    day=payload.visual.day,
                    cloud_percentual=payload.visual.cloud_percentual,
                    image_url=payload.visual.image_url,
                ),
                ndvi=GetNdviImageByDayResponse(
                    day=payload.ndvi.day,
                    cloud_percentual=payload.ndvi.cloud_percentual,
                    image_url=payload.ndvi.image_url,
                    ndvi_mean=payload.ndvi.ndvi_mean,
                    ndvi_min=payload.ndvi.ndvi_min,
                    ndvi_max=payload.ndvi.ndvi_max,
                    sat_image_id=payload.ndvi.sat_image_id,
                ),
                ndmi=GetNdmiImageByDayResponse(
                    day=payload.ndmi.day,
                    cloud_percentual=payload.ndmi.cloud_percentual,
                    image_url=payload.ndmi.image_url,
                    ndmi_mean=payload.ndmi.ndvi_mean,
                    ndmi_min=payload.ndmi.ndvi_min,
                    ndmi_max=payload.ndmi.ndvi_max,
                    sat_image_id=payload.ndmi.sat_image_id,
                ),
            )
        )
