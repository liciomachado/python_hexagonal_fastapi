from typing import Any, List

from app.application.services.planetary_get_options_by_range import PlanetaryGetOptionImagesByRangeServicePort
from app.core.utils.result import AppError, NotFoundError, Result
from datetime import datetime
from pydantic import BaseModel

class GetImagesByRangeRequest(BaseModel):
    dt_start: datetime
    dt_end: datetime
    geom: str
    
class GetImagesByRangeResponse(BaseModel):
    id: str
    datetime: datetime
    cloud_cover: float | None
    assets: dict[str, Any]

class GetImagesByRangeUseCase:
    def __init__(self, planetary_image_service: PlanetaryGetOptionImagesByRangeServicePort):
        self.planetary_image_service = planetary_image_service

    async def execute(self, request: GetImagesByRangeRequest) -> Result[List[GetImagesByRangeResponse], AppError]:
        images = await self.planetary_image_service.search_images(
            geometry=request.geom,
            start_date=request.dt_start,
            end_date=request.dt_end
        )

        if not images:
            return Result.Err(NotFoundError("No images found"))

        response = [
            GetImagesByRangeResponse(
            id=image.id,
            datetime=image.datetime,
            cloud_cover=image.cloud_cover,
            geometry=image.geometry,
            assets=image.assets
            )
            for image in images
        ]
        return Result.Ok(response)