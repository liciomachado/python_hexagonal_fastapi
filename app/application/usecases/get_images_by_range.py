from typing import List
from app.application.services.planetary_get_options_by_range import PlanetaryGetOptionImagesByRangeServicePort
from app.application.services.stac.preferred_provider import PreferredProvider
from app.application.services.stac.satellite_collection import DEFAULT_SATELLITE_COLLECTION, SatelliteCollection
from app.application.validators.usecase_validators import (
    require_valid_date_range,
    require_valid_sentinel_geometry,
)
from app.core.utils.result import AppError, NotFoundError, Result
from datetime import datetime
from pydantic import BaseModel

class GetImagesByRangeRequest(BaseModel):
    dt_start: datetime
    dt_end: datetime
    geom: str
    preferred_provider: PreferredProvider | None = None
    satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION   

class GetImagesByRangeResponse(BaseModel):
    id: str
    datetime: datetime
    cloud_cover: float | None
    cloud_cover_geometry: float | None

class GetImagesByRangeUseCase:
    def __init__(self, planetary_image_service: PlanetaryGetOptionImagesByRangeServicePort):
        self.planetary_image_service = planetary_image_service

    async def execute(self, request: GetImagesByRangeRequest) -> Result[List[GetImagesByRangeResponse], AppError]:
        date_validation = require_valid_date_range(request.dt_start, request.dt_end)
        if date_validation.is_err():
            return Result.Err(date_validation.error())

        geometry_validation = require_valid_sentinel_geometry(request.geom)
        if geometry_validation.is_err():
            return Result.Err(geometry_validation.error())

        images = await self.planetary_image_service.search_images(
            geometry=request.geom,
            start_date=request.dt_start,
            end_date=request.dt_end,
            preferred_provider=request.preferred_provider,
            satellite_collection=request.satellite_collection,
        )

        if not images:
            return Result.Err(NotFoundError("No images found"))

        response = [
            GetImagesByRangeResponse(
                id=image.id,
                datetime=image.datetime,
                cloud_cover=image.cloud_cover,
                cloud_cover_geometry=image.cloud_cover_geometry,
            )
            for image in images
        ]
        return Result.Ok(response)
