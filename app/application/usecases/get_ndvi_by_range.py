from datetime import datetime
from typing import List

from pydantic import BaseModel

from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.application.services.stac.preferred_provider import PreferredProvider
from app.application.services.stac.satellite_collection import DEFAULT_SATELLITE_COLLECTION, SatelliteCollection
from app.application.usecases.get_ndvi_image_by_day import GetNdviImageByDayResponse
from app.application.validators.usecase_validators import (
    require_valid_date_range,
    require_valid_sentinel_geometry,
)
from app.core.utils.result import AppError, Result


class GetNdviByRangeRequest(BaseModel):
    dt_start: datetime
    dt_end: datetime
    geometry: str
    cloud_percentual: float
    generate_image: bool = True
    preferred_provider: PreferredProvider | None = None
    satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION


class GetNdviByRangeUseCase:
    def __init__(self, planetary_visual_image_service: PlanetaryVisualImageServicePort):
        self.planetary_visual_image_service = planetary_visual_image_service

    async def execute(self, request: GetNdviByRangeRequest) -> Result[List[GetNdviImageByDayResponse], AppError]:
        date_validation = require_valid_date_range(request.dt_start, request.dt_end)
        if date_validation.is_err():
            return Result.Err(date_validation.error())

        geometry_validation = require_valid_sentinel_geometry(request.geometry)
        if geometry_validation.is_err():
            return Result.Err(geometry_validation.error())

        response = await self.planetary_visual_image_service.get_ndvi_by_range(
            dt_start=request.dt_start,
            dt_end=request.dt_end,
            geometry=request.geometry,
            cloud_percentual=request.cloud_percentual,
            generate_image=request.generate_image,
            preferred_provider=request.preferred_provider,
            satellite_collection=request.satellite_collection,
        )

        if response.is_err():
            return Result.Err(response.error())

        items = response.value()
        return Result.Ok(
            [
                GetNdviImageByDayResponse(
                    day=item.day,
                    cloud_percentual=item.cloud_percentual,
                    image_url=item.image_url,
                    ndvi_mean=item.ndvi_mean,
                    ndvi_min=item.ndvi_min,
                    ndvi_max=item.ndvi_max,
                    sat_image_id=item.sat_image_id,
                    valid_pixels=item.valid_pixels,
                    total_pixels=item.total_pixels,
                    valid_percentage=item.valid_percentage,
                    quality=item.quality,
                )
                for item in items
            ]
        )