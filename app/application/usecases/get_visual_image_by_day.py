from pydantic import BaseModel
from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.application.services.stac.preferred_provider import PreferredProvider
from app.application.services.stac.satellite_collection import DEFAULT_SATELLITE_COLLECTION, SatelliteCollection
from app.application.validators.usecase_validators import require_valid_sentinel_geometry
from app.core.utils.result import AppError, Result
from datetime import date

class GetVisualImageByDayRequest(BaseModel):
    day: date
    cloud_percentual: float
    geometry: str
    preferred_provider: PreferredProvider | None = None
    satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION

class GetVisualImageByDayResponse(BaseModel):
    day: date
    cloud_percentual: float
    image_url: str

class GetVisualImageByDayUseCase:
    def __init__(self, planetary_visual_image_service: PlanetaryVisualImageServicePort):
        self.planetary_visual_image_service = planetary_visual_image_service

    async def execute(self, request: GetVisualImageByDayRequest) -> Result[GetVisualImageByDayResponse, AppError]:
        geometry_validation = require_valid_sentinel_geometry(request.geometry)
        if geometry_validation.is_err():
            return Result.Err(geometry_validation.error())

        response = await self.planetary_visual_image_service.get_visual_image(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry,
            preferred_provider=request.preferred_provider,
            satellite_collection=request.satellite_collection,
        )
        if response.is_err():
            return Result.Err(response.error())
        response = response.value()
        return Result.Ok(GetVisualImageByDayResponse(
            day=response.day,
            cloud_percentual=response.cloud_percentual,
            image_url=response.image_url
        ))
    