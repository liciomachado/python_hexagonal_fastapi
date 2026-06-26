from dataclasses import dataclass
from datetime import date

from app.application.services.dtos.planetary_ndvi_image_response import PlanetaryNdviImageResponse
from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse


@dataclass
class PlanetaryAllImagesResponse:
    visual: PlanetaryImageVisualResponse
    ndvi: PlanetaryNdviImageResponse
    ndmi: PlanetaryNdviImageResponse
