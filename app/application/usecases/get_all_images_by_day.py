
from datetime import date

from pydantic import BaseModel

from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageServicePort
from app.core.utils.result import AppError, Result


class GetAllImagesByDayRequest(BaseModel):
    day: date
    cloud_percentual: float
    geometry: str
    generate_image: bool = True


from .get_visual_image_by_day import GetVisualImageByDayResponse, GetVisualImageByDayUseCase, GetVisualImageByDayRequest
from .get_ndvi_image_by_day import GetNdviImageByDayResponse, GetNdviImageByDayUseCase, GetNdviImageByDayRequest
from .get_ndmi_image_by_day import GetNdmiImageByDayResponse, GetNdmiImageByDayUseCase, GetNdmiImageByDayRequest
import asyncio

class GetAllImagesByDayResponse(BaseModel):
    visual: GetVisualImageByDayResponse
    ndvi: GetNdviImageByDayResponse
    ndmi: GetNdmiImageByDayResponse

class GetAllImagesByDayUseCase:
    def __init__(self, planetary_visual_image_service: PlanetaryVisualImageServicePort):
        self.planetary_visual_image_service = planetary_visual_image_service

    async def execute(self, request: GetAllImagesByDayRequest) -> Result[GetAllImagesByDayResponse, AppError]:
        # Instanciar os use cases
        visual_uc = GetVisualImageByDayUseCase(self.planetary_visual_image_service)
        ndvi_uc = GetNdviImageByDayUseCase(self.planetary_visual_image_service)
        ndmi_uc = GetNdmiImageByDayUseCase(self.planetary_visual_image_service)

        # Preparar requests
        visual_req = GetVisualImageByDayRequest(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry
        )
        ndvi_req = GetNdviImageByDayRequest(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry,
            generate_image=request.generate_image
        )
        ndmi_req = GetNdmiImageByDayRequest(
            day=request.day,
            cloud_percentual=request.cloud_percentual,
            geometry=request.geometry,
            generate_image=request.generate_image
        )

        # Executar em paralelo
        results = await asyncio.gather(
            visual_uc.execute(visual_req),
            ndvi_uc.execute(ndvi_req),
            ndmi_uc.execute(ndmi_req),
            return_exceptions=True
        )

        # Checar erros e acessar Result corretamente
        visual_result, ndvi_result, ndmi_result = results
        # Se algum for Exception, retorna erro
        # Checar se algum resultado é Exception
        for r in [visual_result, ndvi_result, ndmi_result]:
            if isinstance(r, Exception):
                return Result.Err(str(r))
        # Checar se algum resultado é erro do Result
        
        # Montar resposta agregada
        response = GetAllImagesByDayResponse(
            visual=visual_result.value(),
            ndvi=ndvi_result.value(),
            ndmi=ndmi_result.value()
        )
        return Result.Ok(response)
        