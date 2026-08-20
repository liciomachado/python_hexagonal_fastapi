from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.result_utils import unwrap_result
from app.application.usecases.get_all_images_by_day import (
    GetAllImagesByDayRequest,
    GetAllImagesByDayResponse,
    GetAllImagesByDayUseCase,
)
from app.application.usecases.get_images_by_range import (
    GetImagesByRangeRequest,
    GetImagesByRangeResponse,
    GetImagesByRangeUseCase,
)
from app.application.usecases.get_ndmi_image_by_day import (
    GetNdmiImageByDayRequest,
    GetNdmiImageByDayResponse,
    GetNdmiImageByDayUseCase,
)
from app.application.usecases.get_ndvi_by_range import (
    GetNdviByRangeRequest,
    GetNdviByRangeUseCase,
)
from app.application.usecases.get_ndvi_image_by_day import (
    GetNdviImageByDayRequest,
    GetNdviImageByDayResponse,
    GetNdviImageByDayUseCase,
)
from app.application.usecases.check_planetary_computer_health import (
    CheckPlanetaryComputerHealthResponse,
    CheckPlanetaryComputerHealthUseCase,
)
from app.application.usecases.get_visual_image_by_day import (
    GetVisualImageByDayRequest,
    GetVisualImageByDayResponse,
    GetVisualImageByDayUseCase,
)
from app.infraestructure.dependencies import (
    get_all_images_by_day_usecase,
    get_images_by_range_usecase,
    get_ndmi_image_by_day_usecase,
    get_ndvi_by_range_usecase,
    get_ndvi_image_by_day_usecase,
    get_planetary_health_check_usecase,
    get_visual_image_by_day_usecase,
    validate_api_key,
)

sentinel_router = APIRouter(prefix="/sentinel", tags=["images"])


@sentinel_router.get(
    "/health/planetarycomputer",
    summary="Verifica disponibilidade da API do Planetary Computer",
    response_model=CheckPlanetaryComputerHealthResponse,
)
async def planetary_computer_health(
    usecase: CheckPlanetaryComputerHealthUseCase = Depends(get_planetary_health_check_usecase),
):
    result = await usecase.execute()
    health = unwrap_result(result)
    if not health.healthy:
        return JSONResponse(status_code=503, content=health.model_dump())
    return health


@sentinel_router.post(
    "/days-available-in-range",
    summary="Obtem todas as imagens disponiveis no range definido (melhor cena por dia via cloud_cover_geometry)",
    response_model=List[GetImagesByRangeResponse],
    # dependencies=[Depends(validate_api_key)],
)
async def get_images_by_range(
    request: GetImagesByRangeRequest,
    usecase: GetImagesByRangeUseCase = Depends(get_images_by_range_usecase),
):
    images = await usecase.execute(request)
    return unwrap_result(images)


@sentinel_router.post(
    "/visual",
    summary="Obtem a imagem visual do dia",
    response_model=GetVisualImageByDayResponse,
    # dependencies=[Depends(validate_api_key)],
)
async def get_visual_image_by_day(
    request: GetVisualImageByDayRequest,
    usecase: GetVisualImageByDayUseCase = Depends(get_visual_image_by_day_usecase),
):
    visual_image = await usecase.execute(request)
    return unwrap_result(visual_image)


@sentinel_router.post(
    "/ndvi-image",
    summary="Obtem a imagem NDVI do dia",
    response_model=GetNdviImageByDayResponse,
)
async def get_ndvi_image_by_day(
    request: GetNdviImageByDayRequest,
    usecase: GetNdviImageByDayUseCase = Depends(get_ndvi_image_by_day_usecase),
):
    visual_image = await usecase.execute(request)
    return unwrap_result(visual_image)


@sentinel_router.post(
    "/ndvi-by-range",
    summary="Obtem NDVI por dia no range, filtrando por nuvem sobre o territorio",
    response_model=List[GetNdviImageByDayResponse],
)
async def get_ndvi_by_range(
    request: GetNdviByRangeRequest,
    usecase: GetNdviByRangeUseCase = Depends(get_ndvi_by_range_usecase),
):
    ndvi_images = await usecase.execute(request)
    return unwrap_result(ndvi_images)


@sentinel_router.post(
    "/ndmi-image",
    summary="Obtem a imagem NDMI do dia",
    response_model=GetNdmiImageByDayResponse,
)
async def get_ndmi_image_by_day(
    request: GetNdmiImageByDayRequest,
    usecase: GetNdmiImageByDayUseCase = Depends(get_ndmi_image_by_day_usecase),
):
    visual_image = await usecase.execute(request)
    return unwrap_result(visual_image)


@sentinel_router.post(
    "/all",
    summary="Obtem os dados de Visual, NDMI e NDVI do dia",
    response_model=GetAllImagesByDayResponse,
)
async def get_all_data_image_by_day(
    request: GetAllImagesByDayRequest,
    usecase: GetAllImagesByDayUseCase = Depends(get_all_images_by_day_usecase),
):
    visual_image = await usecase.execute(request)
    return unwrap_result(visual_image)
