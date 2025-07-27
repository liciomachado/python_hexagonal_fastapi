from typing import List
from fastapi import APIRouter, Depends, HTTPException
from app.application.usecases.get_images_by_range import GetImagesByRangeRequest, GetImagesByRangeResponse, GetImagesByRangeUseCase
from app.application.usecases.get_visual_image_by_day import GetVisualImageByDayRequest, GetVisualImageByDayResponse, GetVisualImageByDayUseCase
from app.infraestructure.dependencies import get_images_by_range_usecase, get_visual_image_by_day_usecase, validate_api_key

sentinel_router = APIRouter(prefix="/sentinel", tags=["images"])

@sentinel_router.post(
    "",
    summary="Obtem todas as imagens disponiveis no range definido",
    response_model=List[GetImagesByRangeResponse],
    dependencies=[Depends(validate_api_key)],
)
async def get_images_by_range(request: GetImagesByRangeRequest, usecase: GetImagesByRangeUseCase = Depends(get_images_by_range_usecase)):
    images = await usecase.execute(request)
    if images.is_err():
        raise HTTPException(status_code=400, detail=str(images.error()))
    return images.value()

@sentinel_router.post(
    "/visual",
    summary="Obtem a imagem visual do dia",
    response_model=GetVisualImageByDayResponse,
    dependencies=[Depends(validate_api_key)],
)
async def get_visual_image_by_day(request: GetVisualImageByDayRequest, usecase: GetVisualImageByDayUseCase = Depends(get_visual_image_by_day_usecase)):
    visual_image = await usecase.execute(request)
    if visual_image.is_err():
        raise HTTPException(status_code=400, detail=str(visual_image.error()))
    return visual_image.value()