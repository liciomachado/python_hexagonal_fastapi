from typing import List
from fastapi import APIRouter, Depends, HTTPException
from app.application.usecases.get_images_by_range import GetImagesByRangeRequest, GetImagesByRangeResponse, GetImagesByRangeUseCase
from app.infraestructure.dependencies import get_images_by_range_usecase, validate_api_key

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

