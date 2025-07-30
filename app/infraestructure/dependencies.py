from collections.abc import AsyncGenerator
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, Header, Security
from sqlalchemy.orm import Session
from app.application.services.planetary_get_options_by_range import PlanetaryGetOptionImagesByRangeService
from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageService
from app.application.usecases.get_images_by_range import GetImagesByRangeUseCase
from app.application.usecases.get_ndvi_image_by_day import GetNdviImageByDayUseCase
from app.application.usecases.get_visual_image_by_day import GetVisualImageByDayUseCase
from app.application.usecases.validate_api_key import ValidateApiKeyUseCase
from app.core.db import SessionLocal
from app.core.utils.result import UnauthorizedError
from app.infraestructure.repository.api_key_repository import ApiKeyRepository
from app.infraestructure.repository.user_repository import UserRepository
from app.application.usecases.create_user import CreateUserUseCase


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session

# Injeta o repositório com o banco
def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(db)

# Injeta o caso de uso com o repositório
def get_create_user_usecase(repo: UserRepository = Depends(get_user_repository)) -> CreateUserUseCase:
    return CreateUserUseCase(repo)

def get_api_key_usecase(db: AsyncSession = Depends(get_db)) -> ValidateApiKeyUseCase:
    repo = ApiKeyRepository(db)
    return ValidateApiKeyUseCase(repo)

def get_images_by_range_usecase() -> GetImagesByRangeUseCase:
    planetary_image_service = PlanetaryGetOptionImagesByRangeService()
    return GetImagesByRangeUseCase(planetary_image_service)

def get_visual_image_by_day_usecase() -> GetVisualImageByDayUseCase:
    planetary_visual_image_service = PlanetaryVisualImageService()
    return GetVisualImageByDayUseCase(planetary_visual_image_service)

def get_ndvi_image_by_day_usecase() -> GetNdviImageByDayUseCase:
    planetary_visual_image_service = PlanetaryVisualImageService()
    return GetNdviImageByDayUseCase(planetary_visual_image_service)


API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def validate_api_key(
    x_api_key: str = Security(api_key_header),
    usecase: ValidateApiKeyUseCase = Depends(get_api_key_usecase)
):
    client = await usecase.execute(x_api_key)
    if client.is_err():
        error = client.error()
        if error is UnauthorizedError:
            raise HTTPException(status_code=401, detail=str(error))
        raise HTTPException(status_code=403, detail=str(error))
    return client.value() 
   