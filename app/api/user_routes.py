from fastapi import APIRouter, Depends
from app.infraestructure.dependencies import get_create_user_usecase, validate_api_key
from app.application.usecases.create_user import CreateUserRequest, CreateUserResponse, CreateUserUseCase
from app.core.config import Config

user_router = APIRouter(prefix="/users", tags=["users"])

@user_router.post("", response_model=CreateUserResponse, dependencies=[Depends(validate_api_key)])
async def create_user(request: CreateUserRequest, usecase: CreateUserUseCase = Depends(get_create_user_usecase)):
    user = await usecase.execute(request)
    return user

@user_router.get("/env")
def get_env():
    config = Config.ENVIRONMENT
    return config
