from fastapi import APIRouter, Depends
from app.infraestructure.dependencies import get_create_user_usecase
from app.application.usecases.create_user import CreateUserUseCase
from app.core.config import Config

user_router = APIRouter(prefix="/users", tags=["users"])

@user_router.post("")
def create_user(
    name: str,
    email: str,
    usecase: CreateUserUseCase = Depends(get_create_user_usecase)
):
    user = usecase.execute(name=name, email=email)
    return user

@user_router.get("env")
def create_user():
    config = Config.ENVIRONMENT
    return config
