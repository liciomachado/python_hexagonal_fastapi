# app/core/dependencies.py

from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from app.application.usecases.validate_api_key import ValidateApiKeyUseCase
from app.core.db import SessionLocal
from app.core.utils.result import UnauthorizedError
from app.infraestructure.repository.api_key_repository import ApiKeyRepository
from app.infraestructure.repository.user_repository import UserRepository
from app.application.usecases.create_user import CreateUserUseCase

# Dependência de sessão com banco
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Injeta o repositório com o banco
def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(db)

# Injeta o caso de uso com o repositório
def get_create_user_usecase(repo: UserRepository = Depends(get_user_repository)) -> CreateUserUseCase:
    return CreateUserUseCase(repo)

def get_api_key_usecase(db: Session = Depends(get_db)) -> ValidateApiKeyUseCase:
    repo = ApiKeyRepository(db)
    return ValidateApiKeyUseCase(repo)

def validate_api_key(
    x_api_key: str = Header(..., alias="x-api-key"),
    usecase: ValidateApiKeyUseCase = Depends(get_api_key_usecase)
):
    client = usecase.execute(x_api_key)
    if client.is_err():
        error = client.error()
        if error is UnauthorizedError:
            raise HTTPException(status_code=401, detail=str(error))
        raise HTTPException(status_code=403, detail=str(error))
    return client.value() 
   