# app/core/dependencies.py

from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal
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
