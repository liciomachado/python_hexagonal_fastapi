from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal
from app.infraestructure.repository.user_repository import UserRepository
from app.application.usecases.create_user import CreateUserUseCase

user_router = APIRouter(prefix="/users", tags=["users"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@user_router.post("")
def create_user(name: str, email: str, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    usecase = CreateUserUseCase(repo)
    user = usecase.execute(name=name, email=email)
    return user
