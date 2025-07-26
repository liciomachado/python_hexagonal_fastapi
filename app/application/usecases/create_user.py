from pydantic import BaseModel
from app.domain.ports.user_repository import UserRepositoryPort
from app.domain.entities.user import User

class CreateUserRequest(BaseModel):
    name: str
    email: str

class CreateUserResponse(BaseModel):
    id: int
    name: str
    email: str

class CreateUserUseCase:
    def __init__(self, repo: UserRepositoryPort):
        self.repo = repo

    def execute(self, request: CreateUserRequest) -> CreateUserResponse:
        new_user = User(id=None, name=request.name, email=request.email)
        self.repo.create_user(new_user)
        return CreateUserResponse(id=new_user.id, name=new_user.name, email=new_user.email)