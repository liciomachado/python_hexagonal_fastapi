from app.domain.ports.user_repository import UserRepositoryPort
from app.domain.entities.user import User

class CreateUserUseCase:
    def __init__(self, repo: UserRepositoryPort):
        self.repo = repo

    def execute(self, name: str, email: str) -> User:
        new_user = User(id=None, name=name, email=email)
        return self.repo.create_user(new_user)