from abc import ABC, abstractmethod
from app.domain.entities.user import User

class UserRepositoryPort(ABC):
    @abstractmethod
    async def create_user(self, user: User) -> User:
        pass