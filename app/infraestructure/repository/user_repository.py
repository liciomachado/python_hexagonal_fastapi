from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.ports.user_repository import UserRepositoryPort
from app.domain.entities.user import User

class UserRepository(UserRepositoryPort):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_user(self, user: User) -> User:
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user