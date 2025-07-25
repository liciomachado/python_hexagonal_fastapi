from sqlalchemy.orm import Session
from app.domain.ports.user_repository import UserRepositoryPort
from app.domain.entities.user import User

class UserRepository(UserRepositoryPort):
    def __init__(self, db: Session):
        self.db = db

    def create_user(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user