from abc import ABC, abstractmethod
from app.domain.entities.api_client import ApiClient
from app.domain.entities.user import User

class ApiKeyRepositoryPort(ABC):
    
    @abstractmethod
    async def get_active_client_by_key(self, api_key: str) -> ApiClient | None:
        pass

    @abstractmethod
    async def debit_credit(self, client: ApiClient):
        pass