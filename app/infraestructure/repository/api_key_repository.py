from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.entities.api_client import ApiClient
from app.domain.ports.apikey_repository import ApiKeyRepositoryPort

class ApiKeyRepository(ApiKeyRepositoryPort):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_active_client_by_key(self, api_key: str) -> ApiClient | None:
        result = await self.db.execute(
            select(ApiClient).where(ApiClient.api_key == api_key, ApiClient.ativo == True)
        )
        return result.scalars().first()

    async def debit_credit(self, client: ApiClient):
        if client.creditos <= 0:
            raise Exception("Sem créditos")
        client.creditos -= 1
        self.db.add(client)
        await self.db.commit()
