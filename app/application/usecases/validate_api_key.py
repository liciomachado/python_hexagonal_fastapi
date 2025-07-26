from app.core.utils.result import AppError, ForbiddenError, Result, UnauthorizedError
from app.domain.ports.apikey_repository import ApiKeyRepositoryPort
from app.domain.entities.api_client import ApiClient

class ValidateApiKeyUseCase:
    def __init__(self, repo: ApiKeyRepositoryPort):
        self.repo = repo

    async def execute(self, api_key: str) -> Result[ApiClient, AppError]:
        client = await self.repo.get_active_client_by_key(api_key)

        if not client:
            return Result.Err(UnauthorizedError("API Key inválida ou inativa"))

        if client.creditos <= 0:
            return Result.Err(ForbiddenError("Sem créditos disponíveis"))

        await self.repo.debit_credit(client)
        return Result.Ok(client)
