from app.infraestructure.repository.api_key_repository import ApiKeyRepository
from app.domain.entities.api_client import ApiClient

class ValidateApiKeyUseCase:
    def __init__(self, repo: ApiKeyRepository):
        self.repo = repo

    def execute(self, api_key: str) -> ApiClient:
        client = self.repo.get_active_client_by_key(api_key)

        if not client:
            raise Exception("API Key inválida ou inativa")

        if client.creditos <= 0:
            raise Exception("Sem créditos disponíveis")

        self.repo.debit_credit(client)
        return client
