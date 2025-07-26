# app/infraestructure/repository/api_key_repository.py

from sqlalchemy.orm import Session
from app.domain.entities.api_client import ApiClient
from app.domain.ports.apikey_repository import ApiKeyRepositoryPort

class ApiKeyRepository(ApiKeyRepositoryPort):
    def __init__(self, db: Session):
        self.db = db

    def get_active_client_by_key(self, api_key: str) -> ApiClient | None:
        return self.db.query(ApiClient).filter_by(api_key=api_key, ativo=True).first()

    def debit_credit(self, client: ApiClient):
        if client.creditos <= 0:
            raise Exception("Sem créditos")
        client.creditos -= 1
        self.db.commit()
