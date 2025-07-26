from sqlalchemy import Column, String, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid
from app.core.db import Base

class ApiClient(Base):
    __tablename__ = "api_clients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nome = Column(String)
    api_key = Column(String, unique=True, nullable=False)
    creditos = Column(Integer, nullable=False, default=0)
    ativo = Column(Boolean, default=True, nullable=False)
