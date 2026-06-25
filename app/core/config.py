import os
from dotenv import load_dotenv
from pathlib import Path

# Detecta o ambiente: development, test, staging, production
ENV = os.getenv("ENV", "development")

# Caminho do .env.{ENV}
env_path = Path(".") / f".env.{ENV}"

# Carrega o arquivo correspondente
load_dotenv(dotenv_path=env_path)

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    STAC_EARTHSEARCH_URL = os.getenv(
        "STAC_EARTHSEARCH_URL",
        "https://earth-search.aws.element84.com/v1/search",
    )
    STAC_BREAKER_OPEN_SECONDS = int(os.getenv("STAC_BREAKER_OPEN_SECONDS", "300"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
