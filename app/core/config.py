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
    # ENV = ENV
    # DEBUG = os.getenv("DEBUG", "False").lower() in ["true", "1", "yes"]
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    # CAR_FUNCTION_API_KEY = os.getenv("CAR_FUNCTION_API_KEY", "")
    # RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
    # RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
    # RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")