import os
from dotenv import load_dotenv
from pathlib import Path

ENV = os.getenv("ENV", "development")
env_path = Path(".") / f".env.{ENV}"
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

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

    AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
    AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "sentinel-images")

    IMAGE_POLYGON_INTERP_POINTS = int(os.getenv("IMAGE_POLYGON_INTERP_POINTS", "80"))
    IMAGE_POLYGON_BORDER_WIDTH = int(os.getenv("IMAGE_POLYGON_BORDER_WIDTH", "10"))
    IMAGE_ENABLE_SHARPEN = os.getenv("IMAGE_ENABLE_SHARPEN", "false").lower() == "true"
    SCL_MAX_DIMENSION = int(os.getenv("SCL_MAX_DIMENSION", "512"))
    VSI_CACHE_SIZE = os.getenv("VSI_CACHE_SIZE", "134217728")