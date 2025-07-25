from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    database_url: str
    environment: str = "development"

    class Config:
        env_file = ".env"  # default

@lru_cache()
def get_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "development").lower()

    env_file_map = {
        "test": ".env.test",
        "staging": ".env.staging",
        "production": ".env.production",
        "development": ".env"
    }

    env_file = env_file_map.get(app_env, ".env")
    return Settings(_env_file=env_file)

# Use em toda aplicação como:
settings = get_settings()

print(f"✅ Loaded settings for env={settings.environment}: {settings.environment}")
