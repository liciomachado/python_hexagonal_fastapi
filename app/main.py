from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.middleware.correlation_id import CorrelationIdMiddleware
from app.api.middleware.request_logging import RequestLoggingMiddleware
from app.api.user_routes import user_router
from app.api.sentinel_routes import sentinel_router
from app.core.lifespan import app_lifespan
from app.core.logging_config import setup_logging

setup_logging()

app = FastAPI(title="Hexagonal FastAPI Example", lifespan=app_lifespan)

register_error_handlers(app)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(CorrelationIdMiddleware)

app.include_router(user_router, prefix="/api")
app.include_router(sentinel_router, prefix="/api")


#execute a applicação com o comando: 'uvicorn app.main:app --reload'  na raiz do projeto
#em test: 'set ENV=test && uvicorn app.main:app --reload' na raiz do projeto
#alterar o ENV para o ambiente desejado de acordo com os arquivos .env existentes

#para rodar o docker, execute o comando: docker compose up --build
