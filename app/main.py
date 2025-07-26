from fastapi import FastAPI
from app.api.user_routes import user_router

app = FastAPI(title="Hexagonal FastAPI Example")

app.include_router(user_router, prefix="/api")


#execute a applicação com o comando: 'uvicorn app.main:app --reload'  na raiz do projeto
#em test: 'set ENV=test && uvicorn app.main:app --reload' na raiz do projeto
#alterar o ENV para o ambiente desejado de acordo com os arquivos .env existentes

#para rodar o docker, execute o comando: docker compose up --build