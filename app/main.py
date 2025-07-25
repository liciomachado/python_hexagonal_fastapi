from fastapi import FastAPI
from app.api.user_routes import user_router

app = FastAPI(title="Hexagonal FastAPI Example")

app.include_router(user_router, prefix="/api")