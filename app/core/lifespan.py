from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.http_client import close_shared_http_client
from app.infraestructure.blob_factory import get_blob_storage_service
from app.infraestructure.cache_factory import get_cache_service


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    cache = get_cache_service()
    await cache.connect()

    blob = get_blob_storage_service()
    if blob is not None and hasattr(blob, "connect"):
        await blob.connect()

    yield

    await cache.close()
    if blob is not None and hasattr(blob, "close"):
        await blob.close()
    await close_shared_http_client()
