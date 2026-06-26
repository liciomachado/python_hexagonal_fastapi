import asyncio
import logging
from datetime import datetime, timedelta, timezone

from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)

from app.domain.ports.blob_storage_port import BlobStoragePort

logger = logging.getLogger("app.blob")

SAS_EXPIRY_DAYS = 7


class AzureBlobStorageService(BlobStoragePort):
    """Upload de imagens no Azure Blob Storage com URL SAS temporária de 7 dias."""

    def __init__(self, connection_string: str, container_name: str):
        self._connection_string = connection_string
        self._container_name = container_name

    async def connect(self) -> None:
        await asyncio.to_thread(self._ensure_container_exists)

    async def close(self) -> None:
        return None

    def _ensure_container_exists(self) -> None:
        client = BlobServiceClient.from_connection_string(self._connection_string)
        container = client.get_container_client(self._container_name)
        if not container.exists():
            container.create_container()

    async def upload_image_and_get_url(
        self,
        data: bytes,
        blob_name: str,
        content_type: str = "image/jpeg",
    ) -> str:
        return await asyncio.to_thread(
            self._upload_and_generate_sas_sync,
            data,
            blob_name,
            content_type,
        )

    def _upload_and_generate_sas_sync(
        self,
        data: bytes,
        blob_name: str,
        content_type: str,
    ) -> str:
        client = BlobServiceClient.from_connection_string(self._connection_string)
        blob_client = client.get_blob_client(container=self._container_name, blob=blob_name)
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )

        account_name = client.account_name
        account_key = client.credential.account_key
        if account_key is None:
            raise RuntimeError("Connection string deve conter AccountKey para gerar SAS de leitura.")

        expiry = datetime.now(timezone.utc) + timedelta(days=SAS_EXPIRY_DAYS)
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=self._container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        url = f"{blob_client.url}?{sas_token}"
        logger.info("Blob uploaded blob=%s expiry_days=%s", blob_name, SAS_EXPIRY_DAYS)
        return url
