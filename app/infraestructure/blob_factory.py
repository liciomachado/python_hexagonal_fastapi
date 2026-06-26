from app.domain.ports.blob_storage_port import BlobStoragePort
from app.infraestructure.storage.azure_blob_storage_service import AzureBlobStorageService
from app.core.config import Config

_blob_storage_instance: BlobStoragePort | None = None


def get_blob_storage_service() -> BlobStoragePort | None:
    global _blob_storage_instance
    if not Config.AZURE_BLOB_CONNECTION_STRING:
        return None
    if _blob_storage_instance is None:
        _blob_storage_instance = AzureBlobStorageService(
            connection_string=Config.AZURE_BLOB_CONNECTION_STRING,
            container_name=Config.AZURE_BLOB_CONTAINER_NAME,
        )
    return _blob_storage_instance
