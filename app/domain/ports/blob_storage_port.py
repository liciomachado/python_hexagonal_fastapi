from abc import ABC, abstractmethod


class BlobStoragePort(ABC):
    """Contrato para upload de imagens e geração de URL temporária de leitura."""

    @abstractmethod
    async def upload_image_and_get_url(
        self,
        data: bytes,
        blob_name: str,
        content_type: str = "image/jpeg",
    ) -> str:
        pass
