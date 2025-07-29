from abc import ABC, abstractmethod
from datetime import date
import base64
from io import BytesIO
from PIL import Image
from shapely import wkt
from shapely.geometry import mapping, box, shape
import httpx
from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse
import rasterio
from rasterio.session import AWSSession
from rasterio.io import MemoryFile
from urllib.parse import quote

from app.core.utils.result import AppError, Result

class PlanetaryVisualImageServicePort(ABC):
    @abstractmethod
    async def get_visual_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryImageVisualResponse, AppError]:
        pass

class PlanetaryVisualImageService(PlanetaryVisualImageServicePort):
    BASE_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

    async def get_visual_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryImageVisualResponse, AppError]:
        geom = wkt.loads(geometry)
        bounds = geom.bounds
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny
        size = max(width, height)
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        percentual_cloud = cloud_percentual / 100.0

        # Cria bounding box quadrada ao redor do centro
        square_geom = box(center_x - size/2, center_y - size/2, center_x + size/2, center_y + size/2)
        geojson_geom = mapping(square_geom)

        # Busca imagens do dia
        payload = {
            "collections": ["sentinel-2-l2a"],
            "intersects": geojson_geom,
            "datetime": f"{day.isoformat()}T00:00:00Z/{day.isoformat()}T23:59:59Z",
            "limit": 10
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.BASE_URL, json=payload)
            response.raise_for_status()
            features = response.json().get("features", [])

        if not features:
            raise ValueError("Nenhuma imagem encontrada para a data e geometria fornecidas.")

        # Seleciona a imagem com menor nuvem e maior interseção
        selected = None
        coverage_threshold = percentual_cloud 
        for feature in sorted(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100)):
            image_geom = shape(feature["geometry"])
            if geom.intersection(image_geom).area / geom.area >= coverage_threshold:
                selected = feature
                break

        if not selected:
            return Result.Err(f"Nenhuma imagem cobre ao menos {percentual_cloud * 100}% da geometria.")

        asset_url = selected["assets"]["visual"]["href"]
        signed_url = await self.get_signed_url_if_needed(asset_url)

        # Abre imagem com rasterio + memoryfile
        async with httpx.AsyncClient(timeout=60) as client:
            img_response = await client.get(signed_url)
            img_response.raise_for_status()
            img_bytes = img_response.content

        with MemoryFile(img_bytes) as memfile:
            with memfile.open() as dataset:
                img = dataset.read([1, 2, 3])  # RGB
                img = img.transpose(1, 2, 0)   # CxHxW → HxWxC
                pil_img = Image.fromarray(img.astype('uint8'))
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                base64_img = base64.b64encode(buffered.getvalue()).decode()

        return Result.Ok(PlanetaryImageVisualResponse(
            day=day,
            cloud_percentual=selected["properties"].get("eo:cloud_cover", 0.0),
            base64image=base64_img
        ))

    async def get_signed_url_if_needed(self, asset_url: str) -> str:
        # Se a URL já tem SAS token (verificamos por "sig="), não precisa assinar
        if "sig=" in asset_url:
            return asset_url

        # Caso contrário, assina via Planetary Computer
        encoded_url = quote(asset_url, safe='')
        sign_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={encoded_url}"
        async with httpx.AsyncClient(timeout=30) as client:
            sign_response = await client.get(sign_url)
            sign_response.raise_for_status()
            signed_url = sign_response.json()["href"]
            return signed_url
