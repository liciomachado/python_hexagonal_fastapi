from abc import ABC, abstractmethod
from datetime import date
from io import BytesIO
import base64
from PIL import Image
from shapely import wkt
from shapely.geometry import mapping, box, shape
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
import httpx

from pystac_client import Client
from planetary_computer import sign
import rasterio

from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse
from app.core.utils.result import AppError, BadRequestError, Result


class PlanetaryVisualImageServicePort(ABC):
    @abstractmethod
    async def get_visual_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryImageVisualResponse, AppError]:
        pass


class PlanetaryVisualImageService(PlanetaryVisualImageServicePort):
    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    async def get_visual_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryImageVisualResponse, AppError]:
        try:
            geom = wkt.loads(geometry)
            bounds = geom.bounds
            minx, miny, maxx, maxy = bounds
            width = maxx - minx
            height = maxy - miny
            size = max(width, height)
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            square_geom = box(center_x - size / 2, center_y - size / 2, center_x + size / 2, center_y + size / 2)
            geojson_geom = mapping(square_geom)
            minx, miny, maxx, maxy = square_geom.bounds

            # Conecta ao STAC com pystac-client
            catalog = Client.open(self.STAC_URL)

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=geojson_geom,
                datetime=f"{day.isoformat()}T00:00:00Z/{day.isoformat()}T23:59:59Z",
                max_items=10
            )

            items = list(search.get_items())
            if not items:
                return Result.Err("Nenhuma imagem encontrada para a data e geometria fornecidas.")

            # Ordena por menor cobertura de nuvem
            items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))

            selected = None
            for item in items:
                image_geom = shape(item.geometry)
                if geom.intersection(image_geom).area / geom.area >= cloud_percentual / 100.0:
                    selected = item
                    break

            if not selected:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))

            visual_asset = selected.assets.get("visual")
            if not visual_asset:
                return Result.Err(BadRequestError("Imagem visual não disponível."))

            signed_url = sign(visual_asset.href)

            # Tenta abrir com rasterio
            image = await self._download_crop_image(signed_url, (minx, miny, maxx, maxy))

            return Result.Ok(PlanetaryImageVisualResponse(
                day=day,
                cloud_percentual=selected.properties.get("eo:cloud_cover", 0.0),
                base64image=image
            ))

        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem: {str(ex)}")
    
    async def _download_crop_image(self, signed_url: str, geom_bounds: tuple):
        with rasterio.Env():
            with rasterio.open(signed_url) as src:
                geom_bounds_proj = transform_bounds("EPSG:4326", src.crs, *geom_bounds)
                window = from_bounds(*geom_bounds_proj, transform=src.transform)
                window = window.round_offsets().round_lengths()

                # Lê a janela com precisão (ex: Sentinel geralmente em uint16)
                image = src.read(window=window)

                # Normaliza cada banda para uint8 (0-255)
                def normalize(band):
                    return ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)

                image = self.normalize_image_stack(image)

                # Reordena e aumenta resolução
                image = np.moveaxis(image, 0, -1)
                scale_factor = 4
                new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
                pil_img = Image.fromarray(image).resize(new_size, Image.Resampling.LANCZOS)

                return self.pil_image_to_base64(pil_img)
            
    def normalize_image_stack(self, image):
        p2 = np.percentile(image, 2)
        p98 = np.percentile(image, 98)
        image = np.clip(image, p2, p98)
        return ((image - p2) / (p98 - p2) * 255).astype(np.uint8)

    def pil_image_to_base64(self, pil_img: Image.Image, format: str = "JPEG") -> str:
        buffered = BytesIO()
        pil_img.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64