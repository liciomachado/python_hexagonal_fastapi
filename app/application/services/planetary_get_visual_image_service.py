from abc import ABC, abstractmethod
from datetime import date
from PIL import Image, ImageDraw
import pystac
from shapely import wkt
from shapely.geometry import mapping, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
import pyproj

from pystac_client import Client
from planetary_computer import sign
import rasterio

from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse
from app.core.utils.result import AppError, BadRequestError, Result


class PlanetaryVisualImageServicePort(ABC):
    @abstractmethod
    async def get_visual_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryImageVisualResponse, AppError]:
        pass

BAND_KEYS = {
    "red": "B04",
    "green": "B03",
    "blue": "B02"
}

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
            square_parameter = 2
            center_x = (minx + maxx) / square_parameter
            center_y = (miny + maxy) / square_parameter
            square_geom = box(center_x - size / square_parameter, center_y - size / square_parameter, center_x + size / square_parameter, center_y + size / square_parameter)
            geojson_geom = mapping(square_geom)
            buffer = 0.0015  # graus
            geom_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
            minx, miny, maxx, maxy = square_geom.bounds

            # Conecta ao STAC com pystac-client
            catalog = Client.open(self.STAC_URL)

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=geojson_geom,
                datetime=f"{day.isoformat()}T00:00:00Z/{day.isoformat()}T23:59:59Z",
                max_items=10
            )

            items = list(search.items())
            if not items:
                return Result.Err("Nenhuma imagem encontrada para a data e geometria fornecidas.")

            # Ordena por menor cobertura de nuvem
            items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))

            selected = None
            for item in items:
                if item.geometry is None:
                    continue
                image_geom = shape(item.geometry)
                if geom.intersection(image_geom).area / geom.area >= cloud_percentual / 100.0:
                    selected = item
                    break

            if not selected:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))

            # Obtém os assets RGB
            try:
                assets = self.get_rgb_assets(selected)

            except KeyError as e:
                return Result.Err(BadRequestError(f"Asset RGB {e} não disponível na imagem selecionada."))

            # Chamada ao método que faz o crop e compõe a imagem RGB
            image = await self._download_crop_rgb_image(assets, geom_bounds, geom)

            return Result.Ok(PlanetaryImageVisualResponse(
                day=day,
                cloud_percentual=selected.properties.get("eo:cloud_cover", 0.0),
                base64image=image
            ))

        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem: {str(ex)}")
            
    def normalize_image_stack(self, image):
        p2 = np.percentile(image, 2)
        p98 = np.percentile(image, 98)
        image = np.clip(image, p2, p98)
        return ((image - p2) / (p98 - p2) * 255).astype(np.uint8)

    async def _download_crop_rgb_image(self, band_hrefs: dict, geom_bounds: tuple, geom: BaseGeometry) -> str:
        from PIL import ImageFilter

        bands_data = []
        transform_affine = None
        image_crs = None
        window = None

        # 1. Leitura das bandas e conversão direta para uint8 mantendo cor original
        for band_idx, band_name in enumerate(["red", "green", "blue"]):
            band_asset_key = {"red": "B04", "green": "B03", "blue": "B02"}[band_name]
            href = band_hrefs[band_asset_key]
            href = sign(href)

            with rasterio.Env():
                with rasterio.open(href) as src:
                    if band_idx == 0:
                        image_crs = src.crs
                        transform_affine = src.transform
                        geom_bounds_proj = transform_bounds("EPSG:4326", src.crs, *geom_bounds)
                        window = from_bounds(*geom_bounds_proj, transform=src.transform)
                        window = window.round_offsets().round_lengths()

                    band = src.read(1, window=window, boundless=True)
                    # Conversão direta para uint8 usando divisor fixo (ex: 3000)
                    band = np.clip(band, 0, 3000)
                    band = (band / 3000 * 255).astype(np.uint8)
                    bands_data.append(band)

        image_rgb = np.stack(bands_data, axis=-1)
        pil_img = Image.fromarray(image_rgb)

        # 3. (Opcional: Sharpening pode ser mantido ou removido, aqui mantido para leve nitidez)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)

        # --- Desenhar polígono amarelo (agora em método separado e suavizado) ---
        return self.draw_smooth_polygon_on_image(pil_img, geom, image_crs, transform_affine, window, color="yellow", width=1)

    def draw_smooth_polygon_on_image(self, pil_img, geom, image_crs, transform_affine, window, color="yellow", width=2, interp_points=500):
        """
        Desenha um polígono suavizado (interpolado) sobre a imagem PIL.
        interp_points: número de pontos interpolados para suavizar a linha.
        """
        from shapely.geometry import LineString
        draw = ImageDraw.Draw(pil_img)
        # Transforma geom para o CRS da imagem
        project = pyproj.Transformer.from_crs("EPSG:4326", image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)

        def world_to_pixel(x, y, transform_affine, window):
            col, row = ~transform_affine * (x, y)
            return (col - window.col_off, row - window.row_off)

        coords = list(mapping(geom_proj)["coordinates"][0])
        # Interpolação para suavizar a linha
        line = LineString(coords)
        if len(coords) < interp_points:
            interp_points = max(len(coords)*3, 50)
        interp_line = [line.interpolate(float(i)/interp_points, normalized=True).coords[0] for i in range(interp_points)]
        pixel_coords = []
        for coord in interp_line:
            x, y = coord[:2]
            px, py = world_to_pixel(x, y, transform_affine, window)
            pixel_coords.append((px, py))
        # Fecha o polígono
        pixel_coords.append(pixel_coords[0])
        draw.line(pixel_coords, fill=color, width=width, joint="curve")

        # 4. Aumentar resolução (fator 3, filtro LANCZOS)
        pil_img = pil_img.resize((pil_img.width * 3, pil_img.height * 3), Image.Resampling.LANCZOS)

        return self.pil_image_to_base64(pil_img)
    
    def _normalize_image_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza array de imagem de 16 bits (Sentinel-2) para 8 bits.
        """
        image = image.astype(np.float32)
        image = np.clip(image, 0, 3000)  # Faixa comum para Sentinel-2
        image = (image / 3000) * 255
        return image.astype(np.uint8)

    def pil_image_to_base64(self, pil_img: Image.Image) -> str:
        """
        Converte uma imagem PIL para base64.
        """
        import base64
        from io import BytesIO

        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def get_rgb_assets(self, item: pystac.Item) -> dict:
        return {
            "B04": item.assets["B04"].href,
            "B03": item.assets["B03"].href,
            "B02": item.assets["B02"].href
        }