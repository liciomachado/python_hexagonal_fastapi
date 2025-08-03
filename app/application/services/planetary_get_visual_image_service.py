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

from rasterio.enums import Resampling

from app.application.services.dtos.planetary_ndvi_image_response import PlanetaryNdviImageResponse
from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse
from app.core.utils.result import AppError, BadRequestError, Result


class PlanetaryVisualImageServicePort(ABC):
    @abstractmethod
    async def get_ndmi_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryNdviImageResponse, AppError]:
        pass
    @abstractmethod
    async def get_visual_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryImageVisualResponse, AppError]:
        pass

    @abstractmethod
    async def get_ndvi_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryNdviImageResponse, AppError]:
        pass


class PlanetaryVisualImageService(PlanetaryVisualImageServicePort):
    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    async def get_ndmi_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryNdviImageResponse, AppError]:
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
            buffer = 0.003  # graus
            geom_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
            minx, miny, maxx, maxy = square_geom.bounds

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
            try:
                assets = self._get_ndmi_assets(selected)
            except KeyError as e:
                return Result.Err(BadRequestError(f"Asset NDMI {e} não disponível na imagem selecionada."))
            # Gera NDMI e retorna imagem + média, min e max
            image, ndmi_mean, ndmi_min, ndmi_max = await self._download_crop_ndmi_image(assets, geom_bounds, geom)
            return Result.Ok(PlanetaryNdviImageResponse(
                day=day,
                cloud_percentual=selected.properties.get("eo:cloud_cover", 0.0),
                base64image=image,
                ndvi_mean=ndmi_mean,
                ndvi_min=ndmi_min,
                ndvi_max=ndmi_max,
                sat_image_id=selected.id
            ))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem NDMI: {str(ex)}")

    async def _download_crop_ndmi_image(self, band_hrefs: dict, geom_bounds: tuple, geom: BaseGeometry):
        from PIL import ImageFilter, Image
        # 1. Abra a SWIR (B11) primeiro para referência de resolução
        swir_href = sign(band_hrefs["B11"])
        with rasterio.Env():
            with rasterio.open(swir_href) as swir_src:
                image_crs = swir_src.crs
                transform_affine = swir_src.transform
                geom_bounds_proj = transform_bounds("EPSG:4326", swir_src.crs, *geom_bounds)
                window = from_bounds(*geom_bounds_proj, transform=swir_src.transform)
                window = window.round_offsets().round_lengths()
                upscale_factor = 3
                out_height = int(window.height * upscale_factor)
                out_width = int(window.width * upscale_factor)
                crop_transform = swir_src.window_transform(window)
                crop_transform = crop_transform * crop_transform.scale(1/upscale_factor, 1/upscale_factor)
                swir = swir_src.read(1, window=window, out_shape=(out_height, out_width), resampling=Resampling.lanczos).astype(np.float32)
        # 2. Abra a NIR (B08) e reamostre para shape da SWIR
        nir_href = sign(band_hrefs["B08"])
        with rasterio.Env():
            with rasterio.open(nir_href) as nir_src:
                nir = nir_src.read(1, window=window, out_shape=(out_height, out_width), resampling=Resampling.lanczos).astype(np.float32)
        # NDMI calculation
        ndmi = (nir - swir) / (nir + swir + 1e-6)
        ndmi = np.clip(ndmi, -1, 1)

        # Calcular estatísticas apenas dentro do polígono original (geom)
        from rasterio.features import geometry_mask
        project = pyproj.Transformer.from_crs("EPSG:4326", image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)
        mask = geometry_mask([mapping(geom_proj)], out_shape=ndmi.shape, transform=crop_transform, invert=True)
        ndmi_inside = ndmi[mask]
        ndmi_inside = ndmi_inside[~np.isnan(ndmi_inside) & ~np.isinf(ndmi_inside)]
        if ndmi_inside.size > 0:
            ndmi_mean = float(np.mean(ndmi_inside))
            ndmi_min = float(np.min(ndmi_inside))
            ndmi_max = float(np.max(ndmi_inside))
        else:
            ndmi_mean = ndmi_min = ndmi_max = None

        # Aplicar colormap NDMI customizado
        ndmi_rgb = np.zeros(ndmi.shape + (3,), dtype=np.float32)
        for i in range(len(NDMI_BANDWIDTH_COLORS_VALUES) - 1):
            vmin = NDMI_BANDWIDTH_COLORS_VALUES[i]
            vmax = NDMI_BANDWIDTH_COLORS_VALUES[i + 1]
            cmin = np.array(NDMI_BANDWIDTH_COLORS[i])
            cmax = np.array(NDMI_BANDWIDTH_COLORS[i + 1])
            mask = (ndmi >= vmin) & (ndmi <= vmax)
            if np.any(mask):
                alpha = (ndmi[mask] - vmin) / (vmax - vmin + 1e-8)
                ndmi_rgb[mask] = (1 - alpha)[:, None] * cmin + alpha[:, None] * cmax

        ndmi_rgb = (ndmi_rgb * 255).astype(np.uint8)
        pil_img = Image.fromarray(ndmi_rgb, mode="RGB")
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        self._draw_smooth_polygon_on_image(pil_img, geom, image_crs, transform_affine, window, color="white", width=5)
        return self._pil_image_to_base64(pil_img), ndmi_mean, ndmi_min, ndmi_max
    
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
            buffer = 0.003  # graus
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
                assets = self._get_rgb_assets(selected)

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

    async def get_ndvi_image(self, day: date, cloud_percentual: float, geometry: str) -> Result[PlanetaryNdviImageResponse, AppError]:
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
            buffer = 0.003  # graus
            geom_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
            minx, miny, maxx, maxy = square_geom.bounds

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
            try:
                assets = self._get_ndvi_assets(selected)
            except KeyError as e:
                return Result.Err(BadRequestError(f"Asset NDVI {e} não disponível na imagem selecionada."))
            # Gera NDVI e retorna imagem + média, min e max
            image, ndvi_mean, ndvi_min, ndvi_max = await self._download_crop_ndvi_image(assets, geom_bounds, geom)
            return Result.Ok(PlanetaryNdviImageResponse(
                day=day,
                cloud_percentual=selected.properties.get("eo:cloud_cover", 0.0),
                base64image=image,
                ndvi_mean=ndvi_mean,
                ndvi_min=ndvi_min,
                ndvi_max=ndvi_max,
                sat_image_id=selected.id
            ))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem NDVI: {str(ex)}")

    async def _download_crop_ndvi_image(self, band_hrefs: dict, geom_bounds: tuple, geom: BaseGeometry):
        from PIL import ImageFilter, Image
        bands_data = []
        transform_affine = None
        image_crs = None
        window = None
        upscale_factor = 3  # Fator de aumento de resolução real
        # Leitura das bandas Red e NIR em alta resolução
        crop_transform = None
        for band_idx, band_name in enumerate(["red", "nir"]):
            band_asset_key = {"red": "B04", "nir": "B08"}[band_name]
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
                        out_height = int(window.height * upscale_factor)
                        out_width = int(window.width * upscale_factor)
                        # Transform do crop já ampliado
                        crop_transform = src.window_transform(window)
                        # Ajustar o transform para o upscale
                        crop_transform = crop_transform * crop_transform.scale(1/upscale_factor, 1/upscale_factor)
                    band = src.read(1, window=window, out_shape=(out_height, out_width), resampling=Resampling.lanczos).astype(np.float32)
                    bands_data.append(band)
        red = bands_data[0]
        nir = bands_data[1]
        # NDVI calculation
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = np.clip(ndvi, -1, 1)

        # Calcular estatísticas apenas dentro do polígono original (geom)
        from rasterio.features import geometry_mask
        # Polígono no CRS da imagem
        project = pyproj.Transformer.from_crs("EPSG:4326", image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)
        mask = geometry_mask([mapping(geom_proj)], out_shape=ndvi.shape, transform=crop_transform, invert=True)
        ndvi_inside = ndvi[mask]
        ndvi_inside = ndvi_inside[~np.isnan(ndvi_inside) & ~np.isinf(ndvi_inside)]
        if ndvi_inside.size > 0:
            ndvi_mean = float(np.mean(ndvi_inside))
            ndvi_min = float(np.min(ndvi_inside))
            ndvi_max = float(np.max(ndvi_inside))
        else:
            ndvi_mean = ndvi_min = ndvi_max = None

        # Aplicar colormap NDVI customizado
        ndvi_rgb = np.zeros(ndvi.shape + (3,), dtype=np.float32)
        for i in range(len(NDVI_BANDWIDTH_COLORS_VALUES) - 1):
            vmin = NDVI_BANDWIDTH_COLORS_VALUES[i]
            vmax = NDVI_BANDWIDTH_COLORS_VALUES[i + 1]
            cmin = np.array(BANDWIDTH_COLORS_NDVI[i])
            cmax = np.array(BANDWIDTH_COLORS_NDVI[i + 1])
            mask = (ndvi >= vmin) & (ndvi <= vmax)
            if np.any(mask):
                # Interpolação linear de cor
                alpha = (ndvi[mask] - vmin) / (vmax - vmin + 1e-8)
                ndvi_rgb[mask] = (1 - alpha)[:, None] * cmin + alpha[:, None] * cmax

        ndvi_rgb = (ndvi_rgb * 255).astype(np.uint8)
        pil_img = Image.fromarray(ndvi_rgb, mode="RGB")
        # Sharpen opcional
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        # Desenhar polígono
        self._draw_smooth_polygon_on_image(pil_img, geom, image_crs, transform_affine, window, color="white", width=5)
        # Não precisa mais aumentar resolução aqui
        return self._pil_image_to_base64(pil_img), ndvi_mean, ndvi_min, ndvi_max
        
    async def _download_crop_rgb_image(self, band_hrefs: dict, geom_bounds: tuple, geom: BaseGeometry) -> str:
        from PIL import ImageFilter

        bands_data = []
        transform_affine = None
        image_crs = None
        window = None
        upscale_factor = 3  # Fator de aumento de resolução real

        # 1. Leitura das bandas e conversão direta para uint8 mantendo cor original, já em alta resolução
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
                        out_height = int(window.height * upscale_factor)
                        out_width = int(window.width * upscale_factor)

                    band = src.read(1, window=window, out_shape=(out_height, out_width), resampling=Resampling.lanczos)
                    # Conversão direta para uint8 usando divisor fixo (ex: 3000)
                    band = np.clip(band, 0, 3000)
                    band = (band / 3000 * 255).astype(np.uint8)
                    bands_data.append(band)

        image_rgb = np.stack(bands_data, axis=-1)
        pil_img = Image.fromarray(image_rgb)

        # 3. (Opcional: Sharpening pode ser mantido ou removido, aqui mantido para leve nitidez)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)

        # Desenhar polígono já na imagem ampliada, sem resize adicional (width=5 para melhor visualização)
        return self._draw_smooth_polygon_on_image(pil_img, geom, image_crs, transform_affine, window, color="white", width=5)

    def _draw_smooth_polygon_on_image(self, pil_img, geom, image_crs, transform_affine, window, color="white", width=5, interp_points=200):
        """
        Desenha um polígono suavizado (interpolado) sobre a imagem PIL.
        interp_points: número de pontos interpolados para suavizar a linha.
        Considera o upscale da imagem para desenhar o polígono no local correto.
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
        line = LineString(coords)
        if len(coords) < interp_points:
            interp_points = max(len(coords)*3, 50)
        interp_line = [line.interpolate(float(i)/interp_points, normalized=True).coords[0] for i in range(interp_points)]
        pixel_coords = []
        # Calcular fator de escala real
        # O window.height/width é o tamanho "original" do crop, pil_img.size é o tamanho real após upscale
        if window is not None:
            orig_height = window.height
            orig_width = window.width
            img_width, img_height = pil_img.size
            scale_x = img_width / orig_width if orig_width > 0 else 1.0
            scale_y = img_height / orig_height if orig_height > 0 else 1.0
        else:
            scale_x = scale_y = 1.0
        for coord in interp_line:
            x, y = coord[:2]
            px, py = world_to_pixel(x, y, transform_affine, window)
            px *= scale_x
            py *= scale_y
            pixel_coords.append((px, py))
        # Fecha o polígono
        pixel_coords.append(pixel_coords[0])
        draw.line(pixel_coords, fill=color, width=width, joint="curve")

        return self._pil_image_to_base64(pil_img)

    def _pil_image_to_base64(self, pil_img: Image.Image) -> str:
        """
        Converte uma imagem PIL para base64.
        """
        import base64
        from io import BytesIO

        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def _get_rgb_assets(self, item: pystac.Item) -> dict:
        return {
            "B04": item.assets["B04"].href,
            "B03": item.assets["B03"].href,
            "B02": item.assets["B02"].href
        }
    
    def _get_ndvi_assets(self, item: pystac.Item) -> dict:
        return {
            "B04": item.assets["B04"].href,  # Red
            "B08": item.assets["B08"].href   # NIR
        }

    def _get_ndmi_assets(self, item: pystac.Item) -> dict:
        return {
            "B08": item.assets["B08"].href,  # NIR
            "B11": item.assets["B11"].href   # SWIR
        }
# NDVI colormap
NDVI_BANDWIDTH_COLORS_VALUES = [
    -1.0,
    -0.506082,
    -0.180048,
    0.10949,
    0.309002,
    0.416058,
    0.554744,
    0.73236,
    1.0
]
BANDWIDTH_COLORS_NDVI = [
    (139 / 255, 3 / 255, 6 / 255),
    (215 / 255, 26 / 255, 28 / 255),
    (216 / 255, 77 / 255, 29 / 255),
    (218 / 255, 82 / 255, 33 / 255),
    (253 / 255, 174 / 255, 97 / 255),
    (255 / 255, 255 / 255, 191 / 255),
    (171 / 255, 221 / 255, 164 / 255),
    (43 / 255, 186 / 255, 64 / 255),
    (28 / 255, 120 / 255, 40 / 255),
]

ZERO_DIVISOR_FIX = np.iinfo(np.uint16).max * 2
NDMI_BANDWIDTH_COLORS = [
    (60 / 255, 29 / 255, 18 / 255),      # rgb(60, 29, 18)
    (109 / 255, 64 / 255, 44 / 255),     # rgb(109, 64, 44)
    (149 / 255, 87 / 255, 61 / 255),     # rgb(149, 87, 61)
    (207 / 255, 135 / 255, 104 / 255),   # rgb(207, 135, 104)
    (218 / 255, 229 / 255, 237 / 255),   # rgb(218, 229, 237)
    (94 / 255, 174 / 255, 240 / 255),    # rgb(94, 174, 240)
    (79 / 255, 150 / 255, 235 / 255),    # rgb(79, 150, 235)
    (52 / 255, 113 / 255, 214 / 255),    # rgb(52, 113, 214)
    (16 / 255, 69 / 255, 185 / 255)      # rgb(16, 69, 185)
]
NDMI_BANDWIDTH_COLORS_VALUES = [
    -1.0,
    -0.698296,
    -0.44039,
    -0.216546,
    0.00730000000000008,
    0.22871,
    0.462288,
    0.729928,
    1.0
]


def apply_filters(index: np.ndarray) -> np.ndarray:
    """
    Apply filters to a NumPy array by modifying its values based on specific conditions.

    Parameters:
    -----------
    index : ndarray
        A NumPy array containing the data to be filtered.

    Returns:
    --------
    ndarray
        The filtered NumPy array with the following transformations:
    """
    index[index > 1] = 1.0
    index[index < -1] = -1.0
    index[index == 0] = np.nan
    return index

def calc_ndmi(b_nir: np.ndarray, b_swir: np.ndarray) -> np.ndarray | list:
    """
    Calculate the Normalized Difference Moisture Index (NDMI) for arrays of reflectance values.

    NDMI is a measure of vegetation moisture content. It is calculated using the formula:
    NDMI = (NIR - SWIR) / (NIR + SWIR)

    Parameters:
    b_nir (np.ndarray): An array of reflectance values in the near-infrared band.
    b_swir (np.ndarray): An array of reflectance values in the shortwave infrared band.

    Returns:
    np.ndarray: An array of NDMI values, which range from -1 to 1.
                - Negative values generally indicate low moisture content or bare soil.
                - Values around 0 suggest intermediate moisture.
                - Positive values closer to 1 indicate higher moisture content in vegetation.
                - np.nan is being used to hide 0 values as a mask.
    """
    if len(b_nir) == 0 or len(b_swir) == 0:
        return []

    b_nir = b_nir.astype(float)
    b_swir = b_swir.astype(float)

    denominator = b_nir + b_swir
    denominator[denominator == 0] = ZERO_DIVISOR_FIX  # Fixing division by zero

    with np.errstate(divide="ignore", invalid="ignore"):
        ndmi = np.where(denominator != 0, (b_nir - b_swir) / denominator, 0)

    return apply_filters(ndmi)

def calc_ndvi(b_nir: np.ndarray, b_red: np.ndarray) -> np.ndarray | list:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) for arrays of reflectance values.

    NDVI is a measure of vegetation health and density. It is calculated using the formula:
    NDVI = (NIR - RED) / (NIR + RED)

    Parameters:
    b_nir (np.ndarray): An array of reflectance values in the near-infrared band.
    b_red (np.ndarray): An array of reflectance values in the red band.

    Returns:
    np.ndarray: An array of NDVI values, which range from -1 to 1.
                - Negative values generally indicate non-vegetated surfaces (e.g., water, barren land).
                - Values around 0 suggest sparse or no vegetation.
                - Positive values closer to 1 indicate healthy, dense vegetation.
                - np.nan is being used to hide 0 values as a mask
    """
    if len(b_nir) == 0 or len(b_red) == 0:
        return []

    b_nir = b_nir.astype(float)
    b_red = b_red.astype(float)

    denominator = b_nir + b_red
    denominator[denominator == 0] = ZERO_DIVISOR_FIX

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denominator != 0, (b_nir - b_red) / denominator, 0)

    return apply_filters(ndvi)