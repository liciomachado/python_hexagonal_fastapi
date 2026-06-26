from abc import ABC, abstractmethod
import asyncio
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
import rasterio

from rasterio.enums import Resampling

from app.application.services.dtos.planetary_ndvi_image_response import PlanetaryNdviImageResponse
from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse
from app.application.services.stac.preferred_provider import PreferredProvider
from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.application.services.stac.stac_types import StacProviderName, resolve_band_href
from app.core.utils.result import AppError, BadRequestError, Result

# Sentinel-2 SCL: cloud shadow (3), medium/high cloud (8, 9), thin cirrus (10)
SCL_CLOUD_CLASSES = frozenset({3, 8, 9, 10})

# Dimensões adequadas para exibição em relatório PDF (~150 DPI em A4)
REPORT_MAX_IMAGE_DIMENSION = 1200
REPORT_JPEG_QUALITY = 85
READ_RESAMPLE_FACTOR = 1

# Otimiza leitura de COGs remotos via HTTP (Azure Blob / vsicurl)
RASTERIO_GDAL_CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
}


class PlanetaryVisualImageServicePort(ABC):
    @abstractmethod
    async def get_ndmi_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        pass

    @abstractmethod
    async def get_visual_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        preferred_provider: PreferredProvider | None = None,
    ) -> Result[PlanetaryImageVisualResponse, AppError]:
        pass

    @abstractmethod
    async def get_ndvi_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        pass


class PlanetaryVisualImageService(PlanetaryVisualImageServicePort):
    def __init__(self, stac_facade: StacResilientFacade):
        self._stac_facade = stac_facade

    async def get_ndmi_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        try:
            geom, geojson_geom, geom_bounds = self.map_geom(geometry)
            selected, provider = await self._search_selected_item(
                day=day,
                cloud_percentual=cloud_percentual,
                geom=geom,
                geojson_geom=geojson_geom,
                preferred_provider=preferred_provider,
            )
            if selected is None:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))
            try:
                assets = self._get_ndmi_assets(selected)
            except KeyError as e:
                return Result.Err(BadRequestError(f"Asset NDMI {e} não disponível na imagem selecionada."))
            (image, ndmi_mean, ndmi_min, ndmi_max), geometry_cloud_percentual = await asyncio.gather(
                asyncio.to_thread(
                    self._download_crop_ndmi_image,
                    assets, geom_bounds, geom, generate_image, provider,
                ),
                asyncio.to_thread(
                    self._compute_cloud_percentual_over_geometry,
                    selected, geom, geom_bounds, provider,
                ),
            )
            return Result.Ok(PlanetaryNdviImageResponse(
                day=day,
                cloud_percentual=geometry_cloud_percentual,
                base64image=image,
                ndvi_mean=ndmi_mean,
                ndvi_min=ndmi_min,
                ndvi_max=ndmi_max,
                sat_image_id=selected.id
            ))
        except ValueError as ex:
            return Result.Err(str(ex))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem NDMI: {str(ex)}")
    
    async def get_visual_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        preferred_provider: PreferredProvider | None = None,
    ) -> Result[PlanetaryImageVisualResponse, AppError]:
        try:
            geom, geojson_geom, geom_bounds = self.map_geom(geometry)
            selected, provider = await self._search_selected_item(
                day=day,
                cloud_percentual=cloud_percentual,
                geom=geom,
                geojson_geom=geojson_geom,
                preferred_provider=preferred_provider,
            )
            if selected is None:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))
            try:
                assets = self._get_rgb_assets(selected)
            except KeyError as e:
                return Result.Err(BadRequestError(f"Asset RGB {e} não disponível na imagem selecionada."))
            image, geometry_cloud_percentual = await asyncio.gather(
                asyncio.to_thread(
                    self._download_crop_rgb_image,
                    assets, geom_bounds, geom, provider,
                ),
                asyncio.to_thread(
                    self._compute_cloud_percentual_over_geometry,
                    selected, geom, geom_bounds, provider,
                ),
            )

            return Result.Ok(PlanetaryImageVisualResponse(
                day=day,
                cloud_percentual=geometry_cloud_percentual,
                base64image=image
            ))

        except ValueError as ex:
            return Result.Err(str(ex))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem: {str(ex)}")

    async def get_ndvi_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        try:
            geom, geojson_geom, geom_bounds = self.map_geom(geometry)
            selected, provider = await self._search_selected_item(
                day=day,
                cloud_percentual=cloud_percentual,
                geom=geom,
                geojson_geom=geojson_geom,
                preferred_provider=preferred_provider,
            )
            if selected is None:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))
            try:
                assets = self._get_ndvi_assets(selected)
            except KeyError as e:
                return Result.Err(BadRequestError(f"Asset NDVI {e} não disponível na imagem selecionada."))
            (image, ndvi_mean, ndvi_min, ndvi_max), geometry_cloud_percentual = await asyncio.gather(
                asyncio.to_thread(
                    self._download_crop_ndvi_image,
                    assets, geom_bounds, geom, generate_image, provider,
                ),
                asyncio.to_thread(
                    self._compute_cloud_percentual_over_geometry,
                    selected, geom, geom_bounds, provider,
                ),
            )
            return Result.Ok(PlanetaryNdviImageResponse(
                day=day,
                cloud_percentual=geometry_cloud_percentual,
                base64image=image,
                ndvi_mean=ndvi_mean,
                ndvi_min=ndvi_min,
                ndvi_max=ndvi_max,
                sat_image_id=selected.id
            ))
        except ValueError as ex:
            return Result.Err(str(ex))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem NDVI: {str(ex)}")

    async def _search_selected_item(
        self,
        day: date,
        cloud_percentual: float,
        geom: BaseGeometry,
        geojson_geom: dict,
        preferred_provider: PreferredProvider | None,
    ) -> tuple[pystac.Item | None, StacProviderName]:
        provider_enum = self._parse_preferred_provider(preferred_provider)
        search_result = await self._stac_facade.search_items_by_day(
            geojson_geom=geojson_geom,
            day=day,
            max_items=10,
            preferred_provider=provider_enum,
        )
        items = search_result.items
        if not items:
            raise ValueError("Nenhuma imagem encontrada para a data e geometria fornecidas.")
        items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))
        for item in items:
            if item.geometry is None:
                continue
            image_geom = shape(item.geometry)
            if geom.intersection(image_geom).area / geom.area >= cloud_percentual / 100.0:
                return item, search_result.provider
        return None, search_result.provider

    def _compute_cloud_percentual_over_geometry(
        self,
        item: pystac.Item,
        geom: BaseGeometry,
        geom_bounds: tuple,
        provider: StacProviderName,
    ) -> float:
        from rasterio.features import geometry_mask

        try:
            scl_href = self._sign_url(resolve_band_href(item, "SCL"), provider)
        except KeyError:
            return float(item.properties.get("eo:cloud_cover", 0.0))

        with rasterio.Env(**RASTERIO_GDAL_CONFIG):
            with rasterio.open(scl_href) as src:
                image_crs = src.crs
                geom_bounds_proj = transform_bounds("EPSG:4326", src.crs, *geom_bounds)
                window = from_bounds(*geom_bounds_proj, transform=src.transform)
                window = window.round_offsets().round_lengths()
                scl = src.read(1, window=window)
                crop_transform = src.window_transform(window)

        project = pyproj.Transformer.from_crs("EPSG:4326", image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)
        mask = geometry_mask([mapping(geom_proj)], out_shape=scl.shape, transform=crop_transform, invert=True)
        scl_inside = scl[mask]
        if scl_inside.size == 0:
            return float(item.properties.get("eo:cloud_cover", 0.0))

        cloud_pixels = int(np.isin(scl_inside, list(SCL_CLOUD_CLASSES)).sum())
        return round(cloud_pixels / scl_inside.size * 100, 2)

    def _prepare_image_for_report(self, pil_img: Image.Image) -> Image.Image:
        width, height = pil_img.size
        max_dim = max(width, height)
        if max_dim <= REPORT_MAX_IMAGE_DIMENSION:
            return pil_img
        scale = REPORT_MAX_IMAGE_DIMENSION / max_dim
        new_size = (int(width * scale), int(height * scale))
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)

    def _parse_preferred_provider(self, preferred_provider: PreferredProvider | None) -> StacProviderName | None:
        if preferred_provider is None:
            return None
        return StacProviderName(preferred_provider)

    def _sign_url(self, href: str, provider: StacProviderName) -> str:
        return self._stac_facade.sign_asset_url(href, provider)

    def _download_crop_ndmi_image(
        self,
        band_hrefs: dict,
        geom_bounds: tuple,
        geom: BaseGeometry,
        generate_image: bool,
        provider: StacProviderName,
    ):
        from PIL import ImageFilter, Image
        # 1. Abra a SWIR (B11) primeiro para referência de resolução
        swir_href = self._sign_url(band_hrefs["B11"], provider)
        with rasterio.Env(**RASTERIO_GDAL_CONFIG):
            with rasterio.open(swir_href) as swir_src:
                image_crs = swir_src.crs
                transform_affine = swir_src.transform
                geom_bounds_proj = transform_bounds("EPSG:4326", swir_src.crs, *geom_bounds)
                window = from_bounds(*geom_bounds_proj, transform=swir_src.transform)
                window = window.round_offsets().round_lengths()
                upscale_factor = READ_RESAMPLE_FACTOR
                out_height = max(1, int(window.height * upscale_factor))
                out_width = max(1, int(window.width * upscale_factor))
                crop_transform = swir_src.window_transform(window)
                crop_transform = crop_transform * crop_transform.scale(1/upscale_factor, 1/upscale_factor)
                resampling = Resampling.lanczos if upscale_factor != 1 else Resampling.nearest
                swir = swir_src.read(1, window=window, out_shape=(out_height, out_width), resampling=resampling).astype(np.float32)
        # 2. Abra a NIR (B08) e reamostre para shape da SWIR
        nir_href = self._sign_url(band_hrefs["B08"], provider)
        with rasterio.Env(**RASTERIO_GDAL_CONFIG):
            with rasterio.open(nir_href) as nir_src:
                nir = nir_src.read(1, window=window, out_shape=(out_height, out_width), resampling=resampling).astype(np.float32)
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

        if not generate_image:
            return None, ndmi_mean, ndmi_min, ndmi_max

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
        pil_img = self._draw_smooth_polygon_on_image(pil_img, geom, image_crs, transform_affine, window, color="white", width=3)
        pil_img = self._prepare_image_for_report(pil_img)
        return self._pil_image_to_base64(pil_img), ndmi_mean, ndmi_min, ndmi_max
    
    def _download_crop_ndvi_image(
        self,
        band_hrefs: dict,
        geom_bounds: tuple,
        geom: BaseGeometry,
        generate_image: bool,
        provider: StacProviderName,
    ):
        from PIL import ImageFilter, Image
        bands_data = []
        transform_affine = None
        image_crs = None
        window = None
        upscale_factor = READ_RESAMPLE_FACTOR
        crop_transform = None
        for band_idx, band_name in enumerate(["red", "nir"]):
            band_asset_key = {"red": "B04", "nir": "B08"}[band_name]
            href = band_hrefs[band_asset_key]
            href = self._sign_url(href, provider)
            with rasterio.Env(**RASTERIO_GDAL_CONFIG):
                with rasterio.open(href) as src:
                    if band_idx == 0:
                        image_crs = src.crs
                        transform_affine = src.transform
                        geom_bounds_proj = transform_bounds("EPSG:4326", src.crs, *geom_bounds)
                        window = from_bounds(*geom_bounds_proj, transform=src.transform)
                        window = window.round_offsets().round_lengths()
                        out_height = max(1, int(window.height * upscale_factor))
                        out_width = max(1, int(window.width * upscale_factor))
                        crop_transform = src.window_transform(window)
                        crop_transform = crop_transform * crop_transform.scale(1/upscale_factor, 1/upscale_factor)
                    resampling = Resampling.lanczos if upscale_factor != 1 else Resampling.nearest
                    band = src.read(1, window=window, out_shape=(out_height, out_width), resampling=resampling).astype(np.float32)
                    bands_data.append(band)
        red = bands_data[0]
        nir = bands_data[1]
        # NDVI calculation
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = np.clip(ndvi, -1, 1)

        # Calcular estatísticas apenas dentro do polígono original (geom)
        from rasterio.features import geometry_mask
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

        if not generate_image:
            return None, ndvi_mean, ndvi_min, ndvi_max

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
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = self._draw_smooth_polygon_on_image(pil_img, geom, image_crs, transform_affine, window, color="white", width=3)
        pil_img = self._prepare_image_for_report(pil_img)
        return self._pil_image_to_base64(pil_img), ndvi_mean, ndvi_min, ndvi_max
        
    def _download_crop_rgb_image(
        self,
        band_hrefs: dict,
        geom_bounds: tuple,
        geom: BaseGeometry,
        provider: StacProviderName,
    ) -> str:
        from PIL import ImageFilter

        bands_data = []
        transform_affine = None
        image_crs = None
        window = None
        upscale_factor = READ_RESAMPLE_FACTOR

        for band_idx, band_name in enumerate(["red", "green", "blue"]):
            band_asset_key = {"red": "B04", "green": "B03", "blue": "B02"}[band_name]
            href = band_hrefs[band_asset_key]
            href = self._sign_url(href, provider)

            with rasterio.Env(**RASTERIO_GDAL_CONFIG):
                with rasterio.open(href) as src:
                    if band_idx == 0:
                        image_crs = src.crs
                        transform_affine = src.transform
                        geom_bounds_proj = transform_bounds("EPSG:4326", src.crs, *geom_bounds)
                        window = from_bounds(*geom_bounds_proj, transform=src.transform)
                        window = window.round_offsets().round_lengths()
                        out_height = max(1, int(window.height * upscale_factor))
                        out_width = max(1, int(window.width * upscale_factor))

                    band = src.read(
                        1,
                        window=window,
                        out_shape=(out_height, out_width),
                        resampling=Resampling.lanczos if upscale_factor != 1 else Resampling.nearest,
                    )
                    # Conversão direta para uint8 usando divisor fixo (ex: 3000)
                    band = np.clip(band, 0, 3000)
                    band = (band / 3000 * 255).astype(np.uint8)
                    bands_data.append(band)

        image_rgb = np.stack(bands_data, axis=-1)
        pil_img = Image.fromarray(image_rgb)

        # 3. (Opcional: Sharpening pode ser mantido ou removido, aqui mantido para leve nitidez)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = self._draw_smooth_polygon_on_image(pil_img, geom, image_crs, transform_affine, window, color="white", width=3)
        pil_img = self._prepare_image_for_report(pil_img)
        return self._pil_image_to_base64(pil_img)

    def _extract_boundary_lines(self, geom_proj):
        from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

        if isinstance(geom_proj, Polygon):
            return [LineString(geom_proj.exterior.coords)]
        if isinstance(geom_proj, MultiPolygon):
            return [LineString(poly.exterior.coords) for poly in geom_proj.geoms]
        if isinstance(geom_proj, LineString):
            return [geom_proj]
        if isinstance(geom_proj, MultiLineString):
            return list(geom_proj.geoms)

        boundary = geom_proj.boundary
        if isinstance(boundary, LineString):
            return [boundary]
        if isinstance(boundary, MultiLineString):
            return list(boundary.geoms)

        raise ValueError(f"Unsupported geometry type for drawing: {geom_proj.geom_type}")

    def _draw_smooth_polygon_on_image(self, pil_img, geom, image_crs, transform_affine, window, color="white", width=5, interp_points=200):
        """
        Desenha um polígono suavizado (interpolado) sobre a imagem PIL.
        interp_points: número de pontos interpolados para suavizar a linha.
        Considera o upscale da imagem para desenhar o polígono no local correto.
        """
        draw = ImageDraw.Draw(pil_img)
        # Transforma geom para o CRS da imagem
        project = pyproj.Transformer.from_crs("EPSG:4326", image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)

        def world_to_pixel(x, y, transform_affine, window):
            col, row = ~transform_affine * (x, y)
            return (col - window.col_off, row - window.row_off)

        boundary_lines = self._extract_boundary_lines(geom_proj)

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

        for line in boundary_lines:
            coords = list(line.coords)
            line_interp_points = interp_points
            if len(coords) < line_interp_points:
                line_interp_points = max(len(coords) * 3, 50)
            interp_line = [
                line.interpolate(float(i) / line_interp_points, normalized=True).coords[0]
                for i in range(line_interp_points)
            ]
            pixel_coords = []
            for coord in interp_line:
                x, y = coord[:2]
                px, py = world_to_pixel(x, y, transform_affine, window)
                px *= scale_x
                py *= scale_y
                pixel_coords.append((px, py))
            # Fecha o polígono
            pixel_coords.append(pixel_coords[0])
            draw.line(pixel_coords, fill=color, width=width, joint="curve")

        return pil_img

    def _pil_image_to_base64(self, pil_img: Image.Image) -> str:
        """
        Converte uma imagem PIL para base64.
        """
        import base64
        from io import BytesIO

        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=REPORT_JPEG_QUALITY, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def _get_rgb_assets(self, item: pystac.Item) -> dict:
        return {
            "B04": resolve_band_href(item, "B04"),
            "B03": resolve_band_href(item, "B03"),
            "B02": resolve_band_href(item, "B02"),
        }

    def _get_ndvi_assets(self, item: pystac.Item) -> dict:
        return {
            "B04": resolve_band_href(item, "B04"),
            "B08": resolve_band_href(item, "B08"),
        }

    def _get_ndmi_assets(self, item: pystac.Item) -> dict:
        return {
            "B08": resolve_band_href(item, "B08"),
            "B11": resolve_band_href(item, "B11"),
        }

    def map_geom(self, geometry):
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
        return geom,geojson_geom,geom_bounds
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