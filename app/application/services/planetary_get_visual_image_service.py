from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import pyproj
import pystac
import rasterio
from PIL import Image, ImageDraw, ImageFilter
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from shapely import wkt
from shapely.geometry import mapping, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from app.application.services.dtos.planetary_all_images_response import PlanetaryAllImagesResponse
from app.application.services.dtos.planetary_ndvi_image_response import PlanetaryNdviImageResponse
from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse
from app.application.services.legacy_ndvi_stats import calc_ndvi, compute_legacy_ndvi_stats
from app.application.services.geometry_bounds import compute_cloud_cover_geom_bounds
from app.application.services.geometry_cloud_cover_service import GeometryCloudCoverService
from app.application.services.raster_helpers import build_rasterio_gdal_config, compute_out_shape, window_from_bounds
from app.application.services.sensor_profile import SensorProfile, get_sensor_profile, normalize_band_values
from app.application.services.stac.preferred_provider import PreferredProvider
from app.application.services.stac.satellite_collection import DEFAULT_SATELLITE_COLLECTION, SatelliteCollection
from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.application.services.stac.stac_types import StacProviderName, resolve_band_href
from app.core.cache_key import build_cache_key
from app.core.config import Config
from app.core.performance import PerformanceMetrics
from app.core.utils.result import AppError, BadRequestError, NotFoundError, Result
from app.domain.ports.blob_storage_port import BlobStoragePort
from app.domain.ports.cache_port import CachePort

if TYPE_CHECKING:
    from app.application.services.planetary_get_options_by_range import (
        PlanetaryGetOptionImagesByRangeServicePort,
        RangeDayCandidate,
    )

logger = logging.getLogger("app.image")

REPORT_MAX_IMAGE_DIMENSION = 1200
REPORT_JPEG_QUALITY = 85
NDVI_STATS_ALGORITHM_VERSION = "legacy-v1"


@dataclass
class RasterContext:
    window: object
    crop_transform: object
    image_crs: object
    transform_affine: object
    out_height: int
    out_width: int


class PlanetaryVisualImageServicePort(ABC):
    @abstractmethod
    async def get_ndmi_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        pass

    @abstractmethod
    async def get_visual_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
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
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        pass

    @abstractmethod
    async def get_all_images_by_day(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[PlanetaryAllImagesResponse, AppError]:
        pass

    @abstractmethod
    async def get_ndvi_by_range(
        self,
        dt_start: datetime,
        dt_end: datetime,
        geometry: str,
        cloud_percentual: float,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[list[PlanetaryNdviImageResponse], AppError]:
        pass


class PlanetaryVisualImageService(PlanetaryVisualImageServicePort):
    def __init__(
        self,
        stac_facade: StacResilientFacade,
        cache_service: CachePort | None = None,
        blob_storage: BlobStoragePort | None = None,
        cloud_cover_service: GeometryCloudCoverService | None = None,
        range_images_service: "PlanetaryGetOptionImagesByRangeServicePort | None" = None,
    ):
        self._stac_facade = stac_facade
        self._cache = cache_service
        self._blob_storage = blob_storage
        self._cloud_cover_service = cloud_cover_service or GeometryCloudCoverService(stac_facade)
        self._range_images_service = range_images_service
        self._gdal_config = build_rasterio_gdal_config()
        self._interp_points = Config.IMAGE_POLYGON_INTERP_POINTS
        self._border_width = Config.IMAGE_POLYGON_BORDER_WIDTH
        self._bounds_margin_ratio = Config.IMAGE_BOUNDS_MARGIN_RATIO
        self._bounds_min_span = Config.IMAGE_BOUNDS_MIN_SPAN
        self._enable_sharpen = Config.IMAGE_ENABLE_SHARPEN
        self._signed_url_cache: dict[tuple[str, str], str] = {}
        self._legacy_stats_cache: dict[str, tuple[float | None, float | None, float | None]] = {}

    def _begin_request_scoped_caches(self) -> None:
        """Limpa caches in-process usados apenas dentro de uma requisição."""
        self._signed_url_cache.clear()
        self._legacy_stats_cache.clear()

    async def get_ndmi_image(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        profile = get_sensor_profile(satellite_collection)
        metrics = PerformanceMetrics(context="ndmi")
        cache_key = build_cache_key(
            "ndmi",
            day=day,
            geometry=geometry,
            cloud_percentual=cloud_percentual,
            generate_image=generate_image,
            preferred_provider=preferred_provider,
            satellite_collection=satellite_collection,
        )
        cached = await self._try_get_cached_ndvi(cache_key)
        if cached is not None:
            return Result.Ok(cached)

        self._begin_request_scoped_caches()
        try:
            with metrics.span("prepare_context"):
                selected, provider, geom, geom_bounds, geometry_cloud_percentual = await self._prepare_context(
                    day, cloud_percentual, geometry, preferred_provider, metrics, satellite_collection
                )
            if selected is None:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))

            with metrics.span("bands_read"):
                jpeg_bytes, ndmi_mean, ndmi_min, ndmi_max = await asyncio.to_thread(
                    self._process_ndmi_from_item,
                    selected,
                    geom_bounds,
                    geom,
                    generate_image,
                    provider,
                    profile,
                )

            image_url = None
            if generate_image and jpeg_bytes is not None:
                with metrics.span("blob_upload"):
                    image_url = await self._upload_jpeg(
                        jpeg_bytes,
                        f"{profile.blob_prefix}/{selected.id}/ndmi/{uuid.uuid4().hex}.jpg",
                    )

            response = PlanetaryNdviImageResponse(
                day=day,
                cloud_percentual=geometry_cloud_percentual,
                image_url=image_url,
                ndvi_mean=ndmi_mean,
                ndvi_min=ndmi_min,
                ndvi_max=ndmi_max,
                sat_image_id=selected.id,
            )
            await self._set_cached_ndvi(cache_key, response)
            metrics.log_summary()
            return Result.Ok(response)
        except AppError as ex:
            return Result.Err(ex)
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
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[PlanetaryImageVisualResponse, AppError]:
        profile = get_sensor_profile(satellite_collection)
        metrics = PerformanceMetrics(context="visual")
        cache_key = build_cache_key(
            "visual",
            day=day,
            geometry=geometry,
            cloud_percentual=cloud_percentual,
            preferred_provider=preferred_provider,
            satellite_collection=satellite_collection,
        )
        cached = await self._try_get_cached_visual(cache_key)
        if cached is not None:
            return Result.Ok(cached)

        self._begin_request_scoped_caches()
        try:
            with metrics.span("prepare_context"):
                selected, provider, geom, geom_bounds, geometry_cloud_percentual = await self._prepare_context(
                    day, cloud_percentual, geometry, preferred_provider, metrics, satellite_collection
                )
            if selected is None:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))

            with metrics.span("bands_read"):
                jpeg_bytes = await asyncio.to_thread(
                    self._process_rgb_from_item,
                    selected,
                    geom_bounds,
                    geom,
                    provider,
                    profile,
                )

            with metrics.span("blob_upload"):
                image_url = await self._upload_jpeg(
                    jpeg_bytes,
                    f"{profile.blob_prefix}/{selected.id}/visual/{uuid.uuid4().hex}.jpg",
                )

            response = PlanetaryImageVisualResponse(
                day=day,
                cloud_percentual=geometry_cloud_percentual,
                image_url=image_url,
            )
            await self._set_cached_visual(cache_key, response)
            metrics.log_summary()
            return Result.Ok(response)
        except AppError as ex:
            return Result.Err(ex)
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
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[PlanetaryNdviImageResponse, AppError]:
        profile = get_sensor_profile(satellite_collection)
        metrics = PerformanceMetrics(context="ndvi")
        cache_key = build_cache_key(
            "ndvi",
            day=day,
            geometry=geometry,
            cloud_percentual=cloud_percentual,
            generate_image=generate_image,
            preferred_provider=preferred_provider,
            satellite_collection=satellite_collection,
            ndvi_stats_version=NDVI_STATS_ALGORITHM_VERSION,
        )
        cached = await self._try_get_cached_ndvi(cache_key)
        if cached is not None:
            return Result.Ok(cached)

        self._begin_request_scoped_caches()
        try:
            with metrics.span("prepare_context"):
                selected, provider, geom, geom_bounds, geometry_cloud_percentual = await self._prepare_context(
                    day, cloud_percentual, geometry, preferred_provider, metrics, satellite_collection
                )
            if selected is None:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))

            with metrics.span("bands_read"):
                jpeg_bytes, ndvi_mean, ndvi_min, ndvi_max = await asyncio.to_thread(
                    self._process_ndvi_from_item,
                    selected,
                    geom_bounds,
                    geom,
                    generate_image,
                    provider,
                    profile,
                )

            image_url = None
            if generate_image and jpeg_bytes is not None:
                with metrics.span("blob_upload"):
                    image_url = await self._upload_jpeg(
                        jpeg_bytes,
                        f"{profile.blob_prefix}/{selected.id}/ndvi/{uuid.uuid4().hex}.jpg",
                    )

            response = PlanetaryNdviImageResponse(
                day=day,
                cloud_percentual=geometry_cloud_percentual,
                image_url=image_url,
                ndvi_mean=ndvi_mean,
                ndvi_min=ndvi_min,
                ndvi_max=ndvi_max,
                sat_image_id=selected.id,
            )
            await self._set_cached_ndvi(cache_key, response)
            metrics.log_summary()
            return Result.Ok(response)
        except AppError as ex:
            return Result.Err(ex)
        except ValueError as ex:
            return Result.Err(str(ex))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagem NDVI: {str(ex)}")

    async def get_all_images_by_day(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[PlanetaryAllImagesResponse, AppError]:
        profile = get_sensor_profile(satellite_collection)
        metrics = PerformanceMetrics(context="all")
        cache_key = build_cache_key(
            "all",
            day=day,
            geometry=geometry,
            cloud_percentual=cloud_percentual,
            generate_image=generate_image,
            preferred_provider=preferred_provider,
            satellite_collection=satellite_collection,
            ndvi_stats_version=NDVI_STATS_ALGORITHM_VERSION,
        )
        cached = await self._try_get_cached_all(cache_key)
        if cached is not None:
            return Result.Ok(cached)

        self._begin_request_scoped_caches()
        try:
            with metrics.span("prepare_context"):
                selected, provider, geom, geom_bounds, geometry_cloud_percentual = await self._prepare_context(
                    day, cloud_percentual, geometry, preferred_provider, metrics, satellite_collection
                )
            if selected is None:
                return Result.Err(BadRequestError(f"Nenhuma imagem cobre ao menos {cloud_percentual}% da geometria."))

            with metrics.span("bands_read"):
                pipeline_result = await asyncio.to_thread(
                    self._process_all_from_item,
                    selected,
                    geom_bounds,
                    geom,
                    generate_image,
                    provider,
                    profile,
                )

            visual_jpeg, ndvi_jpeg, ndmi_jpeg, ndvi_stats, ndmi_stats = pipeline_result

            with metrics.span("blob_upload"):
                visual_url = await self._upload_jpeg(
                    visual_jpeg,
                    f"{profile.blob_prefix}/{selected.id}/visual/{uuid.uuid4().hex}.jpg",
                )
                ndvi_url = None
                ndmi_url = None
                if generate_image:
                    if ndvi_jpeg is not None:
                        ndvi_url = await self._upload_jpeg(
                            ndvi_jpeg,
                            f"{profile.blob_prefix}/{selected.id}/ndvi/{uuid.uuid4().hex}.jpg",
                        )
                    if ndmi_jpeg is not None:
                        ndmi_url = await self._upload_jpeg(
                            ndmi_jpeg,
                            f"{profile.blob_prefix}/{selected.id}/ndmi/{uuid.uuid4().hex}.jpg",
                        )

            response = PlanetaryAllImagesResponse(
                visual=PlanetaryImageVisualResponse(
                    day=day,
                    cloud_percentual=geometry_cloud_percentual,
                    image_url=visual_url,
                ),
                ndvi=PlanetaryNdviImageResponse(
                    day=day,
                    cloud_percentual=geometry_cloud_percentual,
                    image_url=ndvi_url,
                    ndvi_mean=ndvi_stats[0],
                    ndvi_min=ndvi_stats[1],
                    ndvi_max=ndvi_stats[2],
                    sat_image_id=selected.id,
                ),
                ndmi=PlanetaryNdviImageResponse(
                    day=day,
                    cloud_percentual=geometry_cloud_percentual,
                    image_url=ndmi_url,
                    ndvi_mean=ndmi_stats[0],
                    ndvi_min=ndmi_stats[1],
                    ndvi_max=ndmi_stats[2],
                    sat_image_id=selected.id,
                ),
            )
            await self._set_cached_all(cache_key, response)
            metrics.log_summary()
            return Result.Ok(response)
        except AppError as ex:
            return Result.Err(ex)
        except ValueError as ex:
            return Result.Err(str(ex))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar imagens: {str(ex)}")

    async def get_ndvi_by_range(
        self,
        dt_start: datetime,
        dt_end: datetime,
        geometry: str,
        cloud_percentual: float,
        generate_image: bool,
        preferred_provider: PreferredProvider | None = None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> Result[list[PlanetaryNdviImageResponse], AppError]:
        if self._range_images_service is None:
            return Result.Err(BadRequestError("Serviço de busca por range não configurado."))

        profile = get_sensor_profile(satellite_collection)
        metrics = PerformanceMetrics(context="ndvi_by_range")

        self._begin_request_scoped_caches()
        try:
            with metrics.span("stac_range_search"):
                candidates, provider, _ = await self._range_images_service.search_best_items_by_day(
                    geometry=geometry,
                    start_date=dt_start,
                    end_date=dt_end,
                    preferred_provider=preferred_provider,
                    satellite_collection=satellite_collection,
                )

            eligible = [
                c
                for c in candidates
                if c.cloud_cover_geometry is not None and c.cloud_cover_geometry <= cloud_percentual
            ]
            if not eligible:
                return Result.Err(NotFoundError("No images found"))

            geom, _, geom_bounds = self.map_geom(geometry)
            semaphore = asyncio.Semaphore(Config.SCL_CONCURRENT_READS)

            async def process_day(candidate: "RangeDayCandidate") -> PlanetaryNdviImageResponse:
                async with semaphore:
                    day = candidate.datetime.date()
                    cache_key = self._build_ndvi_range_day_cache_key(
                        day=day,
                        geometry=geometry,
                        sat_image_id=candidate.id,
                        generate_image=generate_image,
                        preferred_provider=preferred_provider,
                        satellite_collection=satellite_collection,
                    )
                    cached = await self._try_get_cached_ndvi(cache_key)
                    if cached is not None:
                        return cached

                    jpeg_bytes, ndvi_mean, ndvi_min, ndvi_max = await asyncio.to_thread(
                        self._process_ndvi_from_item,
                        candidate.stac_item,
                        geom_bounds,
                        geom,
                        generate_image,
                        provider,
                        profile,
                    )
                    image_url = None
                    if generate_image and jpeg_bytes is not None:
                        image_url = await self._upload_jpeg(
                            jpeg_bytes,
                            f"{profile.blob_prefix}/{candidate.id}/ndvi/{uuid.uuid4().hex}.jpg",
                        )
                    response = PlanetaryNdviImageResponse(
                        day=day,
                        cloud_percentual=candidate.cloud_cover_geometry,
                        image_url=image_url,
                        ndvi_mean=ndvi_mean,
                        ndvi_min=ndvi_min,
                        ndvi_max=ndvi_max,
                        sat_image_id=candidate.id,
                    )
                    await self._set_cached_ndvi(cache_key, response)
                    return response

            with metrics.span("bands_read"):
                responses = await asyncio.gather(*(process_day(c) for c in eligible))

            ordered = sorted(responses, key=lambda item: item.day)
            metrics.log_summary()
            return Result.Ok(ordered)
        except AppError as ex:
            return Result.Err(ex)
        except ValueError as ex:
            return Result.Err(str(ex))
        except Exception as ex:
            return Result.Err(f"Erro inesperado ao buscar NDVI por range: {str(ex)}")

    async def _prepare_context(
        self,
        day: date,
        cloud_percentual: float,
        geometry: str,
        preferred_provider: PreferredProvider | None,
        metrics: PerformanceMetrics,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> tuple[pystac.Item | None, StacProviderName, BaseGeometry, tuple, float]:
        geom, geojson_geom, geom_bounds = self.map_geom(geometry)
        cloud_cover_geom_bounds = compute_cloud_cover_geom_bounds(geom)
        with metrics.span("stac_search"):
            selected, provider = await self._search_selected_item(
                day=day,
                cloud_percentual=cloud_percentual,
                geom=geom,
                geojson_geom=geojson_geom,
                preferred_provider=preferred_provider,
                satellite_collection=satellite_collection,
            )
        if selected is None:
            return None, provider, geom, geom_bounds, 0.0

        with metrics.span("scl_read"):
            geometry_cloud_percentual = await asyncio.to_thread(
                self._cloud_cover_service.compute_cloud_percentual_over_geometry,
                selected,
                geom,
                cloud_cover_geom_bounds,
                provider,
                satellite_collection,
            )
        cloud_error = self._validate_geometry_cloud_percentual(
            geometry_cloud_percentual,
            cloud_percentual,
        )
        if cloud_error is not None:
            raise cloud_error

        return selected, provider, geom, geom_bounds, geometry_cloud_percentual

    async def _search_selected_item(
        self,
        day: date,
        cloud_percentual: float,
        geom: BaseGeometry,
        geojson_geom: dict,
        preferred_provider: PreferredProvider | None,
        satellite_collection: SatelliteCollection = DEFAULT_SATELLITE_COLLECTION,
    ) -> tuple[pystac.Item | None, StacProviderName]:
        provider_enum = self._parse_preferred_provider(preferred_provider)
        search_result = await self._stac_facade.search_items_by_day(
            geojson_geom=geojson_geom,
            day=day,
            max_items=10,
            preferred_provider=provider_enum,
            collection=satellite_collection,
        )
        items = search_result.items
        if not items:
            raise BadRequestError("Nenhuma imagem encontrada para a data e geometria fornecidas.")
        items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))
        for item in items:
            if item.geometry is None:
                continue
            image_geom = shape(item.geometry)
            if geom.intersection(image_geom).area / geom.area >= cloud_percentual / 100.0:
                return item, search_result.provider
        return None, search_result.provider

    def _compute_legacy_ndvi_stats(
        self,
        item: pystac.Item,
        geom: BaseGeometry,
        provider: StacProviderName,
        profile: SensorProfile,
    ) -> tuple[float | None, float | None, float | None]:
        cache_key = build_cache_key(
            "legacy_ndvi_stats",
            sat_image_id=item.id,
            geometry=geom.wkt,
            satellite_collection=profile.collection.value,
            ndvi_stats_version=NDVI_STATS_ALGORITHM_VERSION,
        )
        cached = self._legacy_stats_cache.get(cache_key)
        if cached is not None:
            return cached

        stats = compute_legacy_ndvi_stats(
            item,
            geom,
            profile,
            sign_url=lambda href: self._sign_url(href, provider),
        )
        self._legacy_stats_cache[cache_key] = stats
        return stats

    def _validate_geometry_cloud_percentual(
        self,
        computed_cloud_percentual: float,
        max_allowed_cloud_percentual: float,
    ) -> BadRequestError | None:
        if computed_cloud_percentual > max_allowed_cloud_percentual:
            return BadRequestError(
                f"Cobertura de nuvens sobre a geometria ({computed_cloud_percentual}%) "
                f"excede o limite permitido ({max_allowed_cloud_percentual}%)."
            )
        return None

    def _read_shared_bands(
        self,
        item: pystac.Item,
        geom_bounds: tuple,
        provider: StacProviderName,
        band_keys: tuple[str, ...],
        profile: SensorProfile,
        reference_band_key: str = "B04",
    ) -> tuple[dict[str, np.ndarray], RasterContext]:
        reference_href = self._sign_url(resolve_band_href(item, reference_band_key), provider)
        resampling = Resampling.bilinear
        bands: dict[str, np.ndarray] = {}
        with rasterio.Env(**self._gdal_config):
            with rasterio.open(reference_href) as ref_src:
                image_crs = ref_src.crs
                transform_affine = ref_src.transform
                geom_bounds_proj = transform_bounds("EPSG:4326", ref_src.crs, *geom_bounds)
                window = window_from_bounds(
                    geom_bounds_proj,
                    ref_src.transform,
                    ref_src.width,
                    ref_src.height,
                )
                out_height, out_width = compute_out_shape(window, REPORT_MAX_IMAGE_DIMENSION)
                crop_transform = ref_src.window_transform(window)
                scale_x = window.width / out_width
                scale_y = window.height / out_height
                crop_transform = crop_transform * crop_transform.scale(scale_x, scale_y)
                out_shape = (out_height, out_width)

                if reference_band_key in band_keys:
                    band_window = window_from_bounds(
                        geom_bounds_proj,
                        ref_src.transform,
                        ref_src.width,
                        ref_src.height,
                    )
                    raw = ref_src.read(
                        1,
                        window=band_window,
                        out_shape=out_shape,
                        resampling=resampling,
                    ).astype(np.float32)
                    bands[reference_band_key] = normalize_band_values(raw, profile)

        ctx = RasterContext(
            window=window,
            crop_transform=crop_transform,
            image_crs=image_crs,
            transform_affine=transform_affine,
            out_height=out_height,
            out_width=out_width,
        )

        def _read_band(band_key: str) -> tuple[str, np.ndarray]:
            href = self._sign_url(resolve_band_href(item, band_key), provider)
            with rasterio.Env(**self._gdal_config):
                with rasterio.open(href) as src:
                    band_window = window_from_bounds(
                        geom_bounds_proj,
                        src.transform,
                        src.width,
                        src.height,
                    )
                    raw = src.read(
                        1,
                        window=band_window,
                        out_shape=out_shape,
                        resampling=resampling,
                    ).astype(np.float32)
                    return band_key, normalize_band_values(raw, profile)

        remaining_keys = [key for key in band_keys if key not in bands]

        if remaining_keys:
            max_workers = min(len(remaining_keys), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for band_key, values in executor.map(_read_band, remaining_keys):
                    bands[band_key] = values

        return bands, ctx

    def _process_rgb_from_item(
        self,
        item: pystac.Item,
        geom_bounds: tuple,
        geom: BaseGeometry,
        provider: StacProviderName,
        profile: SensorProfile,
    ) -> bytes:
        red_key, green_key, blue_key = profile.rgb_bands
        bands, ctx = self._read_shared_bands(
            item, geom_bounds, provider, (blue_key, green_key, red_key), profile, reference_band_key=red_key
        )
        clip_max = profile.rgb_clip_max
        b02 = np.clip(bands[blue_key], 0, clip_max)
        b03 = np.clip(bands[green_key], 0, clip_max)
        b04 = np.clip(bands[red_key], 0, clip_max)
        image_rgb = np.stack(
            [
                (b04 / clip_max * 255).astype(np.uint8),
                (b03 / clip_max * 255).astype(np.uint8),
                (b02 / clip_max * 255).astype(np.uint8),
            ],
            axis=-1,
        )
        pil_img = Image.fromarray(image_rgb)
        return self._finalize_image_bytes(pil_img, geom, ctx)

    def _process_ndvi_from_item(
        self,
        item: pystac.Item,
        geom_bounds: tuple,
        geom: BaseGeometry,
        generate_image: bool,
        provider: StacProviderName,
        profile: SensorProfile,
    ) -> tuple[bytes | None, float | None, float | None, float | None]:
        ndvi_mean, ndvi_min, ndvi_max = self._compute_legacy_ndvi_stats(
            item, geom, provider, profile
        )

        if not generate_image:
            return None, ndvi_mean, ndvi_min, ndvi_max

        nir_key, red_key = profile.ndvi_bands
        bands, ctx = self._read_shared_bands(item, geom_bounds, provider, (red_key, nir_key), profile)
        jpeg_bytes, _, _, _ = self._build_index_product(
            bands[nir_key],
            bands[red_key],
            geom,
            ctx,
            True,
            NDVI_BANDWIDTH_COLORS_VALUES,
            BANDWIDTH_COLORS_NDVI,
            is_ndvi=True,
        )
        return jpeg_bytes, ndvi_mean, ndvi_min, ndvi_max

    def _process_ndmi_from_item(
        self,
        item: pystac.Item,
        geom_bounds: tuple,
        geom: BaseGeometry,
        generate_image: bool,
        provider: StacProviderName,
        profile: SensorProfile,
    ) -> tuple[bytes | None, float | None, float | None, float | None]:
        nir_key, swir_key = profile.ndmi_bands
        bands, ctx = self._read_shared_bands(
            item,
            geom_bounds,
            provider,
            (nir_key, swir_key),
            profile,
            reference_band_key=profile.ndmi_reference_band,
        )
        return self._build_index_product(
            bands[nir_key],
            bands[swir_key],
            geom,
            ctx,
            generate_image,
            NDMI_BANDWIDTH_COLORS_VALUES,
            NDMI_BANDWIDTH_COLORS,
            is_ndvi=False,
        )

    def _process_all_from_item(
        self,
        item: pystac.Item,
        geom_bounds: tuple,
        geom: BaseGeometry,
        generate_image: bool,
        provider: StacProviderName,
        profile: SensorProfile,
    ) -> tuple[bytes, bytes | None, bytes | None, tuple, tuple]:
        red_key, green_key, blue_key = profile.rgb_bands
        nir_key, swir_key = profile.ndmi_bands
        bands, ctx = self._read_shared_bands(
            item, geom_bounds, provider, profile.all_bands, profile, reference_band_key=red_key
        )

        clip_max = profile.rgb_clip_max
        b02 = np.clip(bands[blue_key], 0, clip_max)
        b03 = np.clip(bands[green_key], 0, clip_max)
        b04 = np.clip(bands[red_key], 0, clip_max)
        image_rgb = np.stack(
            [
                (b04 / clip_max * 255).astype(np.uint8),
                (b03 / clip_max * 255).astype(np.uint8),
                (b02 / clip_max * 255).astype(np.uint8),
            ],
            axis=-1,
        )
        visual_jpeg = self._finalize_image_bytes(Image.fromarray(image_rgb), geom, ctx)

        ndvi_mean, ndvi_min, ndvi_max = self._compute_legacy_ndvi_stats(
            item, geom, provider, profile
        )
        ndvi_jpeg, _, _, _ = self._build_index_product(
            bands[nir_key],
            bands[red_key],
            geom,
            ctx,
            generate_image,
            NDVI_BANDWIDTH_COLORS_VALUES,
            BANDWIDTH_COLORS_NDVI,
            is_ndvi=True,
        )
        ndmi_jpeg, ndmi_mean, ndmi_min, ndmi_max = self._build_index_product(
            bands[nir_key],
            bands[swir_key],
            geom,
            ctx,
            generate_image,
            NDMI_BANDWIDTH_COLORS_VALUES,
            NDMI_BANDWIDTH_COLORS,
            is_ndvi=False,
        )

        return (
            visual_jpeg,
            ndvi_jpeg,
            ndmi_jpeg,
            (ndvi_mean, ndvi_min, ndvi_max),
            (ndmi_mean, ndmi_min, ndmi_max),
        )

    def _build_index_product(
        self,
        numerator_band: np.ndarray,
        denominator_band: np.ndarray,
        geom: BaseGeometry,
        ctx: RasterContext,
        generate_image: bool,
        color_values: list[float],
        color_palette: list[tuple[float, float, float]],
        is_ndvi: bool,
    ) -> tuple[bytes | None, float | None, float | None, float | None]:
        if is_ndvi:
            index = (numerator_band - denominator_band) / (numerator_band + denominator_band + 1e-6)
        else:
            index = (numerator_band - denominator_band) / (numerator_band + denominator_band + 1e-6)
        index = np.clip(index, -1, 1)

        project = pyproj.Transformer.from_crs("EPSG:4326", ctx.image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)
        mask = geometry_mask(
            [mapping(geom_proj)],
            out_shape=index.shape,
            transform=ctx.crop_transform,
            invert=True,
        )
        index_inside = index[mask]
        index_inside = index_inside[~np.isnan(index_inside) & ~np.isinf(index_inside)]
        if index_inside.size > 0:
            index_mean = float(np.mean(index_inside))
            index_min = float(np.min(index_inside))
            index_max = float(np.max(index_inside))
        else:
            index_mean = index_min = index_max = None

        if not generate_image:
            return None, index_mean, index_min, index_max

        rgb = self._apply_colormap(index, color_values, color_palette)
        pil_img = Image.fromarray(rgb, mode="RGB")
        jpeg_bytes = self._finalize_image_bytes(pil_img, geom, ctx)
        return jpeg_bytes, index_mean, index_min, index_max

    def _apply_colormap(
        self,
        index: np.ndarray,
        color_values: list[float],
        color_palette: list[tuple[float, float, float]],
    ) -> np.ndarray:
        rgb = np.zeros(index.shape + (3,), dtype=np.float32)
        flat_index = index.ravel()
        flat_rgb = rgb.reshape(-1, 3)
        bins = np.searchsorted(color_values, flat_index, side="right") - 1
        bins = np.clip(bins, 0, len(color_values) - 2)
        for i in range(len(color_values) - 1):
            mask = bins == i
            if not np.any(mask):
                continue
            vmin = color_values[i]
            vmax = color_values[i + 1]
            cmin = np.array(color_palette[i])
            cmax = np.array(color_palette[i + 1])
            alpha = (flat_index[mask] - vmin) / (vmax - vmin + 1e-8)
            flat_rgb[mask] = (1 - alpha)[:, None] * cmin + alpha[:, None] * cmax
        return (rgb * 255).astype(np.uint8)

    def _finalize_image_bytes(
        self,
        pil_img: Image.Image,
        geom: BaseGeometry,
        ctx: RasterContext,
    ) -> bytes:
        if self._enable_sharpen:
            pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = self._draw_smooth_polygon_on_image(
            pil_img,
            geom,
            ctx.image_crs,
            ctx.transform_affine,
            ctx.window,
            color="white",
            width=self._border_width,
            interp_points=self._interp_points,
        )
        pil_img = self._prepare_image_for_report(pil_img)
        return self._pil_image_to_jpeg_bytes(pil_img)

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
        cache_key = (provider.value, href)
        cached = self._signed_url_cache.get(cache_key)
        if cached is not None:
            return cached
        signed = self._stac_facade.sign_asset_url(href, provider)
        self._signed_url_cache[cache_key] = signed
        return signed

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

    def _draw_smooth_polygon_on_image(
        self,
        pil_img,
        geom,
        image_crs,
        transform_affine,
        window,
        color="white",
        width=5,
        interp_points=80,
    ):
        draw = ImageDraw.Draw(pil_img)
        project = pyproj.Transformer.from_crs("EPSG:4326", image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)

        def world_to_pixel(x, y, transform, win):
            col, row = ~transform * (x, y)
            return (col - win.col_off, row - win.row_off)

        boundary_lines = self._extract_boundary_lines(geom_proj)

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
            pixel_coords.append(pixel_coords[0])
            draw.line(pixel_coords, fill=color, width=width, joint="curve")

        return pil_img

    def _pil_image_to_jpeg_bytes(self, pil_img: Image.Image) -> bytes:
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=REPORT_JPEG_QUALITY, optimize=False)
        return buffered.getvalue()

    async def _upload_jpeg(self, jpeg_bytes: bytes | None, blob_name: str) -> str:
        if jpeg_bytes is None:
            raise BadRequestError("Imagem não gerada.")
        if self._blob_storage is None:
            raise BadRequestError(
                "Azure Blob Storage não configurado. Defina AZURE_BLOB_CONNECTION_STRING."
            )
        return await self._blob_storage.upload_image_and_get_url(jpeg_bytes, blob_name)

    def map_geom(self, geometry):
        geom = wkt.loads(geometry)
        bounds = geom.bounds
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny
        size = max(width, height, self._bounds_min_span)
        square_parameter = 2
        center_x = (minx + maxx) / square_parameter
        center_y = (miny + maxy) / square_parameter
        square_geom = box(
            center_x - size / square_parameter,
            center_y - size / square_parameter,
            center_x + size / square_parameter,
            center_y + size / square_parameter,
        )
        geojson_geom = mapping(square_geom)
        margin = size * self._bounds_margin_ratio
        half_extent = size / square_parameter + margin
        geom_bounds = (
            center_x - half_extent,
            center_y - half_extent,
            center_x + half_extent,
            center_y + half_extent,
        )
        return geom, geojson_geom, geom_bounds

    def _build_ndvi_range_day_cache_key(
        self,
        day: date,
        geometry: str,
        sat_image_id: str,
        generate_image: bool,
        preferred_provider: PreferredProvider | None,
        satellite_collection: SatelliteCollection,
    ) -> str:
        return build_cache_key(
            "ndvi_range_day",
            day=day,
            geometry=geometry,
            sat_image_id=sat_image_id,
            generate_image=generate_image,
            preferred_provider=preferred_provider,
            satellite_collection=satellite_collection,
            ndvi_stats_version=NDVI_STATS_ALGORITHM_VERSION,
        )

    async def _try_get_cached_visual(self, key: str) -> PlanetaryImageVisualResponse | None:
        if self._cache is None:
            return None
        raw = await self._cache.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        return PlanetaryImageVisualResponse(
            day=date.fromisoformat(data["day"]),
            cloud_percentual=data["cloud_percentual"],
            image_url=data["image_url"],
        )

    async def _set_cached_visual(self, key: str, response: PlanetaryImageVisualResponse) -> None:
        if self._cache is None:
            return
        payload = json.dumps(
            {
                "day": response.day.isoformat(),
                "cloud_percentual": response.cloud_percentual,
                "image_url": response.image_url,
            }
        )
        await self._cache.set(key, payload, ttl_seconds=Config.CACHE_TTL_SECONDS)

    async def _try_get_cached_ndvi(self, key: str) -> PlanetaryNdviImageResponse | None:
        if self._cache is None:
            return None
        raw = await self._cache.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        return PlanetaryNdviImageResponse(
            day=date.fromisoformat(data["day"]),
            cloud_percentual=data["cloud_percentual"],
            image_url=data.get("image_url"),
            ndvi_mean=data.get("ndvi_mean"),
            ndvi_min=data.get("ndvi_min"),
            ndvi_max=data.get("ndvi_max"),
            sat_image_id=data["sat_image_id"],
        )

    async def _set_cached_ndvi(self, key: str, response: PlanetaryNdviImageResponse) -> None:
        if self._cache is None:
            return
        payload = json.dumps(
            {
                "day": response.day.isoformat(),
                "cloud_percentual": response.cloud_percentual,
                "image_url": response.image_url,
                "ndvi_mean": response.ndvi_mean,
                "ndvi_min": response.ndvi_min,
                "ndvi_max": response.ndvi_max,
                "sat_image_id": response.sat_image_id,
            }
        )
        await self._cache.set(key, payload, ttl_seconds=Config.CACHE_TTL_SECONDS)

    async def _try_get_cached_all(self, key: str) -> PlanetaryAllImagesResponse | None:
        if self._cache is None:
            return None
        raw = await self._cache.get(key)
        if raw is None:
            return None
        data = json.loads(raw)

        def _parse_index_block(block: dict) -> PlanetaryNdviImageResponse:
            return PlanetaryNdviImageResponse(
                day=date.fromisoformat(block["day"]),
                cloud_percentual=block["cloud_percentual"],
                image_url=block.get("image_url"),
                ndvi_mean=block.get("ndvi_mean"),
                ndvi_min=block.get("ndvi_min"),
                ndvi_max=block.get("ndvi_max"),
                sat_image_id=block["sat_image_id"],
            )

        visual = data["visual"]
        return PlanetaryAllImagesResponse(
            visual=PlanetaryImageVisualResponse(
                day=date.fromisoformat(visual["day"]),
                cloud_percentual=visual["cloud_percentual"],
                image_url=visual["image_url"],
            ),
            ndvi=_parse_index_block(data["ndvi"]),
            ndmi=_parse_index_block(data["ndmi"]),
        )

    async def _set_cached_all(self, key: str, response: PlanetaryAllImagesResponse) -> None:
        if self._cache is None:
            return

        def _serialize_index_block(item: PlanetaryNdviImageResponse) -> dict:
            return {
                "day": item.day.isoformat(),
                "cloud_percentual": item.cloud_percentual,
                "image_url": item.image_url,
                "ndvi_mean": item.ndvi_mean,
                "ndvi_min": item.ndvi_min,
                "ndvi_max": item.ndvi_max,
                "sat_image_id": item.sat_image_id,
            }

        payload = json.dumps(
            {
                "visual": {
                    "day": response.visual.day.isoformat(),
                    "cloud_percentual": response.visual.cloud_percentual,
                    "image_url": response.visual.image_url,
                },
                "ndvi": _serialize_index_block(response.ndvi),
                "ndmi": _serialize_index_block(response.ndmi),
            }
        )
        await self._cache.set(key, payload, ttl_seconds=Config.CACHE_TTL_SECONDS)


NDVI_BANDWIDTH_COLORS_VALUES = [
    -1.0,
    -0.506082,
    -0.180048,
    0.10949,
    0.309002,
    0.416058,
    0.554744,
    0.73236,
    1.0,
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
    (60 / 255, 29 / 255, 18 / 255),
    (109 / 255, 64 / 255, 44 / 255),
    (149 / 255, 87 / 255, 61 / 255),
    (207 / 255, 135 / 255, 104 / 255),
    (218 / 255, 229 / 255, 237 / 255),
    (94 / 255, 174 / 255, 240 / 255),
    (79 / 255, 150 / 255, 235 / 255),
    (52 / 255, 113 / 255, 214 / 255),
    (16 / 255, 69 / 255, 185 / 255),
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
    1.0,
]


def apply_filters(index: np.ndarray) -> np.ndarray:
    index[index > 1] = 1.0
    index[index < -1] = -1.0
    index[index == 0] = np.nan
    return index


def calc_ndmi(b_nir: np.ndarray, b_swir: np.ndarray) -> np.ndarray | list:
    if len(b_nir) == 0 or len(b_swir) == 0:
        return []

    b_nir = b_nir.astype(float)
    b_swir = b_swir.astype(float)

    denominator = b_nir + b_swir
    denominator[denominator == 0] = ZERO_DIVISOR_FIX

    with np.errstate(divide="ignore", invalid="ignore"):
        ndmi = np.where(denominator != 0, (b_nir - b_swir) / denominator, 0)

    return apply_filters(ndmi)
