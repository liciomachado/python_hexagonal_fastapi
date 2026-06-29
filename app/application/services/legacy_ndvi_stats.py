"""Cálculo de estatísticas NDVI compatível com a API legada (crop-sentinel-api)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio import warp
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from app.application.services.raster_helpers import (
    QA_PIXEL_CLOUD_MASK_BITS,
    build_rasterio_gdal_config,
)
from app.application.services.sensor_profile import SensorProfile
from app.application.services.stac.stac_types import resolve_band_href

SCL_VALID_CLASSES = (4, 5, 6)
NDVI_DECIMALS = 3
ZERO_DIVISOR_FIX = np.iinfo(np.uint16).max * 2
_GDAL_CONFIG = build_rasterio_gdal_config()


def apply_filters(index: np.ndarray) -> np.ndarray:
    index = index.copy()
    index[index > 1] = 1.0
    index[index < -1] = -1.0
    index[index == 0] = np.nan
    return index


def calc_ndvi(b_nir: np.ndarray, b_red: np.ndarray) -> np.ndarray | list:
    if b_nir.size == 0 or b_red.size == 0:
        return []

    b_nir = b_nir.astype(float)
    b_red = b_red.astype(float)

    denominator = b_nir + b_red
    denominator[denominator == 0] = ZERO_DIVISOR_FIX

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denominator != 0, (b_nir - b_red) / denominator, 0)

    return apply_filters(ndvi)


def resize_nearest(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if array.shape == target_shape:
        return array

    src_height, src_width = array.shape
    dst_height, dst_width = target_shape
    row_idx = np.clip(
        (np.arange(dst_height) * src_height / dst_height).astype(int),
        0,
        src_height - 1,
    )
    col_idx = np.clip(
        (np.arange(dst_width) * src_width / dst_width).astype(int),
        0,
        src_width - 1,
    )
    return array[row_idx][:, col_idx]


def build_valid_pixel_mask(
    mask_band: np.ndarray,
    target_shape: tuple[int, int],
    profile: SensorProfile,
) -> np.ndarray:
    if mask_band.size == 0:
        return np.zeros(target_shape, dtype=bool)

    resized = resize_nearest(mask_band.astype(np.float32), target_shape)

    if profile.cloud_mask_band == "SCL":
        return np.isin(resized, SCL_VALID_CLASSES)

    cloud_bits = resized.astype(np.uint32) & QA_PIXEL_CLOUD_MASK_BITS
    return cloud_bits == 0


def aggregate_ndvi_stats(
    ndvi_pixels: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    rounded = np.round(ndvi_pixels, NDVI_DECIMALS)
    valid = rounded[
        ~np.isnan(rounded) & ~np.isinf(rounded) & valid_mask
    ]

    if valid.size == 0:
        return None, None, None

    return float(np.mean(valid)), float(np.min(valid)), float(np.max(valid))


def read_polygon_masked_band(href: str, geom: BaseGeometry) -> np.ndarray | None:
    with rasterio.Env(**_GDAL_CONFIG):
        with rasterio.open(href) as dataset:
            warped_geometry = warp.transform_geom("EPSG:4326", dataset.crs, mapping(geom))
            try:
                masked, _ = mask(dataset, [warped_geometry], crop=True, nodata=0)
            except ValueError:
                return None

            if masked.size == 0:
                return None

            return masked[0]


def compute_ndvi_stats_from_bands(
    band_nir: np.ndarray,
    band_red: np.ndarray,
    band_mask: np.ndarray,
    profile: SensorProfile,
) -> tuple[float | None, float | None, float | None]:
    if band_nir.size == 0 or band_red.size == 0 or band_mask.size == 0:
        return None, None, None

    ndvi_pixels = calc_ndvi(band_nir, band_red)
    if isinstance(ndvi_pixels, list) or ndvi_pixels.size == 0:
        return None, None, None

    valid_mask = build_valid_pixel_mask(band_mask, ndvi_pixels.shape, profile)
    return aggregate_ndvi_stats(ndvi_pixels, valid_mask)


def compute_legacy_ndvi_stats(
    item,
    geom: BaseGeometry,
    profile: SensorProfile,
    sign_url: Callable[[str], str],
) -> tuple[float | None, float | None, float | None]:
    nir_key, red_key = profile.ndvi_bands
    mask_key = profile.cloud_mask_band

    nir_href = sign_url(resolve_band_href(item, nir_key))
    red_href = sign_url(resolve_band_href(item, red_key))
    mask_href = sign_url(resolve_band_href(item, mask_key))

    with ThreadPoolExecutor(max_workers=3) as executor:
        nir_future = executor.submit(read_polygon_masked_band, nir_href, geom)
        red_future = executor.submit(read_polygon_masked_band, red_href, geom)
        mask_future = executor.submit(read_polygon_masked_band, mask_href, geom)
        band_nir = nir_future.result()
        band_red = red_future.result()
        band_mask = mask_future.result()

    if band_nir is None or band_red is None or band_mask is None:
        return None, None, None

    return compute_ndvi_stats_from_bands(band_nir, band_red, band_mask, profile)
