import rasterio
from rasterio.windows import Window, from_bounds

from app.core.config import Config

SCL_CLOUD_CLASSES = frozenset({3, 8, 9, 10})

# Landsat Collection 2 QA_PIXEL bit flags (USGS LSDS-1619)
QA_PIXEL_CLOUD = 1 << 3
QA_PIXEL_CLOUD_SHADOW = 1 << 4
QA_PIXEL_CLOUD_MASK_BITS = QA_PIXEL_CLOUD | QA_PIXEL_CLOUD_SHADOW


def build_rasterio_gdal_config() -> dict[str, str]:
    return {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": Config.VSI_CACHE_SIZE,
    }


def compute_out_shape(window, max_dimension: int) -> tuple[int, int]:
    out_height = max(1, int(window.height))
    out_width = max(1, int(window.width))
    max_dim = max(out_height, out_width)
    if max_dim <= max_dimension:
        return out_height, out_width
    scale = max_dimension / max_dim
    return max(1, int(out_height * scale)), max(1, int(out_width * scale))


def window_from_bounds(
    bounds_proj: tuple[float, float, float, float],
    transform,
    raster_width: int,
    raster_height: int,
):
    window = from_bounds(*bounds_proj, transform=transform)
    window = window.round_offsets(op="floor").round_lengths(op="ceil")
    raster_window = Window(0, 0, raster_width, raster_height)
    try:
        window = window.intersection(raster_window)
    except rasterio.errors.WindowError as exc:
        raise ValueError("Geometria não intersecta a imagem selecionada.") from exc
    if window.width <= 0 or window.height <= 0:
        raise ValueError("Geometria não intersecta a imagem selecionada.")
    return window
