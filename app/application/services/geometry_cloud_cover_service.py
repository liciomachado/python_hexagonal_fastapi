import numpy as np
import pyproj
import pystac
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from app.application.services.raster_helpers import (
    QA_PIXEL_CLOUD_MASK_BITS,
    SCL_CLOUD_CLASSES,
    build_rasterio_gdal_config,
    compute_out_shape,
    window_from_bounds,
)
from app.application.services.sensor_profile import SensorProfile, get_sensor_profile
from app.application.services.stac.satellite_collection import SatelliteCollection
from app.application.services.stac.stac_resilient_facade import StacResilientFacade
from app.application.services.stac.stac_types import StacProviderName, resolve_band_href, resolve_collection_from_item
from app.core.config import Config


class GeometryCloudCoverService:
    def __init__(self, stac_facade: StacResilientFacade):
        self._stac_facade = stac_facade
        self._gdal_config = build_rasterio_gdal_config()

    def compute_cloud_percentual_over_geometry(
        self,
        item: pystac.Item,
        geom: BaseGeometry,
        geom_bounds: tuple[float, float, float, float],
        provider: StacProviderName,
        collection: SatelliteCollection | None = None,
    ) -> float:
        if collection is None:
            collection = resolve_collection_from_item(item)
        profile = get_sensor_profile(collection)

        try:
            mask_href = self._sign_url(resolve_band_href(item, profile.cloud_mask_band), provider)
        except KeyError:
            return float(item.properties.get("eo:cloud_cover", 0.0))

        with rasterio.Env(**self._gdal_config):
            with rasterio.open(mask_href) as src:
                image_crs = src.crs
                geom_bounds_proj = transform_bounds("EPSG:4326", src.crs, *geom_bounds)
                window = window_from_bounds(geom_bounds_proj, src.transform, src.width, src.height)
                mask_out_shape = compute_out_shape(window, Config.SCL_MAX_DIMENSION)
                mask_band = src.read(
                    1,
                    window=window,
                    out_shape=mask_out_shape,
                    resampling=Resampling.average,
                )
                crop_transform = src.window_transform(window)
                scale_x = window.width / mask_out_shape[1]
                scale_y = window.height / mask_out_shape[0]
                crop_transform = crop_transform * crop_transform.scale(scale_x, scale_y)

        project = pyproj.Transformer.from_crs("EPSG:4326", image_crs, always_xy=True).transform
        geom_proj = shapely_transform(project, geom)
        mask = geometry_mask([mapping(geom_proj)], out_shape=mask_band.shape, transform=crop_transform, invert=True)
        mask_inside = mask_band[mask]
        if mask_inside.size == 0:
            return float(item.properties.get("eo:cloud_cover", 0.0))

        if profile.cloud_mask_band == "SCL":
            cloud_pixels = int(np.isin(mask_inside, list(SCL_CLOUD_CLASSES)).sum())
        else:
            cloud_pixels = int((mask_inside.astype(np.uint32) & QA_PIXEL_CLOUD_MASK_BITS).astype(bool).sum())

        return round(cloud_pixels / mask_inside.size * 100, 2)

    def _sign_url(self, href: str, provider: StacProviderName) -> str:
        return self._stac_facade.sign_asset_url(href, provider)
