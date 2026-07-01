import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from shapely import wkt

from app.application.services.geometry_bounds import (
    CLOUD_COVER_GEOMETRY_BUFFER,
    compute_cloud_cover_geom_bounds,
)
from app.application.services.planetary_get_options_by_range import PlanetaryGetOptionImagesByRangeService
from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageService
from app.application.services.stac.satellite_collection import SatelliteCollection
from app.application.services.stac.stac_types import StacProviderName
from app.core.performance import PerformanceMetrics


POLYGON_WKT = "POLYGON((-47.0 -15.0,-46.9 -15.0,-46.9 -14.9,-47.0 -14.9,-47.0 -15.0))"


class GeometryBoundsTests(unittest.TestCase):
    def test_compute_cloud_cover_geom_bounds_applies_fixed_buffer(self):
        geom = wkt.loads(POLYGON_WKT)
        minx, miny, maxx, maxy = geom.bounds

        bounds = compute_cloud_cover_geom_bounds(geom)

        self.assertEqual(
            bounds,
            (
                minx - CLOUD_COVER_GEOMETRY_BUFFER,
                miny - CLOUD_COVER_GEOMETRY_BUFFER,
                maxx + CLOUD_COVER_GEOMETRY_BUFFER,
                maxy + CLOUD_COVER_GEOMETRY_BUFFER,
            ),
        )


class CloudCoverBoundsConsistencyTests(unittest.TestCase):
    def test_options_by_range_uses_shared_cloud_bounds(self):
        service = PlanetaryGetOptionImagesByRangeService(MagicMock())
        geom, bounds = service._parse_geometry(POLYGON_WKT)

        self.assertEqual(bounds, compute_cloud_cover_geom_bounds(geom))

    def test_visual_render_bounds_differ_from_cloud_bounds(self):
        service = PlanetaryVisualImageService(MagicMock())
        geom, _, render_bounds = service.map_geom(POLYGON_WKT)
        cloud_bounds = compute_cloud_cover_geom_bounds(geom)

        self.assertNotEqual(render_bounds, cloud_bounds)


class PrepareContextCloudBoundsTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_context_uses_cloud_bounds_not_render_bounds(self):
        service = PlanetaryVisualImageService(MagicMock())
        cloud_cover_service = MagicMock()
        cloud_cover_service.compute_cloud_percentual_over_geometry.return_value = 5.4
        service._cloud_cover_service = cloud_cover_service

        selected_item = MagicMock()
        selected_item.id = "LE07_L2SP_228068_20080521_02_T1"
        service._search_selected_item = AsyncMock(
            return_value=(selected_item, StacProviderName.PLANETARY),
        )

        selected, provider, geom, render_bounds, geometry_cloud_percentual = await service._prepare_context(
            day=date(2008, 5, 21),
            cloud_percentual=100.0,
            geometry=POLYGON_WKT,
            preferred_provider=None,
            metrics=PerformanceMetrics(context="test"),
            satellite_collection=SatelliteCollection.LANDSAT_C2_L2,
        )

        cloud_bounds = compute_cloud_cover_geom_bounds(geom)
        _, _, expected_render_bounds = service.map_geom(POLYGON_WKT)
        call_bounds = cloud_cover_service.compute_cloud_percentual_over_geometry.call_args[0][2]

        self.assertEqual(call_bounds, cloud_bounds)
        self.assertNotEqual(call_bounds, expected_render_bounds)
        self.assertEqual(render_bounds, expected_render_bounds)
        self.assertEqual(geometry_cloud_percentual, 5.4)
        self.assertEqual(selected, selected_item)
        self.assertEqual(provider, StacProviderName.PLANETARY)


if __name__ == "__main__":
    unittest.main()
