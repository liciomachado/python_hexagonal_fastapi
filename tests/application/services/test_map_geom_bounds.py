import unittest
from unittest.mock import MagicMock

from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageService


class MapGeomBoundsTests(unittest.TestCase):
    def _build_service(self, margin_ratio: float = 0.15, min_span: float = 0.001) -> PlanetaryVisualImageService:
        service = PlanetaryVisualImageService(MagicMock())
        service._bounds_margin_ratio = margin_ratio
        service._bounds_min_span = min_span
        return service

    def _geom_span(self, geom_bounds: tuple[float, float, float, float]) -> float:
        minx, miny, maxx, maxy = geom_bounds
        return max(maxx - minx, maxy - miny)

    def test_map_geom_keeps_consistent_margin_ratio_for_small_and_large_geometries(self):
        service = self._build_service(margin_ratio=0.15)
        small_wkt = "POLYGON((-47.0 -15.0,-46.999 -15.0,-46.999 -14.999,-47.0 -14.999,-47.0 -15.0))"
        large_wkt = "POLYGON((-47.0 -15.0,-46.9 -15.0,-46.9 -14.9,-47.0 -14.9,-47.0 -15.0))"

        _, _, small_bounds = service.map_geom(small_wkt)
        _, _, large_bounds = service.map_geom(large_wkt)

        small_geom, _, _ = service.map_geom(small_wkt)
        large_geom, _, _ = service.map_geom(large_wkt)

        small_ratio = max(small_geom.bounds[2] - small_geom.bounds[0], small_geom.bounds[3] - small_geom.bounds[1]) / self._geom_span(small_bounds)
        large_ratio = max(large_geom.bounds[2] - large_geom.bounds[0], large_geom.bounds[3] - large_geom.bounds[1]) / self._geom_span(large_bounds)

        self.assertAlmostEqual(small_ratio, large_ratio, places=4)
        self.assertAlmostEqual(small_ratio, 1 / (1 + 2 * 0.15), places=4)

    def test_map_geom_uses_min_span_for_point_like_geometries(self):
        service = self._build_service(margin_ratio=0.15, min_span=0.001)
        point_wkt = "POINT(-47.0 -15.0)"

        _, _, geom_bounds = service.map_geom(point_wkt)

        self.assertAlmostEqual(self._geom_span(geom_bounds), 0.001 * (1 + 2 * 0.15), places=6)


if __name__ == "__main__":
    unittest.main()
