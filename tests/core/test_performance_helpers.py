import unittest

from app.application.services.raster_helpers import compute_out_shape, window_from_bounds
from app.core.cache_key import build_cache_key
from app.core.performance import PerformanceMetrics
from rasterio.transform import from_origin


class PerformanceOptimizationHelpersTests(unittest.TestCase):
    def test_compute_out_shape_scales_large_window(self):
        class Window:
            height = 10000
            width = 8000

        out_h, out_w = compute_out_shape(Window(), 1200)
        self.assertLessEqual(max(out_h, out_w), 1200)

    def test_compute_out_shape_keeps_small_window(self):
        class Window:
            height = 500
            width = 400

        out_h, out_w = compute_out_shape(Window(), 1200)
        self.assertEqual(out_h, 500)
        self.assertEqual(out_w, 400)

    def test_build_cache_key_is_deterministic(self):
        key_a = build_cache_key("ndvi", day="2024-06-01", geometry="abc")
        key_b = build_cache_key("ndvi", geometry="abc", day="2024-06-01")
        key_c = build_cache_key("ndvi", day="2024-06-02", geometry="abc")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_performance_metrics_records_span(self):
        metrics = PerformanceMetrics(context="unit")
        with metrics.span("bands_read"):
            pass
        self.assertIn("bands_read", metrics.spans)
        self.assertGreaterEqual(metrics.spans["bands_read"], 0.0)

    def test_window_from_bounds_clips_to_raster_edges(self):
        transform = from_origin(0, 54900, 10, 10)
        bounds_proj = (54890.0, 0.0, 54900.0, 10.0)
        window = window_from_bounds(bounds_proj, transform, 5490, 5490)
        self.assertLessEqual(window.col_off + window.width, 5490)
        self.assertLessEqual(window.row_off + window.height, 5490)
        self.assertGreater(window.width, 0)
        self.assertGreater(window.height, 0)

    def test_window_from_bounds_raises_when_outside_raster(self):
        transform = from_origin(0, 100, 10, 10)
        bounds_proj = (60000.0, -1000.0, 60100.0, 0.0)
        with self.assertRaises(ValueError):
            window_from_bounds(bounds_proj, transform, 5490, 5490)


if __name__ == "__main__":
    unittest.main()
