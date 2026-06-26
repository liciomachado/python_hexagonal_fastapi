import unittest

from app.application.services.planetary_get_visual_image_service import _compute_out_shape
from app.core.cache_key import build_cache_key
from app.core.performance import PerformanceMetrics


class PerformanceOptimizationHelpersTests(unittest.TestCase):
    def test_compute_out_shape_scales_large_window(self):
        class Window:
            height = 10000
            width = 8000

        out_h, out_w = _compute_out_shape(Window(), 1200)
        self.assertLessEqual(max(out_h, out_w), 1200)

    def test_compute_out_shape_keeps_small_window(self):
        class Window:
            height = 500
            width = 400

        out_h, out_w = _compute_out_shape(Window(), 1200)
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


if __name__ == "__main__":
    unittest.main()
