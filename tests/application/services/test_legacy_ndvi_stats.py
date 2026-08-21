import unittest
from unittest import mock

import numpy as np

from app.application.services.legacy_ndvi_stats import (
    EMPTY_NDVI_STATS,
    SCL_VALID_CLASSES,
    aggregate_ndvi_stats,
    build_valid_pixel_mask,
    calc_ndvi,
    classify_ndvi_quality,
    compute_legacy_ndvi_stats,
    compute_ndvi_stats_from_bands,
)
from app.application.services.raster_helpers import QA_PIXEL_CLOUD, QA_PIXEL_CLOUD_SHADOW
from app.application.services.sensor_profile import get_sensor_profile
from app.application.services.stac.satellite_collection import SatelliteCollection


class LegacyNdviStatsTests(unittest.TestCase):
    def test_calc_ndvi_applies_legacy_filters(self):
        nir = np.array([[0, 100], [200, 300]], dtype=np.float32)
        red = np.array([[0, 50], [100, 100]], dtype=np.float32)

        ndvi = calc_ndvi(nir, red)

        self.assertTrue(np.isnan(ndvi[0, 0]))
        self.assertAlmostEqual(float(ndvi[1, 1]), 0.5, places=3)

    def test_sentinel_scl_mask_keeps_only_valid_classes(self):
        profile = get_sensor_profile(SatelliteCollection.SENTINEL2_L2A)
        scl = np.array([[3, 4], [8, 6]], dtype=np.uint8)

        mask = build_valid_pixel_mask(scl, scl.shape, profile)

        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[1, 0])
        self.assertTrue(mask[1, 1])
        self.assertEqual(tuple(SCL_VALID_CLASSES), (4, 5, 6))

    def test_landsat_qa_mask_excludes_cloud_and_shadow(self):
        profile = get_sensor_profile(SatelliteCollection.LANDSAT_C2_L2)
        qa = np.array(
            [
                [0, QA_PIXEL_CLOUD],
                [QA_PIXEL_CLOUD_SHADOW, QA_PIXEL_CLOUD | QA_PIXEL_CLOUD_SHADOW],
            ],
            dtype=np.uint16,
        )

        mask = build_valid_pixel_mask(qa, qa.shape, profile)

        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertFalse(mask[1, 0])
        self.assertFalse(mask[1, 1])

    def test_aggregate_ndvi_stats_rounds_before_mean(self):
        ndvi = np.array([0.3334, 0.3336, 0.5000])
        valid_mask = np.array([True, True, True])

        stats = aggregate_ndvi_stats(ndvi, valid_mask)

        self.assertAlmostEqual(stats.ndvi_mean, 0.389, places=3)
        self.assertAlmostEqual(stats.ndvi_min, 0.333, places=3)
        self.assertAlmostEqual(stats.ndvi_max, 0.5, places=3)
        self.assertEqual(stats.valid_pixels, 3)
        self.assertEqual(stats.total_pixels, 3)
        self.assertEqual(stats.valid_percentage, 100.0)
        self.assertEqual(stats.quality, "GOOD")

    def test_classify_ndvi_quality_thresholds(self):
        self.assertEqual(classify_ndvi_quality(80.0), "GOOD")
        self.assertEqual(classify_ndvi_quality(79.9), "MODERATE")
        self.assertEqual(classify_ndvi_quality(50.0), "MODERATE")
        self.assertEqual(classify_ndvi_quality(49.9), "LOW_QUALITY")

    def test_aggregate_ndvi_stats_quality_bands(self):
        ndvi = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        good = aggregate_ndvi_stats(ndvi, np.array([True, True, True, True]))
        self.assertEqual(good.valid_percentage, 100.0)
        self.assertEqual(good.quality, "GOOD")

        moderate = aggregate_ndvi_stats(ndvi, np.array([True, True, False, False]))
        self.assertEqual(moderate.valid_pixels, 2)
        self.assertEqual(moderate.total_pixels, 4)
        self.assertEqual(moderate.valid_percentage, 50.0)
        self.assertEqual(moderate.quality, "MODERATE")

        low = aggregate_ndvi_stats(ndvi, np.array([True, False, False, False]))
        self.assertEqual(low.valid_percentage, 25.0)
        self.assertEqual(low.quality, "LOW_QUALITY")

    def test_compute_ndvi_stats_from_bands_matches_pipeline(self):
        nir = np.array([682, 4228, 4228, 682], dtype=np.float32)
        red = np.array([800, 2000, 2000, 800], dtype=np.float32)
        scl = np.array([9, 4, 5, 10], dtype=np.uint8)
        profile = get_sensor_profile(SatelliteCollection.SENTINEL2_L2A)

        ndvi = calc_ndvi(nir, red)
        mask = build_valid_pixel_mask(scl, ndvi.shape, profile)
        expected = aggregate_ndvi_stats(ndvi, mask)
        actual = compute_ndvi_stats_from_bands(nir, red, scl, profile)

        self.assertEqual(expected, actual)
        self.assertEqual(actual.valid_pixels, 2)
        self.assertEqual(actual.total_pixels, 4)
        self.assertEqual(actual.valid_percentage, 50.0)
        self.assertEqual(actual.quality, "MODERATE")

    def test_regression_cloudy_pixels_excluded_like_legacy(self):
        """Simula cenário do dia 2026-04-15: nuvens reduzem média sem máscara SCL."""
        nir = np.array([682, 4228, 4228, 682], dtype=np.float32)
        red = np.array([800, 2000, 2000, 800], dtype=np.float32)
        scl = np.array([9, 4, 5, 10], dtype=np.uint8)

        ndvi = calc_ndvi(nir, red)
        profile = get_sensor_profile(SatelliteCollection.SENTINEL2_L2A)
        valid_mask = build_valid_pixel_mask(scl, ndvi.shape, profile)

        unmasked_mean = float(np.nanmean(ndvi))
        stats = aggregate_ndvi_stats(ndvi, valid_mask)

        self.assertLess(unmasked_mean, stats.ndvi_mean)
        self.assertAlmostEqual(unmasked_mean, 0.1391, places=3)
        self.assertAlmostEqual(stats.ndvi_mean, 0.358, places=3)
        self.assertEqual(stats.valid_pixels, 2)
        self.assertEqual(stats.total_pixels, 4)
        self.assertEqual(stats.quality, "MODERATE")

    def test_compute_legacy_ndvi_stats_reads_bands_in_parallel(self):
        profile = get_sensor_profile(SatelliteCollection.SENTINEL2_L2A)

        class DummyItem:
            assets = {"B08": {"href": "nir"}, "B04": {"href": "red"}, "SCL": {"href": "scl"}}

        from shapely.geometry import Polygon

        geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        nir = np.array([[682, 4228], [4228, 682]], dtype=np.float32)
        red = np.array([[800, 2000], [2000, 800]], dtype=np.float32)
        scl = np.array([[4, 4], [5, 5]], dtype=np.uint8)

        with mock.patch(
            "app.application.services.legacy_ndvi_stats.resolve_band_href",
            side_effect=lambda item, key: item.assets[key]["href"],
        ), mock.patch(
            "app.application.services.legacy_ndvi_stats.read_polygon_masked_band",
            side_effect=lambda href, _: {"nir": nir, "red": red, "scl": scl}[href],
        ) as read_mock, mock.patch(
            "app.application.services.legacy_ndvi_stats.ThreadPoolExecutor",
        ) as executor_cls:
            mock_executor = mock.MagicMock()
            executor_cls.return_value.__enter__.return_value = mock_executor

            def fake_submit(fn, *args):
                future = mock.Mock()
                future.result.return_value = fn(*args)
                return future

            mock_executor.submit.side_effect = fake_submit

            stats = compute_legacy_ndvi_stats(
                DummyItem(),
                geom,
                profile,
                sign_url=lambda href: href,
            )

        executor_cls.assert_called_once_with(max_workers=3)
        self.assertEqual(mock_executor.submit.call_count, 3)
        self.assertEqual(read_mock.call_count, 3)
        self.assertIsNotNone(stats.ndvi_mean)
        self.assertEqual(stats.valid_pixels, 4)
        self.assertEqual(stats.quality, "GOOD")

    def test_compute_legacy_ndvi_stats_returns_none_when_bands_missing(self):
        profile = get_sensor_profile(SatelliteCollection.SENTINEL2_L2A)

        class DummyItem:
            assets = {}

        from shapely.geometry import Polygon

        geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        with mock.patch(
            "app.application.services.legacy_ndvi_stats.resolve_band_href",
            return_value="https://example.com/band.tif",
        ), mock.patch(
            "app.application.services.legacy_ndvi_stats.read_polygon_masked_band",
            return_value=None,
        ):
            stats = compute_legacy_ndvi_stats(
                DummyItem(),
                geom,
                profile,
                sign_url=lambda href: href,
            )

        self.assertEqual(stats, EMPTY_NDVI_STATS)


if __name__ == "__main__":
    unittest.main()
