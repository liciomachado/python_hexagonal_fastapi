import unittest

import numpy as np

from app.application.services.sensor_profile import (
    get_sensor_profile,
    normalize_band_values,
)
from app.application.services.stac.satellite_collection import SatelliteCollection


class SensorProfileTests(unittest.TestCase):
    def test_sentinel_profile_has_scl_cloud_mask(self):
        profile = get_sensor_profile(SatelliteCollection.SENTINEL2_L2A)
        self.assertEqual(profile.cloud_mask_band, "SCL")
        self.assertFalse(profile.apply_reflectance_scale)
        self.assertEqual(profile.blob_prefix, "sentinel")

    def test_landsat_profile_has_qa_pixel_cloud_mask(self):
        profile = get_sensor_profile(SatelliteCollection.LANDSAT_C2_L2)
        self.assertEqual(profile.cloud_mask_band, "QA_PIXEL")
        self.assertTrue(profile.apply_reflectance_scale)
        self.assertEqual(profile.blob_prefix, "landsat")

    def test_normalize_band_values_sentinel_passthrough(self):
        profile = get_sensor_profile(SatelliteCollection.SENTINEL2_L2A)
        raw = np.array([100.0, 2000.0, 3000.0], dtype=np.float32)
        result = normalize_band_values(raw, profile)
        np.testing.assert_array_equal(result, raw)

    def test_normalize_band_values_landsat_applies_scale(self):
        profile = get_sensor_profile(SatelliteCollection.LANDSAT_C2_L2)
        raw = np.array([10000.0], dtype=np.float32)
        result = normalize_band_values(raw, profile)
        expected = np.clip(10000.0 * 0.0000275 - 0.2, 0.0, 1.0) * profile.rgb_clip_max
        self.assertAlmostEqual(float(result[0]), expected, places=2)


if __name__ == "__main__":
    unittest.main()
