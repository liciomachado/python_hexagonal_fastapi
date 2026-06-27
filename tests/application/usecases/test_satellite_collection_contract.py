import unittest

from app.application.services.stac.satellite_collection import (
    DEFAULT_SATELLITE_COLLECTION,
    SatelliteCollection,
)
from app.application.usecases.get_images_by_range import GetImagesByRangeRequest
from app.application.usecases.get_visual_image_by_day import GetVisualImageByDayRequest
from datetime import date, datetime


class SatelliteCollectionContractTests(unittest.TestCase):
    def test_visual_request_defaults_to_sentinel(self):
        request = GetVisualImageByDayRequest(
            day=date(2024, 6, 1),
            cloud_percentual=10.0,
            geometry="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        )
        self.assertEqual(request.satellite_collection, DEFAULT_SATELLITE_COLLECTION)
        self.assertEqual(request.satellite_collection, SatelliteCollection.SENTINEL2_L2A)

    def test_range_request_accepts_landsat(self):
        request = GetImagesByRangeRequest(
            dt_start=datetime(2007, 1, 1),
            dt_end=datetime(2007, 12, 31),
            geom="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            satellite_collection=SatelliteCollection.LANDSAT_C2_L2,
        )
        self.assertEqual(request.satellite_collection, SatelliteCollection.LANDSAT_C2_L2)


if __name__ == "__main__":
    unittest.main()
