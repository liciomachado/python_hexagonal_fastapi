import unittest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

from app.application.services.dtos.planetary_ndvi_image_response import PlanetaryNdviImageResponse
from app.application.services.planetary_get_options_by_range import RangeDayCandidate
from app.application.services.planetary_get_visual_image_service import PlanetaryVisualImageService
from app.application.services.stac.satellite_collection import SatelliteCollection
from app.application.services.stac.stac_types import StacProviderName
from app.core.utils.result import NotFoundError


POLYGON_WKT = "POLYGON((0 0,1 0,1 1,0 1,0 0))"


class GetNdviByRangeServiceTests(unittest.IsolatedAsyncioTestCase):
    def _build_service(self, candidates: list[RangeDayCandidate]) -> PlanetaryVisualImageService:
        range_service = MagicMock()
        range_service.search_best_items_by_day = AsyncMock(
            return_value=(candidates, StacProviderName.PLANETARY, SatelliteCollection.SENTINEL2_L2A)
        )
        service = PlanetaryVisualImageService(
            stac_facade=MagicMock(),
            range_images_service=range_service,
        )
        service._process_ndvi_from_item = MagicMock(return_value=(None, 0.5, 0.1, 0.9))
        service._upload_jpeg = AsyncMock(return_value="https://blob/ndvi.jpg")
        service._try_get_cached_ndvi_range = AsyncMock(return_value=None)
        service._set_cached_ndvi_range = AsyncMock()
        return service

    def _candidate(
        self,
        item_id: str,
        day: date,
        cloud_cover_geometry: float | None,
    ) -> RangeDayCandidate:
        stac_item = MagicMock()
        stac_item.id = item_id
        return RangeDayCandidate(
            id=item_id,
            datetime=datetime(day.year, day.month, day.day, 12, 0, 0),
            cloud_cover=10.0,
            cloud_cover_geometry=cloud_cover_geometry,
            stac_item=stac_item,
        )

    async def test_filters_days_above_cloud_limit(self):
        candidates = [
            self._candidate("ok-day", date(2024, 6, 1), 10.0),
            self._candidate("cloudy-day", date(2024, 6, 2), 50.0),
            self._candidate("edge-day", date(2024, 6, 3), 20.0),
        ]
        service = self._build_service(candidates)

        result = await service.get_ndvi_by_range(
            dt_start=datetime(2024, 6, 1),
            dt_end=datetime(2024, 6, 30),
            geometry=POLYGON_WKT,
            cloud_percentual=20.0,
            generate_image=False,
        )

        self.assertTrue(result.is_ok())
        payload = result.value()
        self.assertEqual([item.day for item in payload], [date(2024, 6, 1), date(2024, 6, 3)])
        self.assertEqual(payload[0].cloud_percentual, 10.0)
        self.assertEqual(payload[1].sat_image_id, "edge-day")
        self.assertEqual(service._process_ndvi_from_item.call_count, 2)

    async def test_returns_not_found_when_no_eligible_days(self):
        candidates = [
            self._candidate("cloudy-a", date(2024, 6, 1), 40.0),
            self._candidate("cloudy-b", date(2024, 6, 2), 55.0),
        ]
        service = self._build_service(candidates)

        result = await service.get_ndvi_by_range(
            dt_start=datetime(2024, 6, 1),
            dt_end=datetime(2024, 6, 30),
            geometry=POLYGON_WKT,
            cloud_percentual=20.0,
            generate_image=False,
        )

        self.assertTrue(result.is_err())
        self.assertIsInstance(result.error(), NotFoundError)

    async def test_skips_candidates_without_geometry_cloud(self):
        candidates = [
            self._candidate("missing-geometry", date(2024, 6, 1), None),
            self._candidate("valid", date(2024, 6, 2), 5.0),
        ]
        service = self._build_service(candidates)

        result = await service.get_ndvi_by_range(
            dt_start=datetime(2024, 6, 1),
            dt_end=datetime(2024, 6, 30),
            geometry=POLYGON_WKT,
            cloud_percentual=20.0,
            generate_image=False,
        )

        self.assertTrue(result.is_ok())
        payload = result.value()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0].sat_image_id, "valid")
        self.assertIsInstance(payload[0], PlanetaryNdviImageResponse)


if __name__ == "__main__":
    unittest.main()
