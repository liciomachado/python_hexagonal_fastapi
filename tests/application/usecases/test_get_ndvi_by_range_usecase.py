import unittest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

from app.application.services.dtos.planetary_ndvi_image_response import PlanetaryNdviImageResponse
from app.application.usecases.get_ndvi_by_range import (
    GetNdviByRangeRequest,
    GetNdviByRangeUseCase,
)
from app.core.utils.result import NotFoundError, Result


class GetNdviByRangeUseCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_maps_service_response(self):
        service = MagicMock()
        service.get_ndvi_by_range = AsyncMock(
            return_value=Result.Ok(
                [
                    PlanetaryNdviImageResponse(
                        day=date(2024, 6, 1),
                        cloud_percentual=8.5,
                        image_url=None,
                        ndvi_mean=0.55,
                        ndvi_min=0.1,
                        ndvi_max=0.9,
                        sat_image_id="item-1",
                    ),
                    PlanetaryNdviImageResponse(
                        day=date(2024, 6, 5),
                        cloud_percentual=12.0,
                        image_url="https://blob/ndvi.jpg",
                        ndvi_mean=0.4,
                        ndvi_min=0.0,
                        ndvi_max=0.8,
                        sat_image_id="item-2",
                    ),
                ]
            )
        )
        usecase = GetNdviByRangeUseCase(service)

        result = await usecase.execute(
            GetNdviByRangeRequest(
                dt_start=datetime(2024, 6, 1),
                dt_end=datetime(2024, 6, 30),
                geometry="POLYGON((0 0,1 0,1 1,0 1,0 0))",
                cloud_percentual=20.0,
                generate_image=False,
            )
        )

        self.assertTrue(result.is_ok())
        payload = result.value()
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0].ndvi_mean, 0.55)
        self.assertEqual(payload[0].cloud_percentual, 8.5)
        self.assertEqual(payload[1].sat_image_id, "item-2")
        service.get_ndvi_by_range.assert_awaited_once()

    async def test_execute_propagates_service_error(self):
        service = MagicMock()
        service.get_ndvi_by_range = AsyncMock(return_value=Result.Err(NotFoundError("No images found")))
        usecase = GetNdviByRangeUseCase(service)

        result = await usecase.execute(
            GetNdviByRangeRequest(
                dt_start=datetime(2024, 6, 1),
                dt_end=datetime(2024, 6, 30),
                geometry="POLYGON((0 0,1 0,1 1,0 1,0 0))",
                cloud_percentual=20.0,
            )
        )

        self.assertTrue(result.is_err())

    async def test_execute_rejects_invalid_date_range(self):
        service = MagicMock()
        service.get_ndvi_by_range = AsyncMock()
        usecase = GetNdviByRangeUseCase(service)

        result = await usecase.execute(
            GetNdviByRangeRequest(
                dt_start=datetime(2024, 6, 30),
                dt_end=datetime(2024, 6, 1),
                geometry="POLYGON((0 0,1 0,1 1,0 1,0 0))",
                cloud_percentual=20.0,
            )
        )

        self.assertTrue(result.is_err())
        service.get_ndvi_by_range.assert_not_called()


if __name__ == "__main__":
    unittest.main()
