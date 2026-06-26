import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from app.application.services.dtos.planetary_all_images_response import PlanetaryAllImagesResponse
from app.application.services.dtos.planetary_ndvi_image_response import PlanetaryNdviImageResponse
from app.application.services.dtos.planetary_visual_image_response import PlanetaryImageVisualResponse
from app.application.usecases.get_all_images_by_day import (
    GetAllImagesByDayRequest,
    GetAllImagesByDayUseCase,
)
from app.core.utils.result import Result


class GetAllImagesByDayUseCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_maps_unified_service_response(self):
        service = MagicMock()
        service.get_all_images_by_day = AsyncMock(
            return_value=Result.Ok(
                PlanetaryAllImagesResponse(
                    visual=PlanetaryImageVisualResponse(
                        day=date(2024, 6, 1),
                        cloud_percentual=10.0,
                        image_url="https://blob/visual.jpg",
                    ),
                    ndvi=PlanetaryNdviImageResponse(
                        day=date(2024, 6, 1),
                        cloud_percentual=10.0,
                        image_url="https://blob/ndvi.jpg",
                        ndvi_mean=0.5,
                        ndvi_min=0.1,
                        ndvi_max=0.9,
                        sat_image_id="item-1",
                    ),
                    ndmi=PlanetaryNdviImageResponse(
                        day=date(2024, 6, 1),
                        cloud_percentual=10.0,
                        image_url="https://blob/ndmi.jpg",
                        ndvi_mean=0.2,
                        ndvi_min=-0.1,
                        ndvi_max=0.4,
                        sat_image_id="item-1",
                    ),
                )
            )
        )
        usecase = GetAllImagesByDayUseCase(service)

        result = await usecase.execute(
            GetAllImagesByDayRequest(
                day=date(2024, 6, 1),
                cloud_percentual=10.0,
                geometry="POLYGON((0 0,1 0,1 1,0 1,0 0))",
                generate_image=True,
            )
        )

        self.assertTrue(result.is_ok())
        payload = result.value()
        self.assertEqual(payload.visual.image_url, "https://blob/visual.jpg")
        self.assertEqual(payload.ndvi.ndvi_mean, 0.5)
        self.assertEqual(payload.ndmi.ndmi_mean, 0.2)
        service.get_all_images_by_day.assert_awaited_once()

    async def test_execute_propagates_service_error(self):
        service = MagicMock()
        service.get_all_images_by_day = AsyncMock(return_value=Result.Err("erro"))
        usecase = GetAllImagesByDayUseCase(service)

        result = await usecase.execute(
            GetAllImagesByDayRequest(
                day=date(2024, 6, 1),
                cloud_percentual=10.0,
                geometry="POLYGON((0 0,1 0,1 1,0 1,0 0))",
            )
        )

        self.assertTrue(result.is_err())


if __name__ == "__main__":
    unittest.main()
