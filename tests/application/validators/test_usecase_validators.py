import unittest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

from app.application.usecases.get_images_by_range import (
    GetImagesByRangeRequest,
    GetImagesByRangeUseCase,
)
from app.application.usecases.get_visual_image_by_day import (
    GetVisualImageByDayRequest,
    GetVisualImageByDayUseCase,
)
from app.application.validators.usecase_validators import (
    require_valid_date_range,
    require_valid_sentinel_geometry,
)
from app.core.utils.result import BadRequestError

VALID_POLYGON = (
    "POLYGON(("
    "-51.4399892251983 -24.1616475755989,"
    "-51.4399048581994 -24.1370045406373,"
    "-51.4001411759301 -24.1371139314813,"
    "-51.4002179173859 -24.1617570920765,"
    "-51.4399892251983 -24.1616475755989"
    "))"
)


class UseCaseValidatorsTests(unittest.TestCase):
    def test_require_valid_sentinel_geometry_accepts_valid_wkt(self):
        result = require_valid_sentinel_geometry(VALID_POLYGON)

        self.assertTrue(result.is_ok())

    def test_require_valid_sentinel_geometry_rejects_invalid_wkt(self):
        result = require_valid_sentinel_geometry("POINT(0 0)")

        self.assertTrue(result.is_err())
        self.assertIsInstance(result.error(), BadRequestError)
        self.assertIn("POLYGON e MULTIPOLYGON", result.error().message)

    def test_require_valid_date_range_accepts_valid_period(self):
        result = require_valid_date_range(
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        self.assertTrue(result.is_ok())

    def test_require_valid_date_range_rejects_inverted_period(self):
        result = require_valid_date_range(
            datetime(2024, 2, 1),
            datetime(2024, 1, 1),
        )

        self.assertTrue(result.is_err())
        self.assertIsInstance(result.error(), BadRequestError)
        self.assertIn("dt_start não pode ser maior que dt_end", result.error().message)


class GetImagesByRangeUseCaseValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_returns_400_when_date_range_is_invalid(self):
        service = MagicMock()
        service.search_images = AsyncMock()
        usecase = GetImagesByRangeUseCase(service)

        result = await usecase.execute(
            GetImagesByRangeRequest(
                dt_start=datetime(2024, 2, 1),
                dt_end=datetime(2024, 1, 1),
                geom=VALID_POLYGON,
            )
        )

        self.assertTrue(result.is_err())
        self.assertIn("dt_start não pode ser maior que dt_end", result.error().message)
        service.search_images.assert_not_called()

    async def test_execute_returns_400_when_geometry_is_invalid(self):
        service = MagicMock()
        service.search_images = AsyncMock()
        usecase = GetImagesByRangeUseCase(service)

        result = await usecase.execute(
            GetImagesByRangeRequest(
                dt_start=datetime(2024, 1, 1),
                dt_end=datetime(2024, 1, 31),
                geom="POLYGON((invalid))",
            )
        )

        self.assertTrue(result.is_err())
        self.assertIn("Geometria inválida", result.error().message)
        service.search_images.assert_not_called()


class GetVisualImageByDayUseCaseValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_returns_400_when_geometry_is_invalid(self):
        service = MagicMock()
        service.get_visual_image = AsyncMock()
        usecase = GetVisualImageByDayUseCase(service)

        result = await usecase.execute(
            GetVisualImageByDayRequest(
                day=date(2024, 6, 1),
                cloud_percentual=10.0,
                geometry="POINT(0 0)",
            )
        )

        self.assertTrue(result.is_err())
        self.assertIn("POLYGON e MULTIPOLYGON", result.error().message)
        service.get_visual_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
