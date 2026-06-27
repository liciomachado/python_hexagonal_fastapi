import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from shapely.geometry import box

from app.application.services.geometry_cloud_cover_service import GeometryCloudCoverService
from app.application.services.raster_helpers import QA_PIXEL_CLOUD, QA_PIXEL_CLOUD_SHADOW
from app.application.services.stac.satellite_collection import SatelliteCollection
from app.application.services.stac.stac_types import StacProviderName


class GeometryCloudCoverServiceTests(unittest.TestCase):
    def _build_service(self) -> GeometryCloudCoverService:
        facade = MagicMock()
        facade.sign_asset_url.side_effect = lambda href, provider: href
        return GeometryCloudCoverService(facade)

    @patch("app.application.services.geometry_cloud_cover_service.rasterio.open")
    @patch("app.application.services.geometry_cloud_cover_service.rasterio.Env")
    def test_qa_pixel_cloud_percentual(self, mock_env, mock_open):
        service = self._build_service()
        geom = box(-47.0, -15.0, -46.9, -14.9)
        geom_bounds = (-47.05, -15.05, -46.85, -14.85)

        cloud_value = QA_PIXEL_CLOUD
        clear_value = 0
        qa_data = np.array(
            [[clear_value, cloud_value], [cloud_value, clear_value]],
            dtype=np.uint16,
        )

        mock_src = MagicMock()
        mock_src.crs = "EPSG:32623"
        mock_src.transform = MagicMock()
        mock_src.width = 2
        mock_src.height = 2
        mock_src.read.return_value = qa_data
        mock_src.window_transform.return_value = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_src

        item = MagicMock()
        item.properties = {"eo:cloud_cover": 50.0}
        item.assets = {"qa_pixel": MagicMock(href="https://example.com/qa_pixel.tif")}

        with patch(
            "app.application.services.geometry_cloud_cover_service.transform_bounds",
            return_value=(0, 0, 100, 100),
        ), patch(
            "app.application.services.geometry_cloud_cover_service.window_from_bounds",
            return_value=MagicMock(width=2, height=2),
        ), patch(
            "app.application.services.geometry_cloud_cover_service.compute_out_shape",
            return_value=(2, 2),
        ), patch(
            "app.application.services.geometry_cloud_cover_service.geometry_mask",
            return_value=np.ones((2, 2), dtype=bool),
        ):
            result = service.compute_cloud_percentual_over_geometry(
                item,
                geom,
                geom_bounds,
                StacProviderName.PLANETARY,
                SatelliteCollection.LANDSAT_C2_L2,
            )

        self.assertEqual(result, 50.0)

    @patch("app.application.services.geometry_cloud_cover_service.rasterio.open")
    @patch("app.application.services.geometry_cloud_cover_service.rasterio.Env")
    def test_qa_pixel_detects_cloud_shadow(self, mock_env, mock_open):
        service = self._build_service()
        geom = box(-47.0, -15.0, -46.9, -14.9)
        geom_bounds = (-47.05, -15.05, -46.85, -14.85)

        qa_data = np.array([[QA_PIXEL_CLOUD_SHADOW, 0]], dtype=np.uint16)

        mock_src = MagicMock()
        mock_src.crs = "EPSG:32623"
        mock_src.transform = MagicMock()
        mock_src.width = 2
        mock_src.height = 1
        mock_src.read.return_value = qa_data
        mock_src.window_transform.return_value = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_src

        item = MagicMock()
        item.properties = {"eo:cloud_cover": 0.0}
        item.assets = {"qa_pixel": MagicMock(href="https://example.com/qa_pixel.tif")}

        with patch(
            "app.application.services.geometry_cloud_cover_service.transform_bounds",
            return_value=(0, 0, 100, 100),
        ), patch(
            "app.application.services.geometry_cloud_cover_service.window_from_bounds",
            return_value=MagicMock(width=2, height=1),
        ), patch(
            "app.application.services.geometry_cloud_cover_service.compute_out_shape",
            return_value=(1, 2),
        ), patch(
            "app.application.services.geometry_cloud_cover_service.geometry_mask",
            return_value=np.array([[True, True]]),
        ):
            result = service.compute_cloud_percentual_over_geometry(
                item,
                geom,
                geom_bounds,
                StacProviderName.PLANETARY,
                SatelliteCollection.LANDSAT_C2_L2,
            )

        self.assertEqual(result, 50.0)


if __name__ == "__main__":
    unittest.main()
