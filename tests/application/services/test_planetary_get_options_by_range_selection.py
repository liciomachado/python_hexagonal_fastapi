import unittest
from datetime import datetime
from unittest.mock import MagicMock

from app.application.services.dtos.planetary_images_filter_response import PlanetaryImageFilterResponse
from app.application.services.planetary_get_options_by_range import PlanetaryGetOptionImagesByRangeService


class PlanetaryGetOptionsByRangeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.service = PlanetaryGetOptionImagesByRangeService(stac_facade=MagicMock())

    def test_select_best_item_prefers_lower_cloud_cover_geometry(self):
        items = [
            PlanetaryImageFilterResponse(
                id="scene-high-geometry",
                datetime=datetime(2024, 6, 1, 12, 0, 0),
                cloud_cover=5.0,
                geometry={},
                assets={},
                cloud_cover_geometry=40.0,
            ),
            PlanetaryImageFilterResponse(
                id="scene-low-geometry",
                datetime=datetime(2024, 6, 1, 14, 0, 0),
                cloud_cover=50.0,
                geometry={},
                assets={},
                cloud_cover_geometry=8.0,
            ),
        ]

        best = self.service._select_best_item_by_geometry_cloud(items)

        self.assertEqual(best.id, "scene-low-geometry")

    def test_group_and_select_best_one_per_day_by_geometry_cloud(self):
        items = [
            PlanetaryImageFilterResponse(
                id="day1-a",
                datetime=datetime(2024, 6, 1, 10, 0, 0),
                cloud_cover=1.0,
                geometry={},
                assets={},
                cloud_cover_geometry=30.0,
            ),
            PlanetaryImageFilterResponse(
                id="day1-b",
                datetime=datetime(2024, 6, 1, 15, 0, 0),
                cloud_cover=90.0,
                geometry={},
                assets={},
                cloud_cover_geometry=5.0,
            ),
            PlanetaryImageFilterResponse(
                id="day2-a",
                datetime=datetime(2024, 6, 2, 12, 0, 0),
                cloud_cover=20.0,
                geometry={},
                assets={},
                cloud_cover_geometry=12.0,
            ),
        ]

        result = self.service._group_and_select_best(items)
        ids = {item.id for item in result}

        self.assertEqual(ids, {"day1-b", "day2-a"})

    def test_select_best_item_falls_back_to_scene_cloud_when_geometry_missing(self):
        items = [
            PlanetaryImageFilterResponse(
                id="scene-a",
                datetime=datetime(2024, 6, 1, 12, 0, 0),
                cloud_cover=40.0,
                geometry={},
                assets={},
                cloud_cover_geometry=None,
            ),
            PlanetaryImageFilterResponse(
                id="scene-b",
                datetime=datetime(2024, 6, 1, 14, 0, 0),
                cloud_cover=10.0,
                geometry={},
                assets={},
                cloud_cover_geometry=None,
            ),
        ]

        best = self.service._select_best_item_by_geometry_cloud(items)

        self.assertEqual(best.id, "scene-b")


if __name__ == "__main__":
    unittest.main()
