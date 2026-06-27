import unittest
from unittest.mock import MagicMock

import pystac

from app.application.services.stac.satellite_collection import SatelliteCollection
from app.application.services.stac.stac_types import resolve_band_href, resolve_collection_from_item


class StacTypesTests(unittest.TestCase):
    def _build_item(self, collection: str, assets: dict[str, str]) -> pystac.Item:
        item = MagicMock(spec=pystac.Item)
        item.collection = collection
        item.assets = {
            key: MagicMock(href=href) for key, href in assets.items()
        }
        return item

    def test_resolve_band_href_sentinel_aliases(self):
        item = self._build_item(
            "sentinel-2-l2a",
            {"B04": "https://example.com/b04.tif", "B08": "https://example.com/b08.tif"},
        )
        self.assertEqual(resolve_band_href(item, "B04"), "https://example.com/b04.tif")
        self.assertEqual(resolve_band_href(item, "B08"), "https://example.com/b08.tif")

    def test_resolve_band_href_landsat_aliases(self):
        item = self._build_item(
            "landsat-c2-l2",
            {
                "red": "https://example.com/red.tif",
                "nir08": "https://example.com/nir08.tif",
                "swir16": "https://example.com/swir16.tif",
                "qa_pixel": "https://example.com/qa_pixel.tif",
            },
        )
        self.assertEqual(resolve_band_href(item, "B04"), "https://example.com/red.tif")
        self.assertEqual(resolve_band_href(item, "B08"), "https://example.com/nir08.tif")
        self.assertEqual(resolve_band_href(item, "B11"), "https://example.com/swir16.tif")
        self.assertEqual(resolve_band_href(item, "QA_PIXEL"), "https://example.com/qa_pixel.tif")

    def test_resolve_collection_from_item(self):
        sentinel = self._build_item("sentinel-2-l2a", {})
        landsat = self._build_item("landsat-c2-l2", {})
        self.assertEqual(resolve_collection_from_item(sentinel), SatelliteCollection.SENTINEL2_L2A)
        self.assertEqual(resolve_collection_from_item(landsat), SatelliteCollection.LANDSAT_C2_L2)


if __name__ == "__main__":
    unittest.main()
