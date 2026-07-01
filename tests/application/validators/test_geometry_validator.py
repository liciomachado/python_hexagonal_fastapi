import unittest

from app.application.validators.geometry_validator import validate_epsg4326_wkt_geometry
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

VALID_MULTIPOLYGON = (
    "MULTIPOLYGON((("
    "-51.44 -24.16,-51.40 -24.16,-51.40 -24.13,-51.44 -24.13,-51.44 -24.16"
    ")))"
)


class GeometryValidatorTests(unittest.TestCase):
    def test_accepts_valid_polygon(self):
        validate_epsg4326_wkt_geometry(VALID_POLYGON)

    def test_accepts_valid_multipolygon(self):
        validate_epsg4326_wkt_geometry(VALID_MULTIPOLYGON)

    def test_accepts_srid_4326_prefix(self):
        validate_epsg4326_wkt_geometry(f"SRID=4326;{VALID_POLYGON}")

    def test_rejects_empty_geometry(self):
        with self.assertRaises(BadRequestError) as ctx:
            validate_epsg4326_wkt_geometry("   ")

        self.assertEqual(ctx.exception.message, "Geometria não informada.")

    def test_rejects_invalid_wkt(self):
        with self.assertRaises(BadRequestError) as ctx:
            validate_epsg4326_wkt_geometry("POLYGON((invalid))")

        self.assertIn("não foi possível interpretar o WKT", ctx.exception.message)

    def test_rejects_unsupported_geometry_type(self):
        with self.assertRaises(BadRequestError) as ctx:
            validate_epsg4326_wkt_geometry("POINT(-47.0 -15.0)")

        self.assertIn("POLYGON e MULTIPOLYGON", ctx.exception.message)

    def test_rejects_invalid_srid(self):
        with self.assertRaises(BadRequestError) as ctx:
            validate_epsg4326_wkt_geometry(f"SRID=3857;{VALID_POLYGON}")

        self.assertIn("EPSG:4326", ctx.exception.message)

    def test_rejects_coordinates_outside_wgs84_bounds(self):
        with self.assertRaises(BadRequestError) as ctx:
            validate_epsg4326_wkt_geometry(
                "POLYGON((181.0 -24.0,182.0 -24.0,182.0 -23.0,181.0 -23.0,181.0 -24.0))"
            )

        self.assertIn("coordenadas fora do intervalo EPSG:4326", ctx.exception.message)

    def test_rejects_self_intersecting_polygon(self):
        with self.assertRaises(BadRequestError) as ctx:
            validate_epsg4326_wkt_geometry(
                "POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))"
            )

        self.assertTrue(ctx.exception.message.startswith("Geometria inválida:"))


if __name__ == "__main__":
    unittest.main()
