from collections.abc import Iterator

from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from app.core.utils.result import BadRequestError

ALLOWED_GEOMETRY_TYPES = frozenset({"Polygon", "MultiPolygon"})
EPSG_4326 = "4326"
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0


def validate_epsg4326_wkt_geometry(geometry: str | None) -> None:
    if geometry is None or not geometry.strip():
        raise BadRequestError("Geometria não informada.")

    normalized_wkt = _normalize_wkt(geometry)

    try:
        geom = wkt.loads(normalized_wkt)
    except Exception as exc:
        raise BadRequestError(
            "Geometria inválida: não foi possível interpretar o WKT informado."
        ) from exc

    if geom.is_empty:
        raise BadRequestError("Geometria inválida: a geometria informada está vazia.")

    if geom.geom_type not in ALLOWED_GEOMETRY_TYPES:
        raise BadRequestError(
            "Geometria inválida: apenas POLYGON e MULTIPOLYGON em EPSG:4326 são suportados."
        )

    if not geom.is_valid:
        raise BadRequestError(f"Geometria inválida: {explain_validity(geom)}")

    _validate_wgs84_coordinates(geom)


def _normalize_wkt(geometry: str) -> str:
    wkt_value = geometry.strip()
    if not wkt_value.upper().startswith("SRID="):
        return wkt_value

    srid_part, _, remainder = wkt_value.partition(";")
    if not remainder.strip():
        raise BadRequestError(
            "Geometria inválida: prefixo SRID informado sem geometria WKT."
        )

    srid = srid_part.split("=", 1)[1].strip()
    if srid != EPSG_4326:
        raise BadRequestError(
            "Geometria inválida: apenas geometrias em EPSG:4326 são suportadas."
        )

    return remainder.strip()


def _validate_wgs84_coordinates(geom: BaseGeometry) -> None:
    for longitude, latitude in _iter_coordinates(geom):
        if longitude < MIN_LONGITUDE or longitude > MAX_LONGITUDE:
            raise BadRequestError(
                "Geometria inválida: coordenadas fora do intervalo EPSG:4326 "
                "(longitude entre -180 e 180, latitude entre -90 e 90)."
            )
        if latitude < MIN_LATITUDE or latitude > MAX_LATITUDE:
            raise BadRequestError(
                "Geometria inválida: coordenadas fora do intervalo EPSG:4326 "
                "(longitude entre -180 e 180, latitude entre -90 e 90)."
            )


def _iter_coordinates(geom: BaseGeometry) -> Iterator[tuple[float, float]]:
    if isinstance(geom, Polygon):
        yield from geom.exterior.coords
        for interior in geom.interiors:
            yield from interior.coords
        return

    if isinstance(geom, MultiPolygon):
        for polygon in geom.geoms:
            yield from _iter_coordinates(polygon)
