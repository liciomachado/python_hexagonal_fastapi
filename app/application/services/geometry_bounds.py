from shapely.geometry.base import BaseGeometry

CLOUD_COVER_GEOMETRY_BUFFER = 0.050


def compute_cloud_cover_geom_bounds(geom: BaseGeometry) -> tuple[float, float, float, float]:
    """Bounds used exclusively for cloud-mask raster reads over the user geometry."""
    minx, miny, maxx, maxy = geom.bounds
    buffer = CLOUD_COVER_GEOMETRY_BUFFER
    return (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
