from enum import Enum


class SatelliteCollection(str, Enum):
    SENTINEL2_L2A = "sentinel-2-l2a"
    LANDSAT_C2_L2 = "landsat-c2-l2"


DEFAULT_SATELLITE_COLLECTION = SatelliteCollection.SENTINEL2_L2A
