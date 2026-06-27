from dataclasses import dataclass

from app.application.services.stac.satellite_collection import SatelliteCollection


@dataclass(frozen=True)
class SensorProfile:
    collection: SatelliteCollection
    rgb_bands: tuple[str, str, str]
    ndvi_bands: tuple[str, str]
    ndmi_bands: tuple[str, str]
    ndmi_reference_band: str
    all_bands: tuple[str, ...]
    cloud_mask_band: str
    rgb_clip_max: float
    blob_prefix: str
    apply_reflectance_scale: bool


SENSOR_PROFILES: dict[SatelliteCollection, SensorProfile] = {
    SatelliteCollection.SENTINEL2_L2A: SensorProfile(
        collection=SatelliteCollection.SENTINEL2_L2A,
        rgb_bands=("B04", "B03", "B02"),
        ndvi_bands=("B08", "B04"),
        ndmi_bands=("B08", "B11"),
        ndmi_reference_band="B11",
        all_bands=("B02", "B03", "B04", "B08", "B11"),
        cloud_mask_band="SCL",
        rgb_clip_max=3000.0,
        blob_prefix="sentinel",
        apply_reflectance_scale=False,
    ),
    SatelliteCollection.LANDSAT_C2_L2: SensorProfile(
        collection=SatelliteCollection.LANDSAT_C2_L2,
        rgb_bands=("B04", "B03", "B02"),
        ndvi_bands=("B08", "B04"),
        ndmi_bands=("B08", "B11"),
        ndmi_reference_band="B11",
        all_bands=("B02", "B03", "B04", "B08", "B11"),
        cloud_mask_band="QA_PIXEL",
        rgb_clip_max=10000.0,
        blob_prefix="landsat",
        apply_reflectance_scale=True,
    ),
}


def get_sensor_profile(collection: SatelliteCollection) -> SensorProfile:
    return SENSOR_PROFILES[collection]


def normalize_band_values(values, profile: SensorProfile):
    import numpy as np

    arr = values.astype(np.float32)
    if profile.apply_reflectance_scale:
        arr = np.where(arr > 0, arr * 0.0000275 - 0.2, 0.0)
        arr = np.clip(arr, 0.0, 1.0)
        return arr * profile.rgb_clip_max
    return arr
