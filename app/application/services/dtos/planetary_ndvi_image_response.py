from datetime import date


class PlanetaryNdviImageResponse:
    def __init__(
        self,
        day: date,
        cloud_percentual: float,
        image_url: str | None,
        ndvi_mean: float | None,
        ndvi_min: float | None,
        ndvi_max: float | None,
        sat_image_id: str,
        valid_pixels: int | None = None,
        total_pixels: int | None = None,
        valid_percentage: float | None = None,
        quality: str | None = None,
    ):
        self.day = day
        self.cloud_percentual = cloud_percentual
        self.image_url = image_url
        self.ndvi_mean = ndvi_mean
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.sat_image_id = sat_image_id
        self.valid_pixels = valid_pixels
        self.total_pixels = total_pixels
        self.valid_percentage = valid_percentage
        self.quality = quality
