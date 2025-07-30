from datetime import date


class PlanetaryNdviImageResponse:
    def __init__(self, day: date, cloud_percentual: float, base64image: str, 
                 ndvi_mean: float | None, ndvi_min: float | None, ndvi_max: float | None, sat_image_id: str):
        self.day = day
        self.cloud_percentual = cloud_percentual
        self.base64image = base64image
        self.ndvi_mean = ndvi_mean
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.sat_image_id = sat_image_id