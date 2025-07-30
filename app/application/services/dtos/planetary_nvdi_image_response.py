from datetime import date


class PlanetaryNdviImageResponse:
    def __init__(self, day: date, cloud_percentual: float, base64image: str):
        self.day = day
        self.cloud_percentual = cloud_percentual
        self.base64image = base64image