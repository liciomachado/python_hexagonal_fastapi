from datetime import datetime


class PlanetaryImageVisualResponse:
    def __init__(self, day: datetime, cloud_percentual: float, base64image: str):
        self.day = day
        self.cloud_percentual = cloud_percentual
        self.base64image = base64image