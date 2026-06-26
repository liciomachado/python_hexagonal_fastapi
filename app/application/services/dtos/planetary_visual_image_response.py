from datetime import date


class PlanetaryImageVisualResponse:
    def __init__(self, day: date, cloud_percentual: float, image_url: str):
        self.day = day
        self.cloud_percentual = cloud_percentual
        self.image_url = image_url
