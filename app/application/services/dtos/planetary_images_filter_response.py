from datetime import datetime
from typing import Any


class PlanetaryImageFilterResponse:
    def __init__(
        self,
        id: str,
        datetime: datetime,
        cloud_cover: float | None,
        geometry: dict[str, Any],
        assets: dict[str, Any]
    ):
        self.id = id
        self.datetime = datetime
        self.cloud_cover = cloud_cover
        self.geometry = geometry
        self.assets = assets