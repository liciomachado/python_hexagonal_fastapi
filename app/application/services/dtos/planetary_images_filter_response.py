from datetime import datetime
from typing import Any


class PlanetaryImageFilterResponse:
    def __init__(
        self,
        id: str,
        datetime: datetime,
        cloud_cover: float | None,
        geometry: dict[str, Any],
        assets: dict[str, Any],
        cloud_cover_geometry: float | None = None,
    ):
        self.id = id
        self.datetime = datetime
        self.cloud_cover = cloud_cover
        self.geometry = geometry
        self.assets = assets
        self.cloud_cover_geometry = cloud_cover_geometry