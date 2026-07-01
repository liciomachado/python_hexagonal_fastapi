from datetime import datetime

from app.application.validators.geometry_validator import validate_epsg4326_wkt_geometry
from app.core.utils.result import BadRequestError, Result


def require_valid_sentinel_geometry(geometry: str | None) -> Result[None, BadRequestError]:
    try:
        validate_epsg4326_wkt_geometry(geometry)
    except BadRequestError as error:
        return Result.Err(error)
    return Result.Ok(None)


def require_valid_date_range(
    dt_start: datetime,
    dt_end: datetime,
) -> Result[None, BadRequestError]:
    if dt_start > dt_end:
        return Result.Err(
            BadRequestError("Período inválido: dt_start não pode ser maior que dt_end.")
        )
    return Result.Ok(None)
