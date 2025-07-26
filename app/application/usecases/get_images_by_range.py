from typing import List

from app.core.utils.result import AppError, Result
from datetime import datetime
from pydantic import BaseModel

class GetImagesByRangeRequest(BaseModel):
    dt_start: datetime
    dt_end: datetime
    geom: str
    
class GetImagesByRangeResponse(BaseModel):
    day: datetime

class GetImagesByRangeUseCase:

    async def execute(self, request: GetImagesByRangeRequest) -> Result[List[GetImagesByRangeResponse], AppError]:
        # Mocked response data
        response = [
            GetImagesByRangeResponse(day=request.dt_start),
            GetImagesByRangeResponse(day=request.dt_end)
        ]
        return Result.Ok(response)