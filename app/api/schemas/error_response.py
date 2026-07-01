from pydantic import BaseModel


class InternalServerErrorResponse(BaseModel):
    message: str
    traceId: str
