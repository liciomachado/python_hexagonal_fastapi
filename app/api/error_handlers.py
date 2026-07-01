import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.middleware.correlation_id import CORRELATION_ID_HEADER, get_correlation_id
from app.api.schemas.error_response import InternalServerErrorResponse
from app.core.utils.result import (
    AppError,
    BadRequestError,
    ConflictError,
    ForbiddenError,
    InternalServerError,
    NotFoundError,
    UnauthorizedError,
)

logger = logging.getLogger("app.api.errors")

INTERNAL_SERVER_ERROR_MESSAGE = (
    "Ocorreu um erro inesperado, contate o suporte informando o TraceId da requisição"
)

APP_ERROR_STATUS_MAP: dict[type[AppError], int] = {
    BadRequestError: 400,
    NotFoundError: 404,
    UnauthorizedError: 401,
    ForbiddenError: 403,
    ConflictError: 409,
    InternalServerError: 500,
}


def _status_for_app_error(error: AppError) -> int:
    for error_type, status_code in APP_ERROR_STATUS_MAP.items():
        if isinstance(error, error_type):
            return status_code
    return 500


def _with_correlation_header(response: JSONResponse, request: Request) -> JSONResponse:
    response.headers[CORRELATION_ID_HEADER] = get_correlation_id(request)
    return response


def _internal_server_error_response(request: Request) -> JSONResponse:
    trace_id = get_correlation_id(request)
    content = InternalServerErrorResponse(
        message=INTERNAL_SERVER_ERROR_MESSAGE,
        traceId=trace_id,
    ).model_dump()
    return _with_correlation_header(JSONResponse(status_code=500, content=content), request)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    status_code = _status_for_app_error(exc)
    if status_code == 500:
        logger.error("Internal server error: %s", exc.message)
        return _internal_server_error_response(request)

    return _with_correlation_header(
        JSONResponse(status_code=status_code, content={"detail": exc.message}),
        request,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code == 500:
        logger.error("HTTP 500: %s", exc.detail)
        return _internal_server_error_response(request)

    return _with_correlation_header(
        JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}),
        request,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", exc_info=exc)
    return _internal_server_error_response(request)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
