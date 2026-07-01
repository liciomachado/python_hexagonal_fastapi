import logging
import time

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.middleware.correlation_id import get_correlation_id

logger = logging.getLogger("app.api")


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start = time.perf_counter()
        correlation_id = get_correlation_id(request)
        logger.info(
            "INCOMING %s %s [%s]",
            request.method,
            request.url.path,
            correlation_id,
        )

        status_code = 500

        async def send_with_logging(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_with_logging)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "INCOMING %s %s -> %s (%.1fms) [%s]",
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
            correlation_id,
        )
