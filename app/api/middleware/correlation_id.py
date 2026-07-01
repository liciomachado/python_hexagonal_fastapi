from uuid import uuid4

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.correlation_context import reset_correlation_id, set_correlation_id

CORRELATION_ID_HEADER = "X-Correlation-Id"
CORRELATION_ID_HEADER_BYTES = b"x-correlation-id"


def get_correlation_id(request: Request) -> str:
    correlation_id = getattr(request.state, "correlation_id", None)
    if correlation_id:
        return correlation_id
    return request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())


class CorrelationIdMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        correlation_id = None
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == CORRELATION_ID_HEADER_BYTES:
                correlation_id = header_value.decode("latin-1")
                break

        if not correlation_id:
            correlation_id = str(uuid4())

        scope.setdefault("state", {})["correlation_id"] = correlation_id

        token = set_correlation_id(correlation_id)
        try:
            async def send_with_correlation_id(message: Message) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    if not any(
                        header_name.lower() == CORRELATION_ID_HEADER_BYTES
                        for header_name, _ in headers
                    ):
                        headers.append(
                            (CORRELATION_ID_HEADER_BYTES, correlation_id.encode("latin-1"))
                        )
                        message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_with_correlation_id)
        finally:
            reset_correlation_id(token)
