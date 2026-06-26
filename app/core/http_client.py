import logging

import httpx

logger = logging.getLogger("app.http")

_shared_client: httpx.AsyncClient | None = None


def _provider_label_from_url(url: str) -> str:
    if "planetarycomputer.microsoft.com" in url:
        return "planetary"
    if "earth-search.aws.element84.com" in url:
        return "earth_search"
    return "external"


async def _log_request(request: httpx.Request) -> None:
    label = _provider_label_from_url(str(request.url))
    logger.info("OUTGOING [%s] %s %s", label, request.method, request.url)


async def _log_response(response: httpx.Response) -> None:
    label = _provider_label_from_url(str(response.request.url))
    logger.info(
        "OUTGOING [%s] %s %s -> %s",
        label,
        response.request.method,
        response.request.url,
        response.status_code,
    )


def create_async_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        event_hooks={
            "request": [_log_request],
            "response": [_log_response],
        },
    )


async def get_shared_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = create_async_http_client(timeout=timeout)
    return _shared_client


async def close_shared_http_client() -> None:
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None
