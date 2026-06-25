import logging

import httpx

logger = logging.getLogger("app.http")


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
        event_hooks={
            "request": [_log_request],
            "response": [_log_response],
        },
    )
