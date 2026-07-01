from contextvars import ContextVar, Token

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(correlation_id: str) -> Token:
    return _correlation_id.set(correlation_id)


def get_context_correlation_id() -> str | None:
    return _correlation_id.get()


def reset_correlation_id(token: Token) -> None:
    _correlation_id.reset(token)
