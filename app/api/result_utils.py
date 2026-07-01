from typing import TypeVar

from app.core.utils.result import AppError, InternalServerError, Result

T = TypeVar("T")


def unwrap_result(result: Result[T, AppError | str]) -> T:
    if result.is_ok():
        return result.value()

    error = result.error()
    if isinstance(error, AppError):
        raise error

    raise InternalServerError(str(error))
