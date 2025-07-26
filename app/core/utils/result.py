from typing import Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")

class Result(Generic[T, E]):
    def __init__(self, value: Union[T, None] = None, error: Union[E, None] = None):
        self._value = value
        self._error = error

    @staticmethod
    def Ok(value: T) -> "Result[T, E]":
        return Result(value=value)

    @staticmethod
    def Err(error: E) -> "Result[T, E]":
        return Result(error=error)

    def is_ok(self) -> bool:
        return self._error is None

    def is_err(self) -> bool:
        return self._error is not None

    def value(self) -> T:
        if self.is_err():
            raise Exception(f"Tried to unwrap an error: {self._error}")
        return self._value

    def error(self) -> E:
        if self.is_ok():
            raise Exception(f"Tried to unwrap a value: {self._value}")
        return self._error

class AppError(Exception):
    """Erro base para a aplicação"""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class BadRequestError(AppError):
    """Erro 400"""

class NotFoundError(AppError):
    """Erro 404"""

class UnauthorizedError(AppError):
    """Erro 401"""

class ForbiddenError(AppError):
    """Erro 403"""

class ConflictError(AppError):
    """Erro 409"""
