from datetime import datetime, timezone
import time


class CircuitBreaker:
    def __init__(self, open_seconds: int = 300):
        self._open_seconds = open_seconds
        self._opened_until: float | None = None

    def is_open(self) -> bool:
        if self._opened_until is None:
            return False
        if time.time() >= self._opened_until:
            self._opened_until = None
            return False
        return True

    def open(self) -> None:
        self._opened_until = time.time() + self._open_seconds

    def opened_until(self) -> datetime | None:
        if self._opened_until is None:
            return None
        return datetime.fromtimestamp(self._opened_until, tz=timezone.utc)

    def state(self) -> str:
        return "open" if self.is_open() else "closed"
