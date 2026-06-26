import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("app.performance")


@dataclass
class PerformanceMetrics:
    """Coleta tempos por etapa do pipeline de imagem para benchmark e diagnóstico."""

    context: str = ""
    spans: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.spans[name] = elapsed_ms
            logger.info("PERF [%s] %s=%.2fms", self.context, name, elapsed_ms)

    def log_summary(self) -> None:
        total = sum(self.spans.values())
        logger.info(
            "PERF [%s] summary total=%.2fms spans=%s",
            self.context,
            total,
            self.spans,
        )
