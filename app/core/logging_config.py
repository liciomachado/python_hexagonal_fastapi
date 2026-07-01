import logging
import sys

from app.core.config import Config
from app.core.correlation_context import get_context_correlation_id


class CorrelationIdLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_context_correlation_id() or "-"
        return True


def setup_logging() -> None:
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)

    log_format = "%(asctime)s | %(levelname)s | %(name)s | [%(correlation_id)s] | %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    handler.addFilter(CorrelationIdLogFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    logging.getLogger("app").setLevel(level)
    logging.getLogger("httpx").setLevel(level)
    logging.getLogger("httpcore").setLevel(level)

    if level > logging.DEBUG:
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Informativo a cada rasterio.open(); não indica erro nem impacto em URLs HTTPS/Azure
    logging.getLogger("rasterio.session").setLevel(logging.WARNING)
