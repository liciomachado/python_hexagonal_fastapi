import logging
import sys

from app.core.config import Config


def setup_logging() -> None:
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )

    logging.getLogger("app").setLevel(level)
    logging.getLogger("httpx").setLevel(level)
    logging.getLogger("httpcore").setLevel(level)

    if level > logging.DEBUG:
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Informativo a cada rasterio.open(); não indica erro nem impacto em URLs HTTPS/Azure
    logging.getLogger("rasterio.session").setLevel(logging.WARNING)
