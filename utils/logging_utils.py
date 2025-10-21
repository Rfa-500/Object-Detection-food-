import logging
from typing import Optional


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO, filename: Optional[str] = None) -> None:
    """Configure global logging for the application."""
    handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handlers.append(stream_handler)

    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance with the project configuration."""
    return logging.getLogger(name)
