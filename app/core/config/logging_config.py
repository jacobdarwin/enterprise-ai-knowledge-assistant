"""
Structured logging configuration.

Emits JSON logs in production (machine-parseable, ships well to log
aggregators) and human-readable colored logs in development.
Call `configure_logging()` once at app startup (see app/main.py).
"""

import logging
import sys

import structlog

from app.core.config.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    is_dev = settings.app_env == "development"

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_dev:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Convenience wrapper so call sites just do `log = get_logger(__name__)`."""
    return structlog.get_logger(name)
