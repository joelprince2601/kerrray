"""Logging setup for KerrRay built on :class:`rich.logging.RichHandler`.

All KerrRay loggers live under the ``kerrray`` namespace.
:func:`configure_logging` attaches exactly one ``RichHandler`` to the
``kerrray`` logger (replacing any it attached earlier, so repeated calls never
duplicate output), stops propagation to the root logger (so a root handler
installed by a script, notebook or another library does not print every
KerrRay record a second time) and :func:`get_logger` returns namespaced child
loggers. Log lines go to standard error by default so they never mix with CLI
results on standard output. The CLI calls :func:`configure_logging` from its
``--log-level`` option. No state is kept in this module beyond what the
standard-library logging registry holds.
"""

from __future__ import annotations

import logging
from typing import Final

from rich.console import Console
from rich.logging import RichHandler

ROOT_LOGGER_NAME: Final[str] = "kerrray"
"""Name of the logger every KerrRay logger descends from."""

LOG_LEVEL_NAMES: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
"""Level names advertised by the CLI. Any name known to :mod:`logging` is accepted."""

LOG_FORMAT: Final[str] = "%(message)s"
DATE_FORMAT: Final[str] = "[%Y-%m-%d %H:%M:%S]"


def resolve_level(level: int | str) -> int:
    """Turn a level name (case insensitive) or number into a numeric logging level.

    Raises:
        ValueError: If ``level`` is a name :mod:`logging` does not know.
    """
    if isinstance(level, str):
        numeric = logging.getLevelNamesMapping().get(level.strip().upper())
        if numeric is None:
            raise ValueError(
                f"unknown log level {level!r}; choose one of {', '.join(LOG_LEVEL_NAMES)}"
            )
        return numeric
    return int(level)


def configure_logging(
    level: int | str = logging.INFO, *, console: Console | None = None
) -> logging.Logger:
    """Configure the ``kerrray`` logger with a single Rich handler.

    Args:
        level: A numeric level or a level name such as ``"DEBUG"`` (case
            insensitive), applied to the ``kerrray`` logger.
        console: Rich console the handler writes to. When ``None`` a console
            bound to standard error is created.

    Returns:
        The configured ``kerrray`` logger, with ``propagate`` set to ``False``
        so records are emitted exactly once, by the Rich handler.

    Raises:
        ValueError: If ``level`` is an unknown level name.
    """
    numeric_level = resolve_level(level)
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(numeric_level)
    for existing in list(logger.handlers):
        if isinstance(existing, RichHandler):
            logger.removeHandler(existing)
            existing.close()
    handler = RichHandler(
        console=console if console is not None else Console(stderr=True),
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``kerrray`` namespace.

    Args:
        name: Either a module ``__name__`` that already starts with
            ``kerrray`` (returned as is) or a short name that is prefixed with
            ``kerrray.``.

    Returns:
        The requested :class:`logging.Logger`.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
