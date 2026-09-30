"""Shared pytest fixtures for KerrRay."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from kerrray.utils.logging import ROOT_LOGGER_NAME


@pytest.fixture(autouse=True)
def restore_kerrray_logger() -> Iterator[None]:
    """Undo any logging configuration a test performs.

    ``configure_logging`` (called by the CLI's ``--log-level`` option) attaches
    a handler and sets ``propagate = False`` on the ``kerrray`` logger. Tests
    that rely on ``caplog`` need propagation to pytest's root handler, so every
    test must start from the logger state it found and hand it back unchanged.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    yield
    for handler in list(logger.handlers):
        if handler not in saved_handlers:
            logger.removeHandler(handler)
            handler.close()
    for handler in saved_handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate
