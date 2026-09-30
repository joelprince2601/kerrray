"""Logging tests: Rich handler configuration, propagation and logger namespacing."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from io import StringIO

import pytest
from rich.console import Console
from rich.logging import RichHandler

from kerrray.utils.logging import (
    LOG_LEVEL_NAMES,
    ROOT_LOGGER_NAME,
    configure_logging,
    get_logger,
    resolve_level,
)


@pytest.fixture
def buffer_console() -> Iterator[tuple[Console, StringIO]]:
    """A console writing to memory; handlers added during a test are removed afterwards.

    ``configure_logging`` also disables propagation, which the ``caplog``-based
    tests elsewhere rely on, so propagation is restored here (and by the
    autouse fixture in ``conftest.py``).
    """
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120)
    yield console, stream
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        if isinstance(handler, RichHandler):
            root.removeHandler(handler)
    root.setLevel(logging.NOTSET)
    root.propagate = True


def test_configure_logging_attaches_exactly_one_rich_handler(
    buffer_console: tuple[Console, StringIO],
) -> None:
    console, _ = buffer_console
    first = configure_logging("INFO", console=console)
    second = configure_logging(logging.DEBUG, console=console)
    assert first is second
    assert first.name == ROOT_LOGGER_NAME
    assert first.level == logging.DEBUG
    rich_handlers = [h for h in first.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) == 1


def test_level_names_are_case_insensitive(buffer_console: tuple[Console, StringIO]) -> None:
    console, _ = buffer_console
    assert configure_logging("warning", console=console).level == logging.WARNING


@pytest.mark.parametrize("name", LOG_LEVEL_NAMES)
def test_advertised_level_names_resolve(name: str) -> None:
    assert resolve_level(name) == logging.getLevelNamesMapping()[name]
    assert resolve_level(name.lower()) == resolve_level(name)


def test_numeric_levels_pass_through() -> None:
    assert resolve_level(logging.ERROR) == logging.ERROR


def test_unknown_level_name_rejected(buffer_console: tuple[Console, StringIO]) -> None:
    console, _ = buffer_console
    with pytest.raises(ValueError, match="LOUD"):
        configure_logging("LOUD", console=console)


def test_messages_reach_the_console(buffer_console: tuple[Console, StringIO]) -> None:
    console, stream = buffer_console
    configure_logging("INFO", console=console)
    log = get_logger("test_logging")
    log.info("photon-launch-check")
    log.debug("below-threshold")
    output = stream.getvalue()
    assert "photon-launch-check" in output
    assert "INFO" in output
    assert "below-threshold" not in output


def test_records_are_not_duplicated_by_a_root_handler(
    buffer_console: tuple[Console, StringIO],
) -> None:
    """With propagate=False a root handler (basicConfig, notebooks) prints nothing twice."""
    console, stream = buffer_console
    root_stream = StringIO()
    root_handler = logging.StreamHandler(root_stream)
    logging.getLogger().addHandler(root_handler)
    try:
        logger = configure_logging("INFO", console=console)
        assert logger.propagate is False
        get_logger("test_logging").warning("dup-check")
    finally:
        logging.getLogger().removeHandler(root_handler)
    assert stream.getvalue().count("dup-check") == 1
    assert "dup-check" not in root_stream.getvalue()


def test_get_logger_namespacing() -> None:
    assert get_logger("cli").name == "kerrray.cli"
    assert get_logger("kerrray.utils.config").name == "kerrray.utils.config"
    assert get_logger(ROOT_LOGGER_NAME).name == ROOT_LOGGER_NAME
    assert get_logger("cli").parent is logging.getLogger(ROOT_LOGGER_NAME)
