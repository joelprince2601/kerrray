"""Phase 0 CLI tests: banner and help, ``--log-level``, ``version`` and ``info`` commands."""

from __future__ import annotations

import logging
import os
import platform
from importlib import metadata

import pytest
from rich.logging import RichHandler
from typer.testing import CliRunner

from kerrray import __version__
from kerrray.cli import app
from kerrray.reporting.console import BANNER_SUBTITLE, BANNER_TITLE
from kerrray.utils.logging import ROOT_LOGGER_NAME
from kerrray.utils.manifest import TRACKED_PACKAGES

runner = CliRunner()


def _rich_handlers() -> list[RichHandler]:
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    return [handler for handler in logger.handlers if isinstance(handler, RichHandler)]


def test_help_shows_banner_and_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert BANNER_TITLE in result.output
    assert BANNER_SUBTITLE in result.output
    assert "Usage" in result.output
    assert "--log-level" in result.output
    for command in ("version", "info"):
        assert command in result.output


def test_short_help_flag() -> None:
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0, result.output
    assert BANNER_TITLE in result.output
    assert "Usage" in result.output


def test_no_arguments_shows_banner_and_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert BANNER_TITLE in result.output
    assert "Usage" in result.output


@pytest.mark.parametrize("args", [["info", "--help"], ["info", "-h"], ["version", "-h"]])
def test_subcommand_help_works(args: list[str]) -> None:
    """Both help spellings work on subcommands, not only at the top level."""
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output
    assert args[0] in result.output


def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert BANNER_TITLE in result.output


def test_info_command_reports_real_environment() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert platform.python_version() in result.output
    assert platform.machine() in result.output
    assert str(os.cpu_count()) in result.output
    assert "Git commit" in result.output
    assert "Git dirty" in result.output
    for name in TRACKED_PACKAGES:
        assert name in result.output
        assert metadata.version(name) in result.output


def test_default_log_level_is_info() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.INFO
    assert len(_rich_handlers()) == 1


def test_log_level_option_configures_one_rich_handler_on_stderr() -> None:
    result = runner.invoke(app, ["--log-level", "DEBUG", "version"])
    assert result.exit_code == 0, result.output
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers = _rich_handlers()
    assert len(handlers) == 1
    assert handlers[0].console.stderr is True
    assert logger.level == logging.DEBUG
    assert logger.propagate is False


def test_log_level_is_case_insensitive() -> None:
    result = runner.invoke(app, ["--log-level", "warning", "version"])
    assert result.exit_code == 0, result.output
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.WARNING


def test_invalid_log_level_fails_cleanly() -> None:
    result = runner.invoke(app, ["--log-level", "LOUD", "version"])
    assert result.exit_code == 2, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "--log-level" in result.output
    assert "LOUD" in result.output
    assert "DEBUG" in result.output  # the message lists the accepted names


@pytest.mark.parametrize("args", [["--help"], [], ["version"], ["info"]])
def test_output_survives_ascii_only_console(args: list[str]) -> None:
    """All output goes through Rich, so an ASCII-only console must not crash."""
    result = CliRunner(charset="ascii").invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert BANNER_TITLE in result.output


def test_unknown_command_fails_cleanly() -> None:
    result = runner.invoke(app, ["no-such-command"])
    assert result.exit_code != 0
