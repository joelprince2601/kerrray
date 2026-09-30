"""Console helper tests: value formatting and banner/section rendering."""

from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np
import pytest
from rich.console import Console

from kerrray.reporting.console import (
    ASCII_RULE,
    BANNER_SUBTITLE,
    BANNER_TITLE,
    SECTION_WIDTH,
    UNICODE_RULE,
    format_value,
    render_banner,
    render_section,
)


class _AsciiBuffer(StringIO):
    """In-memory stream that reports an ASCII encoding, like a legacy console."""

    encoding = "ascii"


class _Utf8Buffer(StringIO):
    encoding = "utf-8"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (262144, "262,144"),
        (np.int64(262144), "262,144"),
        (np.int32(7), "7"),
        (1 / 3, "0.333333"),
        (np.float32(1 / 3), "0.333333"),
        (np.float64(1 / 3), "0.333333"),
        (1e-9, "1e-09"),
        (True, "True"),
        (np.bool_(False), "False"),
        (None, "None"),
        ("Kerr", "Kerr"),
    ],
)
def test_format_value(value: Any, expected: str) -> None:
    assert format_value(value) == expected


def test_format_value_precision() -> None:
    assert format_value(np.float32(1 / 3), precision=3) == "0.333"
    assert format_value(2 / 3, precision=10) == "0.6666666667"


def test_ascii_console_gets_ascii_rule_and_pure_ascii_output() -> None:
    stream = _AsciiBuffer()
    console = Console(file=stream, force_terminal=False, width=100)
    assert console.options.ascii_only
    render_banner(console)
    render_section(console, "Ray Trace", [("Rays", np.int64(262144)), ("Integrator", "RK45")])
    output = stream.getvalue()
    output.encode("ascii")  # must not raise
    assert BANNER_TITLE in output and BANNER_SUBTITLE in output
    assert "Ray Trace" in output and "262,144" in output and "RK45" in output
    assert ASCII_RULE * SECTION_WIDTH in output
    assert UNICODE_RULE not in output


def test_utf8_console_gets_unicode_rule() -> None:
    stream = _Utf8Buffer()
    console = Console(file=stream, force_terminal=False, width=100)
    assert not console.options.ascii_only
    render_section(console, "Spacetime", [("Metric", "Kerr"), ("Spin", 0.94)])
    output = stream.getvalue()
    assert UNICODE_RULE * SECTION_WIDTH in output
    assert "Metric" in output and "Kerr" in output and "0.94" in output
