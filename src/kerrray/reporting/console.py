"""Rich console helpers shared by every KerrRay CLI command.

The output style follows PROJECT.md section 29: a banner panel followed by
titled key-value sections, each underlined by a horizontal rule. Everything
KerrRay prints goes through a :class:`rich.console.Console` so that boxes
degrade to ASCII on consoles that cannot encode Unicode (Rich substitutes
ASCII boxes when the output encoding is not UTF-8, and the rule character is
chosen here the same way). All strings emitted by these helpers are pure
ASCII apart from that rule. No console is kept at module level; callers
create one with :func:`get_console` and pass it around.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any, Final

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BANNER_TITLE: Final[str] = "KERRRAY"
BANNER_SUBTITLE: Final[str] = "Relativistic Photon Dynamics Engine"
SECTION_WIDTH: Final[int] = 46
"""Width of the section rule and the banner interior (PROJECT.md section 29)."""
LABEL_WIDTH: Final[int] = 16
"""Minimum width of the key column in a section."""
UNICODE_RULE: Final[str] = chr(0x2500)  # U+2500 BOX DRAWINGS LIGHT HORIZONTAL
ASCII_RULE: Final[str] = "-"

Row = tuple[str, Any]
"""A ``(label, value)`` pair rendered as one section line."""


def get_console(*, stderr: bool = False, **options: Any) -> Console:
    """Create a Rich console (stdout by default). Extra options go to ``Console``."""
    return Console(stderr=stderr, **options)


def banner() -> Panel:
    """The KERRRAY banner panel: title and subtitle centred in a rounded box."""
    body = Text(justify="center")
    body.append(BANNER_TITLE, style="bold")
    body.append("\n")
    body.append(BANNER_SUBTITLE)
    return Panel(body, box=box.ROUNDED, width=SECTION_WIDTH + 2, expand=False)


def render_banner(console: Console) -> None:
    """Print the banner to ``console``."""
    console.print(banner())


def format_value(value: Any, *, precision: int = 6) -> str:
    """Format a value for a section line.

    Integers get thousands separators, real numbers ``precision`` significant
    digits, everything else ``str(value)``. Booleans and ``None`` are printed
    literally. The checks use the :mod:`numbers` ABCs, so NumPy scalars
    (``np.int64``, ``np.float32``, ...) format like the Python ``int`` and
    ``float`` they wrap; ``np.bool_`` is not registered as a number and prints
    as ``True``/``False``.
    """
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real):
        return f"{float(value):.{precision}g}"
    return str(value)


def section(
    title: str, rows: Sequence[Row], *, ascii_only: bool = False, precision: int = 6
) -> RenderableType:
    """Build a section: bold title, rule, then aligned ``label  value`` lines.

    Args:
        title: Section heading, e.g. ``"Spacetime"``.
        rows: ``(label, value)`` pairs; values pass through :func:`format_value`.
        ascii_only: Use ``-`` instead of the Unicode box-drawing rule.
        precision: Significant digits for float values.
    """
    rule = (ASCII_RULE if ascii_only else UNICODE_RULE) * SECTION_WIDTH
    grid = Table.grid(padding=(0, 2))
    grid.add_column(min_width=LABEL_WIDTH)
    grid.add_column()
    for label, value in rows:
        grid.add_row(str(label), format_value(value, precision=precision))
    return Group(Text(title, style="bold"), Text(rule, style="dim"), grid)


def render_section(
    console: Console, title: str, rows: Sequence[Row], *, precision: int = 6
) -> None:
    """Print a section (see :func:`section`), choosing the rule style from ``console``."""
    ascii_only = console.options.ascii_only
    console.print(section(title, rows, ascii_only=ascii_only, precision=precision))
    console.print()
