"""Rows-of-dicts to Rich tables and Markdown tables (no pandas; docs/decisions.md, D-003).

A table is a sequence of mappings; the column order is the order in which
keys are first seen (or an explicit ``columns`` list). Numbers are formatted
with :func:`kerrray.reporting.console.format_value` (integers with thousands
separators, reals with ``precision`` significant digits), so a value printed
in a report is exactly the value printed on the console.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import Any

from rich import box
from rich.table import Table

from kerrray.reporting.console import format_value

__all__ = ["column_names", "format_cell", "markdown_table", "rich_table"]

Row = Mapping[str, Any]


def column_names(rows: Sequence[Row], columns: Sequence[str] | None = None) -> list[str]:
    """Column order: ``columns`` if given, else keys in order of first appearance."""
    if columns is not None:
        return list(columns)
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(str(key), None)
    return list(seen)


def format_cell(value: Any, *, precision: int = 6) -> str:
    """Text for one cell: ``""`` for ``None``, otherwise :func:`format_value`."""
    if value is None:
        return ""
    return format_value(value, precision=precision)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _numeric_column(rows: Sequence[Row], name: str) -> bool:
    values = [row.get(name) for row in rows if row.get(name) is not None]
    return bool(values) and all(_is_number(value) for value in values)


def rich_table(
    rows: Sequence[Row],
    *,
    columns: Sequence[str] | None = None,
    title: str | None = None,
    precision: int = 6,
) -> Table:
    """Build a :class:`rich.table.Table`; numeric columns are right-aligned."""
    names = column_names(rows, columns)
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_edge=False)
    for name in names:
        table.add_column(name, justify="right" if _numeric_column(rows, name) else "left")
    for row in rows:
        table.add_row(*(format_cell(row.get(name), precision=precision) for name in names))
    return table


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(
    rows: Sequence[Row],
    *,
    columns: Sequence[str] | None = None,
    precision: int = 6,
) -> str:
    """Render rows as a GitHub-flavoured Markdown table (no trailing newline).

    Numeric columns get a right-aligned separator (``---:``). An empty column
    list yields an empty string.
    """
    names = column_names(rows, columns)
    if not names:
        return ""
    header = "| " + " | ".join(_escape_markdown(name) for name in names) + " |"
    separator = "| " + " | ".join(
        "---:" if _numeric_column(rows, name) else "---" for name in names
    ) + " |"
    lines = [header, separator]
    for row in rows:
        cells = (_escape_markdown(format_cell(row.get(name), precision=precision)) for name in names)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
