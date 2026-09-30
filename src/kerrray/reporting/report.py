"""Scientific report writer: ``report.md`` and ``metadata.json`` (PROJECT.md section 36).

:class:`ReportSections` holds the text of the nine mandatory sections in the
order PROJECT.md section 36 lists them; :func:`write_report` writes them as
numbered Markdown headings, embeds tables and figures under *Results* (figures
by path relative to the report directory) and records the environment the
report was generated in (git commit, packages, hardware) in ``metadata.json``
through :mod:`kerrray.utils.manifest`. The text of each section is supplied by
the experiment driver from computed values; nothing here invents a number.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

from kerrray.utils.manifest import (
    EnvironmentInfo,
    collect_environment,
    sanitise_for_json,
    utc_timestamp,
)

__all__ = [
    "METADATA_FILENAME",
    "REPORT_FILENAME",
    "REPORT_HEADINGS",
    "ReportSections",
    "write_report",
]

REPORT_FILENAME: Final[str] = "report.md"
METADATA_FILENAME: Final[str] = "metadata.json"
FIGURES_SECTION: Final[str] = "results"
"""Section under which tables and figures are embedded."""

REPORT_HEADINGS: Final[tuple[tuple[str, str], ...]] = (
    ("objective", "Objective"),
    ("mathematical_model", "Mathematical model"),
    ("numerical_method", "Numerical method"),
    ("parameters", "Parameters"),
    ("results", "Results"),
    ("error_analysis", "Error analysis"),
    ("interpretation", "Interpretation"),
    ("limitations", "Limitations"),
    ("reproducibility", "Reproducibility information"),
)
"""``(field name, heading)`` pairs in the PROJECT.md section 36 order."""

MISSING_TEXT: Final[str] = "_Not provided._"


@dataclass(frozen=True)
class ReportSections:
    """Markdown text of the nine report sections (PROJECT.md section 36).

    ``mathematical_model`` and ``numerical_method`` are the fields that
    docs/architecture.md section 5 abbreviates as ``model`` and ``method``.
    """

    objective: str = ""
    mathematical_model: str = ""
    numerical_method: str = ""
    parameters: str = ""
    results: str = ""
    error_analysis: str = ""
    interpretation: str = ""
    limitations: str = ""
    reproducibility: str = ""

    def ordered(self) -> list[tuple[str, str, str]]:
        """``(field, heading, text)`` triples in report order."""
        return [(name, heading, getattr(self, name)) for name, heading in REPORT_HEADINGS]


def _check_sections_match_headings() -> None:
    names = [spec.name for spec in fields(ReportSections)]
    if names != [name for name, _ in REPORT_HEADINGS]:
        raise RuntimeError("ReportSections fields and REPORT_HEADINGS are out of sync")


_check_sections_match_headings()


def _relative(path: Path, report_dir: Path) -> str:
    rel = os.path.relpath(Path(path).resolve(), report_dir.resolve())
    return rel.replace(os.sep, "/")


def write_report(
    report_dir: Path | str,
    sections: ReportSections,
    figures: Sequence[Path | str] = (),
    tables: Sequence[str] = (),
    *,
    title: str = "KerrRay experiment report",
    environment: EnvironmentInfo | None = None,
) -> Path:
    """Write ``report.md`` and ``metadata.json`` into ``report_dir`` and return the report path.

    Args:
        report_dir: ``reports/<run_id>``; created if needed.
        sections: Section texts (Markdown). Empty sections are written as
            ``_Not provided._`` so the nine headings are always present.
        figures: Figure files (normally inside ``report_dir``), embedded under
            *Results* as ``![stem](relative/path)``.
        tables: Markdown table strings (see
            :func:`kerrray.reporting.tables.markdown_table`), embedded under
            *Results* before the figures.
        title: Top-level heading of the report.
        environment: Pre-collected environment snapshot; collected when ``None``.
    """
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    env = environment if environment is not None else collect_environment()
    timestamp = utc_timestamp()
    figure_paths = [_relative(Path(fig), directory) for fig in figures]

    lines: list[str] = [f"# {title}", ""]
    for number, (name, heading, text) in enumerate(sections.ordered(), start=1):
        lines.append(f"## {number}. {heading}")
        lines.append("")
        lines.append(text.strip() if text.strip() else MISSING_TEXT)
        lines.append("")
        if name == FIGURES_SECTION:
            for table in tables:
                lines.append(table.rstrip())
                lines.append("")
            for fig, rel in zip(figures, figure_paths):
                lines.append(f"![{Path(fig).stem}]({rel})")
                lines.append("")
        if name == "reproducibility":
            lines.append(f"Generated {timestamp}; git commit `{env.git_commit}`"
                         f"{' (dirty working tree)' if env.git_dirty else ''}; "
                         f"environment in `{METADATA_FILENAME}`.")
            lines.append("")

    report_path = directory / REPORT_FILENAME
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    metadata = {
        "title": title,
        "timestamp": timestamp,
        "git_commit": env.git_commit,
        "git_dirty": env.git_dirty,
        "report": REPORT_FILENAME,
        "sections": [heading for _, heading in REPORT_HEADINGS],
        "figures": figure_paths,
        "n_tables": len(tables),
        "environment": env.as_dict(),
    }
    text = json.dumps(sanitise_for_json(metadata), indent=2, allow_nan=False)
    (directory / METADATA_FILENAME).write_text(text + "\n", encoding="utf-8", newline="\n")
    return report_path
