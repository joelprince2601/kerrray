"""Schwarzschild and Kerr validation, conservation checks and convergence tests
(PROJECT.md sections 12, 13 and 41; docs/validation.md).

This package module holds the types shared by the validation drivers (it
imports none of its submodules, so they can import it freely):

* :class:`ValidationCheck`: one named check with computed values, the
  reference values it is compared with, the errors, the tolerances and the
  verdict;
* :class:`ValidationReport`: the checks of one validation run and the
  :class:`~kerrray.experiments.base.RunRecord` that recorded it;
* :func:`integrator_options_from_config` and
  :func:`termination_options_from_config`: the mapping of the configuration
  blocks to the integrator options.

Drivers: :mod:`kerrray.validation.schwarzschild`
(``run_schwarzschild_validation``, EXP-001 and EXP-002) and
:mod:`kerrray.validation.kerr` (``run_kerr_validation``). Helpers:
:mod:`kerrray.validation.conservation` (drift summaries),
:mod:`kerrray.validation.convergence` (order estimates) and
:mod:`kerrray.validation.kerr_bisection` (bracketed capture-threshold search).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kerrray.geodesics import IntegratorOptions, TerminationOptions

if TYPE_CHECKING:  # pragma: no cover
    from kerrray.experiments.base import RunRecord
    from kerrray.utils.config import KerrRayConfig

__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "integrator_options_from_config",
    "termination_options_from_config",
    "worst_error",
]


def worst_error(error: dict[str, Any]) -> float:
    """Largest finite number in an error mapping (one level of nesting; booleans ignored)."""
    values: list[float] = []
    for v in error.values():
        items = v.values() if isinstance(v, dict) else (v,)
        values.extend(float(x) for x in items if isinstance(x, (int, float)) and not isinstance(x, bool))
    finite = [x for x in values if math.isfinite(x)]
    return max(finite) if finite else math.nan


@dataclass(frozen=True)
class ValidationCheck:
    """One named check: computed values, references, errors, tolerances and verdict."""

    name: str
    description: str
    computed: dict[str, Any]
    reference: dict[str, Any]
    error: dict[str, Any]
    tolerance: dict[str, Any]
    passed: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping of the check."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of a validation run (docs/architecture.md section 5)."""

    name: str
    spin: float
    checks: tuple[ValidationCheck, ...]
    run: RunRecord | None = None

    @classmethod
    def from_checks(cls, name: str, spin: float, checks: list[ValidationCheck]) -> ValidationReport:
        """Assemble a report from a list of checks."""
        return cls(name=name, spin=spin, checks=tuple(checks))

    @property
    def computed_values(self) -> dict[str, dict[str, Any]]:
        """``{check name: computed values}``."""
        return {c.name: c.computed for c in self.checks}

    @property
    def references(self) -> dict[str, dict[str, Any]]:
        """``{check name: reference values}``."""
        return {c.name: c.reference for c in self.checks}

    @property
    def errors(self) -> dict[str, dict[str, Any]]:
        """``{check name: errors}``."""
        return {c.name: c.error for c in self.checks}

    @property
    def verdicts(self) -> dict[str, bool]:
        """``{check name: passed}``."""
        return {c.name: c.passed for c in self.checks}

    def check(self, name: str) -> ValidationCheck:
        """The check called ``name`` (``KeyError`` if absent)."""
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def all_passed(self) -> bool:
        """True when every check passed."""
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable mapping (the manifest's ``results``)."""
        return {
            "validation": self.name,
            "spin": self.spin,
            "all_passed": self.all_passed,
            "verdicts": self.verdicts,
            "checks": {c.name: c.to_dict() for c in self.checks},
        }

    def summary_rows(self) -> list[tuple[str, Any]]:
        """``(label, value)`` rows for a console section: one verdict per check."""
        rows: list[tuple[str, Any]] = [(c.name, "PASS" if c.passed else "FAIL") for c in self.checks]
        rows.append(("all checks", "PASS" if self.all_passed else "FAIL"))
        return rows


def integrator_options_from_config(cfg: KerrRayConfig) -> IntegratorOptions:
    """Map the ``integration`` (and ``raytrace.dtype``) blocks to :class:`IntegratorOptions`."""
    integ = cfg.integration
    return IntegratorOptions(
        method=integ.method,
        rtol=integ.rtol,
        atol=integ.atol,
        step_size=integ.step_size,
        max_steps=integ.max_steps,
        lambda_max=integ.lambda_max,
        dtype=cfg.raytrace.dtype,
    )


def termination_options_from_config(cfg: KerrRayConfig) -> TerminationOptions:
    """Map the ``termination`` block to :class:`TerminationOptions`."""
    return TerminationOptions(
        horizon_epsilon=cfg.termination.horizon_epsilon, escape_radius=cfg.termination.escape_radius
    )
