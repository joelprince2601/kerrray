"""Configuration of the solver benchmark and the step-size study.

Parameter sub-blocks (``experiment.parameters.solver`` and
``experiment.parameters.convergence``, docs/decisions.md D-009), the solver
configuration records and the builders of :class:`IntegratorOptions` and
:class:`TerminationOptions`. Split from :mod:`kerrray.benchmarks.solver`
(which re-exports every name) to keep the modules short; the module
docstring of :mod:`kerrray.benchmarks.solver` documents the defaults.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from kerrray.benchmarks.rayset import ESCAPE_SCALE, LAUNCH_RADIUS_FRACTION
from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime
from kerrray.photons.orbits import critical_impact_parameters
from kerrray.utils.config import KerrRayConfig
from kerrray.utils.config_parsing import ConfigError

__all__ = [
    "DEFAULT_LAUNCH_RADIUS",
    "DEFAULT_MAX_SECONDS_PER_CONFIG",
    "DEFAULT_N_RAYS",
    "DEFAULT_REPEATS",
    "DEFAULT_SPINS",
    "EXPERIMENT_NAME",
    "SOLVER_NAMES",
    "ConvergenceParams",
    "SolverConfiguration",
    "SolverParams",
    "benchmark_termination",
    "build_configurations",
    "integrator_options",
    "minimum_launch_radius",
    "termination_options",
]

EXPERIMENT_NAME: Final[str] = "benchmark_solver"
DEFAULT_N_RAYS: Final[int] = 40
DEFAULT_REPEATS: Final[int] = 3
DEFAULT_SPINS: Final[tuple[float, float]] = (0.0, 0.9)
DEFAULT_LAUNCH_RADIUS: Final[float] = 50.0
"""Launch and escape radius of the benchmark rays in units of M (module docstring)."""
DEFAULT_MAX_SECONDS_PER_CONFIG: Final[float] = 600.0
"""Wall-clock budget of one fixed-step configuration (all repeats plus the comparison pass)."""
SOLVER_NAMES: Final[dict[str, str]] = {"rk4": "RK4", "rk45": "RK45", "dop853": "DOP853"}


@dataclass(frozen=True)
class SolverParams:
    """``experiment.parameters.solver`` sub-block (numerical role, see module docstring)."""

    n_rays: int = DEFAULT_N_RAYS
    repeats: int = DEFAULT_REPEATS
    spins: list[float] = field(default_factory=lambda: list(DEFAULT_SPINS))
    launch_radius: float = DEFAULT_LAUNCH_RADIUS
    reference_rtol: float = 1e-13
    reference_atol: float = 1e-15
    max_seconds_per_config: float = DEFAULT_MAX_SECONDS_PER_CONFIG

    def __post_init__(self) -> None:
        if self.n_rays < 1:
            raise ConfigError(f"'solver.n_rays' must be >= 1, got {self.n_rays!r}")
        if self.repeats < 1:
            raise ConfigError(f"'solver.repeats' must be >= 1, got {self.repeats!r}")
        if not self.spins or any(not abs(s) < 1.0 for s in self.spins):
            raise ConfigError(f"'solver.spins' must be a non-empty list with |spin| < 1, got {self.spins!r}")
        for key in ("launch_radius", "reference_rtol", "reference_atol", "max_seconds_per_config"):
            value = getattr(self, key)
            if not (math.isfinite(value) and value > 0.0):
                raise ConfigError(f"'solver.{key}' must be > 0, got {value!r}")

    def replace(self, **changes: Any) -> SolverParams:
        """Copy with ``changes`` applied (``None`` values are ignored)."""
        values = {k: getattr(self, k) for k in self.__dataclass_fields__}
        values.update({k: v for k, v in changes.items() if v is not None})
        return SolverParams(**values)


def minimum_launch_radius(spins: Sequence[float], mass: float) -> float:
    """Smallest launch radius at which the escaping rays of :func:`build_ray_set` can be built (units of M)."""
    b_max = max(max(critical_impact_parameters(Spacetime(mass=mass, spin=float(s)))) for s in spins)
    return ESCAPE_SCALE[0] * b_max / LAUNCH_RADIUS_FRACTION


@dataclass(frozen=True)
class ConvergenceParams:
    """``experiment.parameters.convergence`` sub-block with the D-009 defaults."""

    resolutions: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    tolerances: list[float] = field(default_factory=lambda: [1e-6, 1e-8, 1e-10, 1e-12])
    step_sizes: list[float] = field(default_factory=lambda: [0.1, 0.01, 0.001, 0.0001])

    def __post_init__(self) -> None:
        for key in ("tolerances", "step_sizes"):
            values = getattr(self, key)
            if not values or any(not (math.isfinite(v) and v > 0.0) for v in values):
                raise ConfigError(f"'convergence.{key}' must be a non-empty list of positive numbers")


@dataclass(frozen=True)
class SolverConfiguration:
    """One solver setting: ``solver`` is ``rk4`` / ``rk45`` / ``dop853``, ``setting`` is ``h`` or ``rtol``."""

    solver: str
    setting: float
    integ: IntegratorOptions

    @property
    def name(self) -> str:
        """Display name, e.g. ``RK45``."""
        return SOLVER_NAMES[self.solver]

    @property
    def setting_label(self) -> str:
        """``h = 0.1`` or ``rtol = 1e-08``."""
        return f"h = {self.setting:g}" if self.solver == "rk4" else f"rtol = {self.setting:.0e}"

    @property
    def label(self) -> str:
        """``RK4 h = 0.1`` style label."""
        return f"{self.name} {self.setting_label}"


def termination_options(cfg: KerrRayConfig) -> TerminationOptions:
    """Production termination rules from the configuration."""
    return TerminationOptions(
        horizon_epsilon=cfg.termination.horizon_epsilon, escape_radius=cfg.termination.escape_radius
    )


def benchmark_termination(cfg: KerrRayConfig, launch_radius: float) -> TerminationOptions:
    """Termination of the benchmark rays: configured ``horizon_epsilon``, escape at ``launch_radius``."""
    return TerminationOptions(horizon_epsilon=cfg.termination.horizon_epsilon, escape_radius=launch_radius)


def integrator_options(cfg: KerrRayConfig, method: str, *, rtol: float | None = None, step_size: float | None = None) -> IntegratorOptions:
    """Integrator options for ``method`` with the configured budgets and the atol/rtol ratio."""
    base = cfg.integration
    ratio = base.atol / base.rtol
    r = base.rtol if rtol is None else rtol
    return IntegratorOptions(
        method=method,
        rtol=r,
        atol=r * ratio,
        step_size=base.step_size if step_size is None else step_size,
        max_steps=base.max_steps,
        lambda_max=base.lambda_max,
    )


def build_configurations(
    cfg: KerrRayConfig, conv: ConvergenceParams, *, launch_radius: float, include_dop853: bool = True
) -> tuple[list[SolverConfiguration], list[dict[str, Any]]]:
    """Return the configurations to run and the skipped fixed-step ones (with the reason)."""
    configurations: list[SolverConfiguration] = []
    skipped: list[dict[str, Any]] = []
    for h in conv.step_sizes:
        estimate = int(math.ceil(2.0 * launch_radius / h))
        if estimate > cfg.integration.max_steps:
            skipped.append(
                {"solver": "rk4", "setting": h, "estimated_steps": estimate, "max_steps": cfg.integration.max_steps,
                 "reason": f"about {estimate} steps per escaping ray (2 r0 / h) exceed integration.max_steps = "
                           f"{cfg.integration.max_steps}"}
            )
            continue
        configurations.append(SolverConfiguration("rk4", h, integrator_options(cfg, "rk4", step_size=h)))
    for rtol in conv.tolerances:
        configurations.append(SolverConfiguration("rk45", rtol, integrator_options(cfg, "rk45", rtol=rtol)))
    if include_dop853:
        for rtol in conv.tolerances:
            configurations.append(SolverConfiguration("dop853", rtol, integrator_options(cfg, "dop853", rtol=rtol)))
    return configurations, skipped
