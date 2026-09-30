"""Typed configuration blocks (PROJECT.md section 34, docs/architecture.md section 7).

One frozen dataclass per top-level YAML block. Each ``__post_init__`` validates
ranges and raises :class:`~kerrray.utils.config_parsing.ConfigError`; the
schema assembly, file loading and overrides live in :mod:`kerrray.utils.config`,
which re-exports every name defined here. The observer-angle clamp is decision
D-008 and the integration method names are decision D-005 (docs/decisions.md).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from kerrray.utils.config_parsing import ConfigError, to_plain
from kerrray.utils.logging import get_logger
from kerrray.utils.seeds import DEFAULT_SEED

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_DIR",
    "INCLINATION_EPSILON_DEG",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_DTYPES",
    "SUPPORTED_INTEGRATION_METHODS",
    "BlackHoleConfig",
    "ExperimentConfig",
    "IntegrationConfig",
    "ObserverConfig",
    "RaytraceConfig",
    "TerminationConfig",
]

logger = get_logger(__name__)

SUPPORTED_INTEGRATION_METHODS: Final[frozenset[str]] = frozenset({"rk4", "rk45", "dop853"})
"""Integration methods the schema accepts (docs/decisions.md, D-005): ``rk4`` is
the fixed-step classical Runge-Kutta scheme, ``rk45`` the adaptive embedded
Dormand-Prince 5(4) scheme and ``dop853`` SciPy's eighth-order Dormand-Prince
reference integrator used for cross-checks."""

SUPPORTED_BACKENDS: Final[frozenset[str]] = frozenset({"numpy", "numba", "cuda"})
"""Ray-tracing backends the schema accepts. ``cuda`` is a named but unavailable
backend: GPU acceleration is optional future work (docs/decisions.md, D-007)."""

SUPPORTED_DTYPES: Final[frozenset[str]] = frozenset({"float32", "float64"})
"""Floating-point precisions of the batched integrator (PROJECT.md section 27)."""

INCLINATION_EPSILON_DEG: Final[float] = 1e-3
"""Half-width in degrees of the excluded band around the spin axis (D-008)."""

DEFAULT_OUTPUT_DIR: Final[str] = "runs"
"""Directory that receives ``<run_id>/manifest.json`` (PROJECT.md section 35)."""

DEFAULT_REPORT_DIR: Final[str] = "reports"
"""Directory that receives ``<run_id>/report.md`` (PROJECT.md section 36)."""


def _positive(value: float, key: str) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise ConfigError(f"'{key}' must be a finite number > 0, got {value!r}")


def _choice(value: str, allowed: frozenset[str], key: str) -> None:
    if value not in allowed:
        raise ConfigError(f"'{key}' must be one of {', '.join(sorted(allowed))}, got {value!r}")


@dataclass(frozen=True)
class BlackHoleConfig:
    """``black_hole`` block: mass ``M`` (> 0) and dimensionless spin (|spin| < 1)."""

    mass: float
    spin: float

    def __post_init__(self) -> None:
        _positive(self.mass, "black_hole.mass")
        if not (math.isfinite(self.spin) and abs(self.spin) < 1.0):
            raise ConfigError(f"'black_hole.spin' must satisfy |spin| < 1, got {self.spin!r}")


@dataclass(frozen=True)
class ObserverConfig:
    """``observer`` block: radius (> 0), polar angle in degrees (0..180), azimuth.

    ``inclination_deg`` is the Boyer-Lindquist polar angle ``theta`` of the
    observer (D-001). Values within :data:`INCLINATION_EPSILON_DEG` of either
    pole are clamped into ``[epsilon, 180 - epsilon]`` with a logged warning
    (D-008): WHAT, the observer is moved off the axis by at most 1e-3 degrees;
    WHY, the ZAMO tetrad that defines the camera is singular on the axis
    (``sin(theta) = 0``); LIMITATION, an "on-axis" image is really the image
    at ``theta = 1e-3`` degrees, and the shadow is symmetric only to that
    accuracy. Values outside ``[0, 180]`` remain errors.
    """

    radius: float
    inclination_deg: float
    phi_deg: float = 0.0

    def __post_init__(self) -> None:
        _positive(self.radius, "observer.radius")
        if not (math.isfinite(self.inclination_deg) and 0.0 <= self.inclination_deg <= 180.0):
            raise ConfigError(
                f"'observer.inclination_deg' must lie in [0, 180], got {self.inclination_deg!r}"
            )
        if not math.isfinite(self.phi_deg):
            raise ConfigError(f"'observer.phi_deg' must be finite, got {self.phi_deg!r}")
        low, high = INCLINATION_EPSILON_DEG, 180.0 - INCLINATION_EPSILON_DEG
        clamped = min(max(self.inclination_deg, low), high)
        if clamped != self.inclination_deg:
            logger.warning(
                "observer.inclination_deg = %g lies on the spin axis where the ZAMO tetrad "
                "is singular; clamped to %g deg (docs/decisions.md, D-008)",
                self.inclination_deg,
                clamped,
            )
            object.__setattr__(self, "inclination_deg", clamped)

    @property
    def inclination_rad(self) -> float:
        """The clamped observer polar angle ``theta`` in radians."""
        return math.radians(self.inclination_deg)

    @property
    def phi_rad(self) -> float:
        """The observer azimuth ``phi`` in radians."""
        return math.radians(self.phi_deg)


@dataclass(frozen=True)
class IntegrationConfig:
    """``integration`` block: scheme, tolerances, step cap, step size, affine budget.

    ``step_size`` is the fixed step in the affine parameter (units of ``M``) for
    ``rk4`` and the initial step for the adaptive schemes; ``lambda_max`` is the
    affine-parameter budget after which a ray is ``MAX_AFFINE_PARAMETER``
    (docs/architecture.md sections 1 and 4).
    """

    method: str
    rtol: float
    atol: float
    max_steps: int
    step_size: float = 0.01
    lambda_max: float = 1.0e4

    def __post_init__(self) -> None:
        _choice(self.method, SUPPORTED_INTEGRATION_METHODS, "integration.method")
        _positive(self.rtol, "integration.rtol")
        _positive(self.atol, "integration.atol")
        if self.max_steps < 1:
            raise ConfigError(f"'integration.max_steps' must be >= 1, got {self.max_steps!r}")
        _positive(self.step_size, "integration.step_size")
        _positive(self.lambda_max, "integration.lambda_max")


@dataclass(frozen=True)
class RaytraceConfig:
    """``raytrace`` block: image resolution, field of view, backend and precision.

    ``resolution`` is the number of pixels per side (>= 1); ``fov`` the
    half-width of the image plane in units of ``M`` (docs/architecture.md
    section 1.3); ``backend`` one of :data:`SUPPORTED_BACKENDS`; ``dtype`` one
    of :data:`SUPPORTED_DTYPES`.
    """

    resolution: int
    fov: float = 12.0
    backend: str = "numpy"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.resolution < 1:
            raise ConfigError(f"'raytrace.resolution' must be >= 1, got {self.resolution!r}")
        _positive(self.fov, "raytrace.fov")
        _choice(self.backend, SUPPORTED_BACKENDS, "raytrace.backend")
        _choice(self.dtype, SUPPORTED_DTYPES, "raytrace.dtype")


@dataclass(frozen=True)
class TerminationConfig:
    """``termination`` block: horizon capture tolerance and escape radius (both > 0)."""

    horizon_epsilon: float
    escape_radius: float

    def __post_init__(self) -> None:
        _positive(self.horizon_epsilon, "termination.horizon_epsilon")
        _positive(self.escape_radius, "termination.escape_radius")


@dataclass(frozen=True)
class ExperimentConfig:
    """Optional ``experiment`` block: run identity, seed, output directories, parameters.

    ``seed`` (>= 0) feeds :func:`kerrray.utils.seeds.set_seed` and is recorded
    in every manifest; ``output_dir`` and ``report_dir`` are the parents of the
    per-run directories of PROJECT.md sections 35 and 36. ``parameters`` is
    stored as a read-only ``MappingProxyType`` over a private deep copy, so
    keys cannot be added, replaced or removed through the configuration
    object. Values nested inside it (lists, sub-mappings) are the parsed YAML
    data themselves; experiment drivers must treat them as read-only and copy
    (for example via :meth:`~kerrray.utils.config.KerrRayConfig.to_mapping` or
    :func:`kerrray.utils.config_parsing.to_plain`) before modifying anything.
    The schema does not interpret ``parameters``; the owning experiment
    driver parses its named sub-block with
    :func:`kerrray.utils.config.experiment_block` (docs/decisions.md, D-004
    and D-009).
    """

    name: str
    description: str = ""
    seed: int = DEFAULT_SEED
    output_dir: str = DEFAULT_OUTPUT_DIR
    report_dir: str = DEFAULT_REPORT_DIR
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("'experiment.name' must not be empty")
        if self.seed < 0:
            raise ConfigError(f"'experiment.seed' must be >= 0, got {self.seed!r}")
        for key in ("output_dir", "report_dir"):
            if not getattr(self, key).strip():
                raise ConfigError(f"'experiment.{key}' must not be empty")
        if not isinstance(self.parameters, Mapping):
            raise ConfigError(
                f"'experiment.parameters' must be a mapping, got {type(self.parameters).__name__}"
            )
        object.__setattr__(self, "parameters", MappingProxyType(to_plain(self.parameters)))
