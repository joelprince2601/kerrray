"""Typed configuration schema for KerrRay (PROJECT.md section 34, docs/architecture.md section 7).

A configuration is a YAML document with these top-level blocks::

    black_hole:
      mass: 1.0                 # > 0, geometric units (kerrray.utils.units)
      spin: 0.9                 # dimensionless a_* = a / M, |spin| < 1
    observer:
      radius: 1000.0            # > 0, Boyer-Lindquist r in units of M
      inclination_deg: 60.0     # 0..180, Boyer-Lindquist polar angle theta of the
                                # observer (0 = spin axis, 90 = equatorial plane);
                                # clamped 1e-3 deg away from the axis with a warning (D-008)
      phi_deg: 0.0              # optional azimuth in degrees, default 0
    integration:
      method: rk45              # rk4 | rk45 | dop853 (D-005)
      rtol: 1e-9                # > 0
      atol: 1e-11               # > 0
      max_steps: 100000         # >= 1
      step_size: 0.01           # > 0, optional: fixed step (rk4) / initial step, units of M
      lambda_max: 10000.0       # > 0, optional: affine-parameter budget, units of M
    raytrace:
      resolution: 64            # >= 1, image has resolution x resolution pixels
      fov: 12.0                 # > 0, optional: image-plane half-width in units of M
      backend: numpy            # numpy | numba | cuda, optional (cuda unavailable, D-007)
      dtype: float64            # float32 | float64, optional
    termination:
      horizon_epsilon: 1e-6     # > 0, CAPTURED when r <= r_+ + epsilon
      escape_radius: 1000.0     # > 0, ESCAPED needs r >= escape_radius AND outward motion
    experiment:                 # OPTIONAL block: run identity and experiment parameters
      name: kerr                # defaults to the file stem (load_config) or "unnamed"
      description: ...          # optional free text
      seed: 0                   # >= 0, optional: random seed recorded in the manifest
      output_dir: runs          # optional: parent of <run_id>/manifest.json
      report_dir: reports       # optional: parent of <run_id>/report.md
      parameters: {...}         # optional mapping of named sub-blocks (D-009)

The section 34 keys of the five physics blocks are required (there are no
hidden defaults, D-004) except ``observer.phi_deg``; the section 34 example
therefore loads verbatim. The keys added by docs/architecture.md section 7
(``step_size``, ``lambda_max``, ``fov``, ``backend``, ``dtype``, ``seed``,
``output_dir``, ``report_dir``) carry the documented defaults above so older
files still load, and every file in ``configs/`` states them explicitly. The
``experiment`` block may be omitted entirely: its ``name`` is then derived
from the configuration file name by :func:`load_config`
(``configs/kerr.yaml`` -> ``kerr``) or set to :data:`DEFAULT_EXPERIMENT_NAME`
when there is no file. Unknown keys anywhere raise :class:`ConfigError`.
PyYAML reads scalars written like ``1e-9`` as strings, so numeric fields also
accept numeric strings. Experiment drivers read their named sub-block of
``experiment.parameters`` with :func:`experiment_block`. The block dataclasses
themselves live in :mod:`kerrray.utils.config_blocks` and are re-exported here.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, TypeVar

import yaml

from kerrray.utils.config_blocks import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    INCLINATION_EPSILON_DEG,
    SUPPORTED_BACKENDS,
    SUPPORTED_DTYPES,
    SUPPORTED_INTEGRATION_METHODS,
    BlackHoleConfig,
    ExperimentConfig,
    IntegrationConfig,
    ObserverConfig,
    RaytraceConfig,
    TerminationConfig,
)
from kerrray.utils.config_parsing import (
    ConfigError,
    parse_block,
    parse_yaml,
    reject_unknown,
    to_plain,
    validate_block_type,
)
from kerrray.utils.logging import get_logger

__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_REPORT_DIR",
    "EXPERIMENT_BLOCK",
    "INCLINATION_EPSILON_DEG",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_DTYPES",
    "SUPPORTED_INTEGRATION_METHODS",
    "BlackHoleConfig",
    "ConfigError",
    "ExperimentConfig",
    "IntegrationConfig",
    "KerrRayConfig",
    "ObserverConfig",
    "RaytraceConfig",
    "TerminationConfig",
    "apply_overrides",
    "dump_config",
    "experiment_block",
    "load_config",
    "load_yaml_mapping",
]

_T = TypeVar("_T")

logger = get_logger(__name__)

EXPERIMENT_BLOCK: Final[str] = "experiment"
"""Name of the optional top-level block."""

DEFAULT_EXPERIMENT_NAME: Final[str] = "unnamed"
"""``experiment.name`` used when a configuration has no ``experiment`` block
and did not come from a file (:func:`load_config` uses the file stem instead)."""


def _block_to_mapping(block: Any) -> dict[str, Any]:
    """Plain ``dict`` of one block's fields (deep-copied, read-only views unwrapped)."""
    return {spec.name: to_plain(getattr(block, spec.name)) for spec in fields(block)}


@dataclass(frozen=True)
class KerrRayConfig:
    """Complete, validated KerrRay configuration (section 34 blocks plus ``experiment``)."""

    black_hole: BlackHoleConfig
    observer: ObserverConfig
    integration: IntegrationConfig
    raytrace: RaytraceConfig
    termination: TerminationConfig
    experiment: ExperimentConfig

    def __post_init__(self) -> None:
        if self.termination.escape_radius < self.observer.radius:
            logger.warning(
                "termination.escape_radius (%g) is smaller than observer.radius (%g): "
                "rays start outside the escape sphere (see docs/decisions.md, D-002)",
                self.termination.escape_radius,
                self.observer.radius,
            )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source: str = "<mapping>",
        default_name: str = DEFAULT_EXPERIMENT_NAME,
    ) -> KerrRayConfig:
        """Build and validate a configuration from a nested mapping.

        Args:
            data: Mapping with the section 34 blocks, e.g. from ``yaml.safe_load``.
                The ``experiment`` block may be absent or empty.
            source: Label used in error messages (usually the file path).
            default_name: ``experiment.name`` used when the block, or its
                ``name`` key, is absent.

        Raises:
            ConfigError: On unknown keys, missing keys, wrong types or constraint violations.
        """
        try:
            if not isinstance(data, Mapping):
                raise ConfigError(f"top level must be a mapping, got {type(data).__name__}")
            reject_unknown(data, _BLOCK_TYPES, "top level")
            blocks: dict[str, Any] = {}
            for name, block_type in _BLOCK_TYPES.items():
                if name == EXPERIMENT_BLOCK:
                    blocks[name] = _parse_experiment(data.get(name), default_name)
                elif name not in data:
                    raise ConfigError(f"missing required block '{name}'")
                else:
                    blocks[name] = parse_block(block_type, data[name], name)
            return cls(**blocks)
        except ConfigError as exc:
            raise ConfigError(f"{source}: {exc}") from exc

    @classmethod
    def from_yaml(
        cls, text: str, *, source: str = "<string>", default_name: str = DEFAULT_EXPERIMENT_NAME
    ) -> KerrRayConfig:
        """Parse a YAML document (see :func:`load_config` for files)."""
        return cls.from_mapping(parse_yaml(text, source), source=source, default_name=default_name)

    def to_mapping(self) -> dict[str, Any]:
        """Return a fresh, mutable nested ``dict`` in the section 34 layout.

        The result is deep-copied from the configuration (read-only views are
        unwrapped), so editing it never changes the configuration object.
        """
        return {name: _block_to_mapping(getattr(self, name)) for name in _BLOCK_TYPES}

    def to_yaml(self) -> str:
        """Serialise to YAML text that :meth:`from_yaml` reads back unchanged."""
        return yaml.safe_dump(self.to_mapping(), sort_keys=False, default_flow_style=False)

    def with_overrides(self, overrides: Mapping[str, Any]) -> KerrRayConfig:
        """Return a new configuration with dotted-key overrides applied and re-validated."""
        merged = apply_overrides(self.to_mapping(), overrides)
        return type(self).from_mapping(
            merged, source="<override>", default_name=self.experiment.name
        )


_BLOCK_TYPES: Final[dict[str, type]] = {
    "black_hole": BlackHoleConfig,
    "observer": ObserverConfig,
    "integration": IntegrationConfig,
    "raytrace": RaytraceConfig,
    "termination": TerminationConfig,
    EXPERIMENT_BLOCK: ExperimentConfig,
}
"""Top-level blocks in file order; every block except ``experiment`` is required."""


def _check_schema() -> None:
    """Fail at import time if any block field lacks a converter (schema programming error)."""
    for block_type in _BLOCK_TYPES.values():
        validate_block_type(block_type)


_check_schema()


def _parse_experiment(raw: Any, default_name: str) -> ExperimentConfig:
    """Parse the optional ``experiment`` block; absent or empty means defaults."""
    if raw is None:
        return ExperimentConfig(name=default_name)
    if isinstance(raw, Mapping) and "name" not in raw:
        raw = {**raw, "name": default_name}
    return parse_block(ExperimentConfig, raw, EXPERIMENT_BLOCK)


def experiment_block(cfg: KerrRayConfig, name: str, block_type: type[_T]) -> _T:
    """Parse ``cfg.experiment.parameters[name]`` into the frozen dataclass ``block_type``.

    The sub-block names and their defaults are fixed in docs/architecture.md
    section 7 (docs/decisions.md, D-009). Field annotations select converters
    from :data:`kerrray.utils.config_parsing.CONVERTERS` (``float``, ``int``,
    ``str``, ``bool``, ``float | None``, ``list[float]``, ``list[int]``,
    ``list[str]``, ...); unknown keys inside the sub-block are errors.

    Args:
        cfg: The configuration.
        name: Sub-block name under ``experiment.parameters`` (``"lensing"``, ...).
        block_type: Frozen dataclass describing the sub-block; every field
            that may be omitted needs a default.

    Returns:
        The parsed dataclass, or ``block_type()`` (all defaults) when the
        sub-block is absent or empty.

    Raises:
        ConfigError: On unknown keys, wrong value types, constraint violations
            raised by ``block_type.__post_init__``, or a required field
            missing (also when the whole sub-block is absent).
        TypeError: If a field annotation of ``block_type`` has no converter.
    """
    validate_block_type(block_type)
    path = f"{EXPERIMENT_BLOCK}.parameters.{name}"
    raw = cfg.experiment.parameters.get(name)
    if raw is None:
        try:
            return block_type()
        except TypeError as exc:
            raise ConfigError(f"sub-block '{path}' is required: {exc}") from None
    return parse_block(block_type, raw, path)


def apply_overrides(data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``data`` with dotted-key overrides applied.

    ``{"black_hole.spin": 0.5}`` sets ``data["black_hole"]["spin"]``. Missing
    intermediate mappings are created; whether the resulting keys and values
    are valid is decided by :meth:`KerrRayConfig.from_mapping`. Values may be
    strings (as typed on a command line): the schema converters coerce them.
    """
    result: dict[str, Any] = copy.deepcopy(dict(data))
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        if not dotted or any(not part for part in parts):
            raise ConfigError(f"invalid override key {dotted!r}")
        node = result
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            elif not isinstance(child, dict):
                raise ConfigError(f"cannot apply override {dotted!r}: '{part}' is not a mapping")
            node = child
        node[parts[-1]] = value
    return result


def load_yaml_mapping(path: Path | str) -> dict[str, Any]:
    """Read a UTF-8 YAML file whose top level is a mapping, without schema validation.

    Raises:
        ConfigError: If the file cannot be read, is not valid UTF-8, is not
            valid YAML, or its top level is not a non-empty mapping. The
            message names the file.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read configuration file {file_path}: {exc}") from exc
    return parse_yaml(text, str(file_path))


def load_config(path: Path | str, overrides: Mapping[str, Any] | None = None) -> KerrRayConfig:
    """Load and validate a configuration file, optionally applying dotted overrides.

    When the file has no ``experiment`` block (or no ``experiment.name``) the
    experiment is named after the file stem, e.g. ``configs/kerr.yaml`` -> ``kerr``.
    """
    file_path = Path(path)
    data = load_yaml_mapping(file_path)
    if overrides:
        data = apply_overrides(data, overrides)
    return KerrRayConfig.from_mapping(data, source=str(path), default_name=file_path.stem)


def dump_config(config: KerrRayConfig, path: Path | str) -> Path:
    """Write ``config`` as YAML (UTF-8, LF line endings) and return the path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(config.to_yaml())
    return file_path
