"""Run identifiers, environment capture and run manifests (PROJECT.md section 35).

Every KerrRay run writes ``runs/<run_id>/manifest.json`` and
``runs/<run_id>/config.yaml``. The manifest keys are fixed by
:data:`MANIFEST_KEYS`: the eleven section 35 entries in order, followed by
``git_dirty`` (whether the working tree had uncommitted or untracked changes,
so a ``git_commit`` is never mistaken for the exact code that ran).
:func:`build_manifest` assembles them from a configuration and the captured
environment, and :func:`write_run` writes both files as strictly standard
JSON (non-finite floats are written as the strings ``"nan"``, ``"inf"`` and
``"-inf"``, see :func:`sanitise_for_json`). Nothing here performs network
access.
"""

from __future__ import annotations

import json
import math
import os
import platform
import secrets
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import numpy as np

from kerrray.utils.config import KerrRayConfig, dump_config

TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "kerrray",
    "numpy",
    "scipy",
    "sympy",
    "matplotlib",
    "typer",
    "rich",
    "pyyaml",
)
"""Distributions whose versions are recorded in every manifest; KerrRay itself first,
so a run is attributable to an engine version even when ``git_commit`` is unknown."""

MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "run_id",
    "timestamp",
    "git_commit",
    "config",
    "python_version",
    "package_versions",
    "hardware",
    "solver",
    "tolerances",
    "random_seed",
    "results",
    "git_dirty",
)
"""Required manifest keys: PROJECT.md section 35 in order, then ``git_dirty``."""

MANIFEST_FILENAME: Final[str] = "manifest.json"
CONFIG_FILENAME: Final[str] = "config.yaml"
UNKNOWN_COMMIT: Final[str] = "unknown"
NOT_INSTALLED: Final[str] = "not installed"
_RUN_ID_TIME_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"
_RUN_ID_SUFFIX_BYTES: Final[int] = 3
_GIT_TIMEOUT_SECONDS: Final[float] = 10.0


def _as_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return now.astimezone(timezone.utc)


def new_run_id(now: datetime | None = None) -> str:
    """Return ``YYYYMMDDTHHMMSSZ-xxxxxx``: a UTC timestamp plus 6 random hex characters.

    The suffix comes from :mod:`secrets`, independent of the experiment seed,
    so two runs with identical configuration never share an id.
    """
    stamp = _as_utc(now).strftime(_RUN_ID_TIME_FORMAT)
    return f"{stamp}-{secrets.token_hex(_RUN_ID_SUFFIX_BYTES)}"


def utc_timestamp(now: datetime | None = None) -> str:
    """ISO 8601 UTC timestamp with second precision, e.g. ``2026-09-28T10:15:30+00:00``."""
    return _as_utc(now).isoformat(timespec="seconds")


def package_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed version of each distribution via :mod:`importlib.metadata`."""
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = NOT_INSTALLED
    return versions


def _is_commit_hash(text: str) -> bool:
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def _run_git(args: Sequence[str], repo_dir: Path | None) -> str | None:
    """Run ``git <args>`` in ``repo_dir`` and return its stdout, or ``None`` on any failure.

    ``repo_dir`` defaults to this file's directory so an editable install
    reports the checkout it runs from. Failures (git missing, not a
    repository, timeout, non-zero exit) are never raised.
    """
    cwd = repo_dir if repo_dir is not None else Path(__file__).resolve().parent
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_commit(repo_dir: Path | None = None) -> str:
    """Commit hash of the source tree from ``git rev-parse HEAD``.

    Args:
        repo_dir: Directory inside the repository; defaults to this file's
            directory so an editable install reports the checkout it runs from.

    Returns:
        The 40-character hash, or ``"unknown"`` when git is unavailable, the
        directory is not a repository, or the output is not a commit hash.
    """
    output = _run_git(["rev-parse", "HEAD"], repo_dir)
    if output is None:
        return UNKNOWN_COMMIT
    commit = output.strip()
    return commit if _is_commit_hash(commit) else UNKNOWN_COMMIT


def git_dirty(repo_dir: Path | None = None) -> bool | None:
    """Whether the working tree differs from ``HEAD`` (``git status --porcelain``).

    Modified, staged, deleted and untracked files all count as dirty, because
    any of them means the code or inputs that ran are not fully described by
    :func:`git_commit`.

    Args:
        repo_dir: Directory inside the repository; same default as :func:`git_commit`.

    Returns:
        ``True`` if the tree is dirty, ``False`` if clean, ``None`` when git is
        unavailable or the directory is not a repository.
    """
    output = _run_git(["status", "--porcelain"], repo_dir)
    if output is None:
        return None
    return bool(output.strip())


@dataclass(frozen=True)
class EnvironmentInfo:
    """Snapshot of the interpreter, packages, hardware and git state a run executes on."""

    python_version: str
    python_implementation: str
    package_versions: dict[str, str]
    platform: str
    system: str
    machine: str
    processor: str
    cpu_count: int | None
    git_commit: str
    git_dirty: bool | None

    def hardware(self) -> dict[str, Any]:
        """The ``hardware`` manifest entry."""
        return {
            "platform": self.platform,
            "system": self.system,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
        }

    def as_dict(self) -> dict[str, Any]:
        """Plain ``dict`` form of the snapshot."""
        return asdict(self)


def collect_environment(
    packages: Sequence[str] = TRACKED_PACKAGES, repo_dir: Path | None = None
) -> EnvironmentInfo:
    """Capture Python, package, platform, CPU and git information (all real values)."""
    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        package_versions=package_versions(packages),
        platform=platform.platform(),
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        git_commit=git_commit(repo_dir),
        git_dirty=git_dirty(repo_dir),
    )


def build_manifest(
    run_id: str,
    config: KerrRayConfig,
    *,
    random_seed: int,
    results: Mapping[str, Any] | None = None,
    environment: EnvironmentInfo | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the manifest mapping with every key in :data:`MANIFEST_KEYS`, in order."""
    env = environment if environment is not None else collect_environment()
    return {
        "run_id": run_id,
        "timestamp": utc_timestamp(timestamp),
        "git_commit": env.git_commit,
        "config": config.to_mapping(),
        "python_version": env.python_version,
        "package_versions": dict(env.package_versions),
        "hardware": env.hardware(),
        "solver": {
            "method": config.integration.method,
            "max_steps": config.integration.max_steps,
        },
        "tolerances": {
            "rtol": config.integration.rtol,
            "atol": config.integration.atol,
            "horizon_epsilon": config.termination.horizon_epsilon,
        },
        "random_seed": random_seed,
        "results": dict(results) if results else {},
        "git_dirty": env.git_dirty,
    }


@dataclass(frozen=True)
class RunPaths:
    """Files written for one run."""

    run_dir: Path
    manifest_path: Path
    config_path: Path


def _non_finite_label(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def sanitise_for_json(obj: Any) -> Any:
    """Return ``obj`` rebuilt from JSON-native types only (strict JSON, no NaN/Infinity).

    * NumPy arrays and scalars become lists and Python scalars.
    * Non-finite floats become the strings ``"nan"``, ``"inf"`` and ``"-inf"``
      (``float(...)`` reads them back), because ``NaN``/``Infinity`` literals
      are not valid JSON and strict parsers reject them.
    * Mapping keys become strings; tuples become lists; :class:`~pathlib.Path`
      and :class:`~datetime.datetime` become strings.

    Anything else is returned unchanged and left for :func:`json.dumps` to
    accept or reject.
    """
    if isinstance(obj, np.ndarray):
        return sanitise_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitise_for_json(obj.item())
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else _non_finite_label(obj)
    if isinstance(obj, Mapping):
        return {str(key): sanitise_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise_for_json(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def write_run(
    run_dir: Path | str, config: KerrRayConfig, manifest_fields: Mapping[str, Any]
) -> RunPaths:
    """Write ``manifest.json`` and ``config.yaml`` into ``run_dir`` (``runs/<run_id>``).

    Args:
        run_dir: The run directory, created if needed.
        config: The configuration the run used; written as ``config.yaml``.
        manifest_fields: Manifest mapping, normally from :func:`build_manifest`.
            Every key in :data:`MANIFEST_KEYS` must be present; extra keys are kept.

    Raises:
        ValueError: If required manifest keys are missing, or if a value that
            is not valid strict JSON survives :func:`sanitise_for_json`.
        TypeError: If a value has a type JSON cannot represent.
    """
    missing = [key for key in MANIFEST_KEYS if key not in manifest_fields]
    if missing:
        raise ValueError(f"manifest is missing required keys: {', '.join(missing)}")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_FILENAME
    text = json.dumps(sanitise_for_json(dict(manifest_fields)), indent=2, allow_nan=False)
    manifest_path.write_text(text + "\n", encoding="utf-8", newline="\n")
    config_path = dump_config(config, directory / CONFIG_FILENAME)
    return RunPaths(run_dir=directory, manifest_path=manifest_path, config_path=config_path)
