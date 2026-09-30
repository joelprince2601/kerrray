"""Experiment driver scaffolding: run directories, manifests and summaries.

Every experiment module in :mod:`kerrray.experiments` exposes
``run(cfg: KerrRayConfig) -> RunRecord`` and implements it with
:func:`run_experiment`, which owns the reproducibility bookkeeping of
PROJECT.md sections 35 and 36:

1. create ``<experiment.output_dir>/<run_id>/`` and write ``manifest.json``
   (status ``"running"``, empty results) and ``config.yaml`` through
   :mod:`kerrray.utils.manifest` *before* any computation, so an interrupted
   run still leaves its configuration behind;
2. call the experiment function with an :class:`ExperimentContext` (the
   configuration, the run directories, a namespaced logger, a NumPy random
   generator seeded from ``experiment.seed`` and a wall-clock timer);
3. rewrite the manifest with the JSON-sanitised results, the runtime and the
   status ``"completed"``, or ``"failed"`` with the exception text before the
   exception is re-raised;
4. write ``<experiment.report_dir>/<run_id>/summary.json`` and return a
   :class:`RunRecord`.

Manifest keys beyond :data:`kerrray.utils.manifest.MANIFEST_KEYS` added here:
``experiment`` (the experiment name), ``status``, ``runtime_s``, ``finished``
(UTC timestamp) and ``error`` (``None`` unless the run failed).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np

from kerrray.utils.config import KerrRayConfig, experiment_block
from kerrray.utils.logging import get_logger
from kerrray.utils.manifest import (
    EnvironmentInfo,
    build_manifest,
    collect_environment,
    new_run_id,
    sanitise_for_json,
    utc_timestamp,
    write_run,
)
from kerrray.utils.seeds import set_seed

__all__ = [
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "SUMMARY_FILENAME",
    "ExperimentContext",
    "ExperimentFunction",
    "RunRecord",
    "Timer",
    "run_experiment",
]

_T = TypeVar("_T")

STATUS_RUNNING: Final[str] = "running"
STATUS_COMPLETED: Final[str] = "completed"
STATUS_FAILED: Final[str] = "failed"
SUMMARY_FILENAME: Final[str] = "summary.json"
"""Name of the per-run summary written under ``<report_dir>/<run_id>/``."""


class Timer:
    """Wall-clock timer (``time.perf_counter``) started when constructed.

    ``elapsed_s()`` is the time since construction; ``lap(label)`` records
    the time since the previous lap (or construction) under ``label`` in
    :attr:`laps` and returns it, so drivers can time individual stages.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._last = self._start
        self.laps: dict[str, float] = {}

    def elapsed_s(self) -> float:
        """Seconds since the timer was created."""
        return time.perf_counter() - self._start

    def lap(self, label: str) -> float:
        """Record and return the seconds since the previous lap under ``label``."""
        now = time.perf_counter()
        duration = now - self._last
        self._last = now
        self.laps[label] = duration
        return duration


@dataclass(frozen=True)
class ExperimentContext:
    """Everything an experiment function receives from :func:`run_experiment`.

    Attributes:
        cfg: The validated configuration of the run.
        run_id: Run identifier (docs/decisions.md, D-006).
        run_dir: ``<output_dir>/<run_id>``, already created; drivers may write
            intermediate data files here.
        report_dir: ``<report_dir>/<run_id>``, already created; figures and the
            report go here.
        logger: Logger ``kerrray.experiments.<name>``.
        rng: NumPy generator from :func:`kerrray.utils.seeds.set_seed` with
            ``cfg.experiment.seed``; the only source of randomness a driver
            should use.
        timer: :class:`Timer` started just before the function is called.
    """

    cfg: KerrRayConfig
    run_id: str
    run_dir: Path
    report_dir: Path
    logger: logging.Logger
    rng: np.random.Generator
    timer: Timer

    def parameters(self, name: str, block_type: type[_T]) -> _T:
        """Parse ``cfg.experiment.parameters[name]`` into ``block_type`` (see
        :func:`kerrray.utils.config.experiment_block`)."""
        return experiment_block(self.cfg, name, block_type)


@dataclass(frozen=True)
class RunRecord:
    """What :func:`run_experiment` returns after a completed run."""

    run_id: str
    run_dir: Path
    report_dir: Path
    results: dict[str, Any]
    runtime_s: float
    manifest_path: Path
    summary_path: Path


ExperimentFunction = Callable[[ExperimentContext], Mapping[str, Any]]
"""An experiment body: takes the context, returns JSON-serialisable results."""


def _write_summary(report_dir: Path, manifest: Mapping[str, Any], manifest_path: Path) -> Path:
    summary = {
        "run_id": manifest["run_id"],
        "experiment": manifest["experiment"],
        "status": manifest["status"],
        "timestamp": manifest["timestamp"],
        "finished": manifest["finished"],
        "runtime_s": manifest["runtime_s"],
        "random_seed": manifest["random_seed"],
        "git_commit": manifest["git_commit"],
        "manifest_path": str(manifest_path),
        "error": manifest["error"],
        "results": manifest["results"],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / SUMMARY_FILENAME
    text = json.dumps(sanitise_for_json(summary), indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def run_experiment(
    name: str,
    cfg: KerrRayConfig,
    fn: ExperimentFunction,
    *,
    run_id: str | None = None,
    environment: EnvironmentInfo | None = None,
) -> RunRecord:
    """Run ``fn`` as a reproducible experiment named ``name`` (see the module docstring).

    Args:
        name: Experiment name recorded in the manifest (normally the module name).
        cfg: Validated configuration; ``cfg.experiment`` supplies the seed and
            the output and report directories (relative paths are resolved
            against the current working directory).
        fn: The experiment body. It must return a mapping of JSON-serialisable
            results; NumPy scalars and arrays are converted with
            :func:`kerrray.utils.manifest.sanitise_for_json`.
        run_id: Explicit run identifier; a new one is generated when ``None``.
        environment: Pre-collected environment snapshot (tests reuse one to
            avoid repeated ``git`` calls); collected when ``None``.

    Returns:
        The :class:`RunRecord` with the sanitised results and file paths.

    Raises:
        Exception: Whatever ``fn`` raised, after the manifest was rewritten with
            status ``"failed"``; ``TypeError`` if ``fn`` returned a non-mapping.
    """
    run_id = run_id if run_id is not None else new_run_id()
    run_dir = Path(cfg.experiment.output_dir) / run_id
    report_dir = Path(cfg.experiment.report_dir) / run_id
    env = environment if environment is not None else collect_environment()
    logger = get_logger(f"kerrray.experiments.{name}")

    manifest = build_manifest(run_id, cfg, random_seed=cfg.experiment.seed, environment=env)
    manifest.update(
        {"experiment": name, "status": STATUS_RUNNING, "runtime_s": None, "finished": None,
         "error": None}
    )
    paths = write_run(run_dir, cfg, manifest)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger.info("experiment %s: run %s started in %s", name, run_id, paths.run_dir)

    timer = Timer()
    ctx = ExperimentContext(
        cfg=cfg,
        run_id=run_id,
        run_dir=paths.run_dir,
        report_dir=report_dir,
        logger=logger,
        rng=set_seed(cfg.experiment.seed),
        timer=timer,
    )
    try:
        raw = fn(ctx)
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"experiment {name!r} must return a mapping of results, "
                f"got {type(raw).__name__}"
            )
        results = sanitise_for_json(dict(raw))
    except Exception as exc:
        manifest.update(
            {
                "status": STATUS_FAILED,
                "runtime_s": timer.elapsed_s(),
                "finished": utc_timestamp(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_run(run_dir, cfg, manifest)
        _write_summary(report_dir, manifest, paths.manifest_path)
        logger.error("experiment %s: run %s failed: %s", name, run_id, manifest["error"])
        raise

    manifest.update(
        {
            "results": results,
            "status": STATUS_COMPLETED,
            "runtime_s": timer.elapsed_s(),
            "finished": utc_timestamp(),
        }
    )
    paths = write_run(run_dir, cfg, manifest)
    summary_path = _write_summary(report_dir, manifest, paths.manifest_path)
    logger.info(
        "experiment %s: run %s completed in %.3f s", name, run_id, manifest["runtime_s"]
    )
    return RunRecord(
        run_id=run_id,
        run_dir=paths.run_dir,
        report_dir=report_dir,
        results=results,
        runtime_s=float(manifest["runtime_s"]),
        manifest_path=paths.manifest_path,
        summary_path=summary_path,
    )
