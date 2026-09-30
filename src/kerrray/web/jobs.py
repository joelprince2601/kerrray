"""Background jobs with live progress for KerrRay's interactive app.

Long computations (a shadow trace, the Schwarzschild validation) run in a
worker thread. The engine reports progress through a callback; the browser
polls :meth:`JobManager.poll` a few times per second and draws what has been
computed so far. Cancelling sets a flag that the progress callback turns into
an exception, which stops the integrator at its next sweep.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kerrray.utils.logging import get_logger

logger = get_logger(__name__)

MAX_JOBS = 16
"""Finished jobs beyond this count are forgotten, oldest first."""


class JobCancelled(Exception):
    """Raised inside a job's progress callback after the job was cancelled."""


@dataclass
class Job:
    """One background computation."""

    id: str
    kind: str
    started: float
    status: str = "running"  # running | done | error | cancelled
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    finished: float | None = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)


def _serialise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class JobManager:
    """Thread-safe registry of background jobs."""

    def __init__(self, runners: dict[str, Callable[..., dict[str, Any]]]) -> None:
        self._runners = runners
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def start(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        """Start job ``kind`` with ``params``; returns ``{"id": ...}``."""
        if kind not in self._runners:
            raise ValueError(f"unknown job kind {kind!r}")
        job = Job(id=f"{kind}-{next(self._ids)}", kind=kind, started=time.perf_counter())

        def progress(update: dict[str, Any]) -> None:
            if job.cancel_flag.is_set():
                raise JobCancelled()
            job.progress = update

        def work() -> None:
            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    job.result = self._runners[kind](params, progress=progress)
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                logger.error("job %s failed:\n%s", job.id, traceback.format_exc())
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "error"
            finally:
                job.finished = time.perf_counter()

        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        threading.Thread(target=work, name=job.id, daemon=True).start()
        return {"id": job.id}

    def poll(self, job_id: str) -> dict[str, Any]:
        """Status, elapsed time, latest progress and (when done) the result."""
        job = self._get(job_id)
        end = job.finished if job.finished is not None else time.perf_counter()
        out: dict[str, Any] = {
            "id": job.id,
            "status": job.status,
            "elapsed_s": end - job.started,
            "progress": {k: _serialise(v) for k, v in dict(job.progress).items()},
        }
        if job.status == "done":
            out["result"] = job.result
        if job.status == "error":
            out["error"] = job.error
        return out

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a running job to stop at its next progress report."""
        job = self._get(job_id)
        job.cancel_flag.set()
        return {"id": job.id, "status": job.status}

    def _get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"unknown job {job_id!r}")
            return self._jobs[job_id]

    def _prune(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.status != "running"), key=lambda j: j.started)
        for job in finished[: max(0, len(self._jobs) - MAX_JOBS)]:
            del self._jobs[job.id]
