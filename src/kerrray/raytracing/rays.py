"""Chunked batch ray tracing on a named backend (docs/architecture.md section 5).

:func:`trace_rays` splits a batch of initial states into chunks of at most
``chunk_size`` rays, integrates each chunk with the selected backend and
concatenates the per-ray fields of the resulting
:class:`~kerrray.photons.BatchResult` objects. Chunking bounds the memory of
the vectorised integrator (its right-hand side allocates a few dozen
temporaries of shape ``(n_active, 8)``) and gives the caller a progress hook.
Every ray carries its own adaptive step size and error control inside the
batched integrator, so the result of a ray does not depend on which other
rays share its chunk; the chunked run is therefore identical to an unchunked
one (verified in ``tests/test_backends.py``).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics import EventOptions, IntegratorOptions, TerminationOptions
from kerrray.geodesics.state import STATE_SIZE
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult
from kerrray.raytracing.backends.base import Backend, get_backend

__all__ = ["DEFAULT_CHUNK_SIZE", "ProgressCallback", "concatenate_results", "trace_rays"]

DEFAULT_CHUNK_SIZE = 16384
"""Rays per chunk: 16384 x 8 float64 states are 1 MB, so the ~60 temporaries of
the NumPy right-hand side stay near 60 MB per chunk."""

ProgressCallback = Callable[[float], None]
"""Called with the completed fraction in ``(0, 1]`` after every chunk."""


def _optional_concat(parts: list[NDArray | None]) -> NDArray | None:
    """Concatenate optional per-ray arrays; ``None`` when any part lacks them."""
    if any(part is None for part in parts):
        return None
    return np.concatenate([part for part in parts if part is not None], axis=0)


def concatenate_results(parts: list[BatchResult], runtime_s: float | None = None) -> BatchResult:
    """Join the per-ray fields of ``parts`` in order into one :class:`BatchResult`.

    ``runtime_s`` defaults to the sum of the parts' runtimes; :func:`trace_rays`
    passes the wall-clock time of the whole call instead.
    """
    if not parts:
        raise ValueError("concatenate_results needs at least one BatchResult")
    if len(parts) == 1 and runtime_s is None:
        return parts[0]
    return BatchResult(
        Y=np.concatenate([p.Y for p in parts], axis=0),
        state=np.concatenate([p.state for p in parts]),
        n_steps=np.concatenate([p.n_steps for p in parts]),
        lam=np.concatenate([p.lam for p in parts]),
        max_null_error=np.concatenate([p.max_null_error for p in parts]),
        max_energy_drift=np.concatenate([p.max_energy_drift for p in parts]),
        max_lz_drift=np.concatenate([p.max_lz_drift for p in parts]),
        max_carter_drift=np.concatenate([p.max_carter_drift for p in parts]),
        runtime_s=float(sum(p.runtime_s for p in parts)) if runtime_s is None else float(runtime_s),
        event_Y=_optional_concat([p.event_Y for p in parts]),
        event_hit=_optional_concat([p.event_hit for p in parts]),
        n_rejected=_optional_concat([p.n_rejected for p in parts]),
    )


def trace_rays(
    st: Spacetime,
    Y0: ArrayLike,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    backend: str | Backend = "numpy",
    events: EventOptions | None = None,
    progress: ProgressCallback | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> BatchResult:
    """Integrate the states ``Y0`` (N, 8) in chunks on ``backend``.

    Args:
        st: The spacetime.
        Y0: Initial states, one row per ray (finite, shape ``(N, 8)``, ``N >= 1``).
        integ: Scheme, tolerances and budgets (``"rk4"`` or ``"rk45"``).
        term: Capture margin and escape radius.
        backend: A backend name for :func:`kerrray.raytracing.backends.get_backend`
            or a :class:`~kerrray.raytracing.backends.base.Backend` instance.
        events: Optional disk-plane event passed to every chunk.
        progress: Called with the completed fraction after each chunk
            (the last call receives exactly ``1.0``).
        chunk_size: Maximum rays per chunk (``>= 1``).

    Returns:
        One :class:`BatchResult` for all ``N`` rays in the input order;
        ``runtime_s`` is the wall-clock time of this call.
    """
    y0 = np.asarray(Y0)
    if y0.ndim != 2 or y0.shape[1] != STATE_SIZE:
        raise ValueError(f"Y0 must have shape (N, {STATE_SIZE}), got {y0.shape}")
    n_rays = y0.shape[0]
    if n_rays < 1:
        raise ValueError("Y0 must contain at least one ray")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size!r}")
    engine = get_backend(backend) if isinstance(backend, str) else backend
    t0 = time.perf_counter()
    parts: list[BatchResult] = []
    for start in range(0, n_rays, chunk_size):
        stop = min(start + chunk_size, n_rays)
        parts.append(engine.integrate_batch(st, y0[start:stop], integ, term, events=events))
        if progress is not None:
            progress(stop / n_rays)
    return concatenate_results(parts, runtime_s=time.perf_counter() - t0)
