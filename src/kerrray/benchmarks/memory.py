"""Memory accounting for the batched integrators (PROJECT.md section 26; docs/decisions.md D-007).

Two complementary numbers are reported for every benchmark run:

* :func:`peak_memory` measures the peak of the *traced* Python-level
  allocations while a callable runs, with :mod:`tracemalloc` (no ``psutil``).
  NumPy registers its array buffers with ``tracemalloc``, so the NumPy
  backend's temporaries are counted; buffers allocated inside Numba-compiled
  code (its own runtime allocator) are **not** traced, so for the ``numba``
  backend the traced peak covers only the wrapper's arrays (documented
  limitation, docs/numerical_analysis.md, CPU benchmark).
* :func:`array_bytes` is the analytic estimate ``N * 8 * itemsize *
  working_arrays`` of the memory the batched integrator needs for ``N`` rays.
  :data:`NUMPY_WORKING_ARRAYS` is the number of ``(N, 8)`` arrays of the
  working dtype live at the peak of one iteration of
  :func:`kerrray.geodesics.integrate_batch`: the persistent ``Y`` and ``F``,
  the gathered ``y`` and ``F[idx]``, the six new Runge-Kutta stages, ``y_new``
  and ``err``, the temporaries of the last right-hand-side evaluation (fifteen
  metric terms plus about fourteen intermediates, each ``(N,)``, and the
  ``(N, 8)`` output) and the error-norm temporaries; the per-ray ``(N,)``
  bookkeeping arrays add about two more. A code count gives about 20; the
  measured ``tracemalloc`` peak of the NumPy backend on Kerr camera rays is
  23.4 to 24.3 such arrays (256 to 2304 rays, float64, ``rk45``), so the
  measured value 24 is used. :data:`NUMBA_WORKING_ARRAYS` counts the
  persistent ``Y`` and ``F`` of the Numba backend plus the ``(7, 8)`` stage
  block and the six ``(8,)`` scratch vectors each thread holds; those
  per-thread arrays do not scale with ``N``, so at the benchmark sizes the
  estimate is effectively ``2 N * 8 * itemsize``.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from typing import Final, TypeVar

import numpy as np
from numpy.typing import DTypeLike

from kerrray.geodesics.state import STATE_SIZE

__all__ = [
    "NUMBA_WORKING_ARRAYS",
    "NUMPY_WORKING_ARRAYS",
    "WORKING_ARRAYS",
    "array_bytes",
    "format_bytes",
    "peak_memory",
]

_T = TypeVar("_T")

NUMPY_WORKING_ARRAYS: Final[int] = 24
"""``(N, 8)`` working arrays of the NumPy backend at its peak (measured; module docstring)."""

NUMBA_WORKING_ARRAYS: Final[int] = 2
"""``(N, 8)`` arrays of the Numba backend that scale with ``N`` (``Y`` and ``F``)."""

WORKING_ARRAYS: Final[dict[str, int]] = {"numpy": NUMPY_WORKING_ARRAYS, "numba": NUMBA_WORKING_ARRAYS}
"""Working-array count per backend name for :func:`array_bytes`."""

PER_RAY_SCALARS_BYTES: Final[int] = 8 * 15
"""Per-ray float64/int64 bookkeeping (lambda, h, counters, state, diagnostics and their references)."""


def array_bytes(
    n_rays: int,
    dtype: DTypeLike = np.float64,
    working_arrays: int = NUMPY_WORKING_ARRAYS,
    *,
    per_ray_scalars: int = PER_RAY_SCALARS_BYTES,
) -> int:
    """Estimated bytes of ``working_arrays`` state-sized arrays for ``n_rays`` plus per-ray scalars.

    ``n_rays * STATE_SIZE * itemsize(dtype) * working_arrays + n_rays * per_ray_scalars``.
    """
    if n_rays < 0 or working_arrays < 0 or per_ray_scalars < 0:
        raise ValueError("n_rays, working_arrays and per_ray_scalars must be >= 0")
    itemsize = np.dtype(dtype).itemsize
    return int(n_rays) * STATE_SIZE * itemsize * int(working_arrays) + int(n_rays) * int(per_ray_scalars)


def peak_memory(fn: Callable[[], _T]) -> tuple[_T, int]:
    """Run ``fn()`` and return ``(result, peak_traced_bytes)`` measured with :mod:`tracemalloc`.

    The peak is the maximum of the traced allocations *above the level at the
    start of the call*: tracing is started here (and stopped afterwards) when
    it is not already active; when a caller already traces, the peak counter
    is reset instead and the current usage at entry is subtracted. Tracing
    slows Python-level allocation, so timing runs must be made separately.
    """
    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if started_here:
            tracemalloc.stop()
    return result, max(int(peak) - int(base), 0)


def format_bytes(n_bytes: int) -> str:
    """Human-readable size (``"12.3 MB"``) with decimal (SI) prefixes."""
    value = float(n_bytes)
    for unit in ("B", "kB", "MB", "GB"):
        if abs(value) < 1000.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000.0
    return f"{value:.1f} GB"
