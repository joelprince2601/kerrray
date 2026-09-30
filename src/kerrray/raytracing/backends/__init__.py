"""Batched ray-integration backends (docs/architecture.md sections 5 and 9).

``numpy`` is the vectorised reference; ``numba`` (performance role,
``numba_backend.py``) is imported lazily and reported unavailable when it
cannot be imported; ``cuda`` is a named but unavailable backend because GPU
acceleration is optional future work (docs/decisions.md D-007).
"""

from kerrray.raytracing.backends.base import (
    BACKEND_NAMES,
    CUDA_UNAVAILABLE_MESSAGE,
    Backend,
    BackendUnavailable,
    available_backends,
    get_backend,
)
from kerrray.raytracing.backends.numpy_backend import NumpyBackend

__all__ = [
    "BACKEND_NAMES",
    "CUDA_UNAVAILABLE_MESSAGE",
    "Backend",
    "BackendUnavailable",
    "NumpyBackend",
    "available_backends",
    "get_backend",
]
