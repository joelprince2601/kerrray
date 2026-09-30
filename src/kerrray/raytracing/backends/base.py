"""Backend protocol and registry for batched ray integration.

A backend integrates a batch of photon states with the same tableau, the
same right-hand side and the same termination rules as the NumPy reference
(docs/architecture.md section 9). Three names are known
(:data:`kerrray.utils.config_blocks.SUPPORTED_BACKENDS`):

* ``"numpy"``: :class:`kerrray.raytracing.backends.numpy_backend.NumpyBackend`,
  always available (the reference).
* ``"numba"``: ``kerrray.raytracing.backends.numba_backend.NumbaBackend``
  (performance role), imported lazily so that this package imports cleanly
  without Numba; a missing module or a backend that reports itself
  unavailable becomes :class:`BackendUnavailable`.
* ``"cuda"``: named but never available. GPU acceleration is optional
  future work (docs/decisions.md D-007); no CUDA device is used and no GPU
  number is ever reported.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

from numpy.typing import ArrayLike

from kerrray.geodesics import EventOptions, IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult
from kerrray.raytracing.backends.numpy_backend import NumpyBackend
from kerrray.utils.config_blocks import SUPPORTED_BACKENDS

__all__ = [
    "BACKEND_NAMES",
    "CUDA_UNAVAILABLE_MESSAGE",
    "Backend",
    "BackendUnavailable",
    "available_backends",
    "get_backend",
]

BACKEND_NAMES: Final[tuple[str, ...]] = ("numpy", "numba", "cuda")
"""Backend names in the order :func:`available_backends` reports them."""

CUDA_UNAVAILABLE_MESSAGE: Final[str] = (
    "GPU acceleration is optional future work (docs/decisions.md D-007); "
    "no CUDA device is used"
)


class BackendUnavailable(RuntimeError):
    """Raised by :func:`get_backend` when a named backend cannot be used here."""


@runtime_checkable
class Backend(Protocol):
    """A batched integrator with the interface of :func:`kerrray.geodesics.integrate_batch`."""

    name: str

    def integrate_batch(
        self,
        st: Spacetime,
        Y0: ArrayLike,
        integ: IntegratorOptions,
        term: TerminationOptions,
        events: EventOptions | None = None,
    ) -> BatchResult:
        """Integrate the states ``Y0`` (N, 8) and return a :class:`BatchResult`."""
        ...


def _numba_backend() -> Backend:
    """Import and instantiate the Numba backend, mapping every failure to :class:`BackendUnavailable`."""
    try:
        from kerrray.raytracing.backends.numba_backend import NumbaBackend
    except ImportError as exc:
        raise BackendUnavailable(
            f"backend 'numba' is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        return NumbaBackend()
    except BackendUnavailable as exc:
        raise BackendUnavailable(f"backend 'numba' is unavailable: {exc}") from exc


def get_backend(name: str) -> Backend:
    """Return the backend called ``name`` (``"numpy"`` or ``"numba"``).

    Raises:
        BackendUnavailable: For ``"cuda"`` (always, with
            :data:`CUDA_UNAVAILABLE_MESSAGE`) and for ``"numba"`` when its
            module cannot be imported or reports itself unavailable; the
            message carries the reason.
        ValueError: For a name outside :data:`BACKEND_NAMES`.
    """
    if name == "numpy":
        return NumpyBackend()
    if name == "numba":
        return _numba_backend()
    if name == "cuda":
        raise BackendUnavailable(CUDA_UNAVAILABLE_MESSAGE)
    raise ValueError(f"unknown backend {name!r}; expected one of {sorted(SUPPORTED_BACKENDS)}")


def available_backends() -> dict[str, bool]:
    """``{"numpy": True, "numba": <importable>, "cuda": False}`` (docs/architecture.md section 5)."""
    try:
        _numba_backend()
        numba_ok = True
    except BackendUnavailable:
        numba_ok = False
    return {"numpy": True, "numba": numba_ok, "cuda": False}
