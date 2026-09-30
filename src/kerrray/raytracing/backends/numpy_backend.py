"""Reference CPU backend: the vectorised NumPy batch integrator.

:class:`NumpyBackend` wraps :func:`kerrray.geodesics.integrate_batch` (the
Hamiltonian right-hand side with the per-ray adaptive Dormand-Prince 5(4) or
fixed-step RK4 schemes of :mod:`kerrray.geodesics.tableaus`) behind the
:class:`kerrray.raytracing.backends.base.Backend` protocol. It is the
reference every other backend must agree with to floating-point round-off
(docs/architecture.md section 9, docs/decisions.md D-007).
"""

from __future__ import annotations

from typing import Final

from numpy.typing import ArrayLike

from kerrray.geodesics import EventOptions, IntegratorOptions, TerminationOptions, integrate_batch
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult

__all__ = ["NumpyBackend"]


class NumpyBackend:
    """The NumPy reference backend (``name == "numpy"``).

    Stateless: every call goes straight to
    :func:`kerrray.geodesics.integrate_batch`, so results are identical to a
    direct call of the integrator.
    """

    name: Final[str] = "numpy"

    def integrate_batch(
        self,
        st: Spacetime,
        Y0: ArrayLike,
        integ: IntegratorOptions,
        term: TerminationOptions,
        events: EventOptions | None = None,
    ) -> BatchResult:
        """Integrate the states ``Y0`` (N, 8); see :func:`kerrray.geodesics.integrate_batch`."""
        return integrate_batch(st, Y0, integ, term, events=events)

    def __repr__(self) -> str:
        return "NumpyBackend()"
