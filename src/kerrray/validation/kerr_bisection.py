"""Critical impact parameter by bracketed classification search (PROJECT.md sections 12, 13).

For equatorial photons launched inward with impact parameter ``b = |L_z|/E``
the outcome is monotone in ``b``: the radial potential ``R(r)`` has a turning
point outside the horizon exactly when ``b`` exceeds the critical value
``b_c`` of the circular photon orbit (Bardeen, Press and Teukolsky 1972,
section II; docs/derivations.md section 6), so rays with ``b < b_c`` are
``CAPTURED`` and rays with ``b > b_c`` are ``ESCAPED``. The search keeps a
bracket ``[b_lo, b_hi]`` with ``CAPTURED`` at ``b_lo`` and ``ESCAPED`` at
``b_hi`` and shrinks it by classifying rays inside it.

With ``n_probe = 1`` this is the classical bisection (one ray at the
midpoint, the bracket halves). With ``n_probe > 1`` the ``n_probe`` interior
points are integrated in one vectorised batch
(:func:`kerrray.geodesics.integrate_batch`) and the bracket shrinks by
``n_probe + 1`` per iteration; the result is the same bracketing procedure
with fewer sequential integrations (``log_9`` instead of ``log_2`` of the
width reduction for ``n_probe = 8``). Prograde and retrograde searches of the
same spacetime share one batch. Any ray that ends in a state other than
``CAPTURED`` or ``ESCAPED``, or a non-monotone pattern (an ``ESCAPED`` probe
below a ``CAPTURED`` one), raises ``RuntimeError``: the validation fails
loudly rather than reporting a number it cannot trust.

Horizon crossings of fixed-step schemes (``horizon_crossing_as_capture``).
Approximation label (PROJECT.md section 43):

* WHAT: with ``horizon_crossing_as_capture=True`` a ray that ends
  ``OUT_OF_DOMAIN`` with a finite final radius ``r <= r_+ +
  horizon_epsilon`` counts as ``CAPTURED``.
* WHY: in Boyer-Lindquist coordinates ``p_r = -sqrt(R)/Delta`` diverges at
  the horizon. The adaptive scheme shrinks its step and lands inside the
  capture margin; the fixed-step RK4 scheme evaluates its stages across
  ``Delta = 0`` and the step throws ``r`` to large negative values, which the
  classifier (correctly) reports as ``OUT_OF_DOMAIN``. Measured with
  ``h = 0.2 ... 0.01`` M for a Schwarzschild ray with ``b = b_c - 0.01``: every
  run ends ``OUT_OF_DOMAIN`` with ``r < 0`` after its last in-domain point at
  ``r ~ 2.0-2.2 M`` (moving inward) except ``h = 0.01`` with a ``1e-2`` margin.
* LIMITATION: it relies on the discrete path having been inbound at
  ``r_+ + horizon_epsilon`` or below at its final state; a ray that fails
  anywhere else (``NUMERICAL_FAILURE``, or ``OUT_OF_DOMAIN`` at ``r >
  r_+``) still raises. The option is off by default and used only by the
  RK4 step-size study, which reports how many rays it relabelled.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate_batch, photon_from_constants
from kerrray.geometry import Spacetime, outer_horizon
from kerrray.photons import TerminationState

__all__ = ["CriticalImpactBisection", "bisect_critical_impacts", "probe_states"]


@dataclass(frozen=True)
class CriticalImpactBisection:
    """Result of one bracketing search.

    Attributes:
        spin, prograde: The spacetime spin and the sense of the search.
        b_c: Midpoint of the final bracket (units of M).
        b_lo, b_hi: Final bracket (``CAPTURED`` at ``b_lo``, ``ESCAPED`` at ``b_hi``).
        width: ``b_hi - b_lo``.
        iterations: Bracket-shrinking iterations performed.
        n_rays: Rays integrated (including the two endpoint checks).
        max_steps: Largest number of accepted steps of any ray.
        runtime_s: Wall-clock seconds of the search.
        horizon_crossings: Rays relabelled ``CAPTURED`` by
            ``horizon_crossing_as_capture`` (module docstring; 0 when off).
    """

    spin: float
    prograde: bool
    b_c: float
    b_lo: float
    b_hi: float
    width: float
    iterations: int
    n_rays: int
    max_steps: int
    runtime_s: float
    horizon_crossings: int = 0


def _lz_sign(st: Spacetime, prograde: bool) -> float:
    """``+1`` or ``-1`` such that ``a L_z > 0`` for prograde (``L_z > 0`` for ``a = 0``)."""
    return 1.0 if prograde == (st.a >= 0.0) else -1.0


def probe_states(
    st: Spacetime,
    bs: NDArray[np.float64],
    prograde: NDArray[np.bool_],
    r0: float,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    horizon_crossing_as_capture: bool = False,
) -> tuple[NDArray[np.int64], int, int]:
    """Classify inward equatorial rays with impact parameters ``bs`` in one batch.

    Returns the termination codes, the largest accepted-step count and the
    number of rays relabelled ``CAPTURED`` (module docstring).
    """
    y0 = np.stack(
        [
            photon_from_constants(st, r0, 0.5 * math.pi, 1.0, _lz_sign(st, bool(p)) * b, 0.0, sign_r=-1, sign_theta=1)
            for b, p in zip(bs, prograde, strict=True)
        ]
    )
    result = integrate_batch(st, y0, integ, term)
    states = np.array(result.state, dtype=np.int64, copy=True)
    n_crossed = 0
    if horizon_crossing_as_capture:
        r_end = np.asarray(result.Y[:, 1], dtype=np.float64)
        crossed = (states == TerminationState.OUT_OF_DOMAIN) & np.isfinite(r_end) & (
            r_end <= outer_horizon(st) + term.horizon_epsilon
        )
        states[crossed] = TerminationState.CAPTURED
        n_crossed = int(np.count_nonzero(crossed))
    return states, int(np.max(result.n_steps)), n_crossed


class _Search:
    """Mutable bracket of one search."""

    def __init__(self, prograde: bool, b_lo: float, b_hi: float) -> None:
        self.prograde = prograde
        self.lo = b_lo
        self.hi = b_hi
        self.iterations = 0

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def update(self, bs: NDArray[np.float64], states: NDArray[np.int64]) -> None:
        captured = bs[states == TerminationState.CAPTURED]
        escaped = bs[states == TerminationState.ESCAPED]
        if captured.size and escaped.size and captured.max() > escaped.min():
            raise RuntimeError(
                f"non-monotone classification in b for prograde={self.prograde}: captured up to "
                f"{captured.max()!r} but escaped from {escaped.min()!r}"
            )
        if captured.size:
            self.lo = max(self.lo, float(captured.max()))
        if escaped.size:
            self.hi = min(self.hi, float(escaped.min()))
        self.iterations += 1


def _check_states(bs: NDArray[np.float64], states: NDArray[np.int64]) -> None:
    bad = [(float(b), TerminationState(int(s)).name) for b, s in zip(bs, states, strict=True)
           if int(s) not in (TerminationState.CAPTURED, TerminationState.ESCAPED)]
    if bad:
        raise RuntimeError(f"rays ended neither CAPTURED nor ESCAPED (b, state): {bad}")


def bisect_critical_impacts(
    st: Spacetime,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    r0: float,
    b_lo: float,
    b_hi: float,
    tol: float,
    max_iterations: int,
    n_probe: int = 8,
    directions: tuple[bool, ...] = (True, False),
    horizon_crossing_as_capture: bool = False,
) -> tuple[CriticalImpactBisection, ...]:
    """Bracketing search for ``b_c`` in each of ``directions`` (prograde ``True``, retrograde ``False``).

    Args:
        st: The spacetime.
        integ, term: Integrator and termination options for every ray.
        r0: Launch radius (units of M); the rays start in the equatorial plane
            moving inward.
        b_lo, b_hi: Initial bracket; ``b_lo`` must be captured and ``b_hi``
            escaped (checked with two extra rays per direction).
        tol: Stop when the bracket width is ``<= tol``.
        max_iterations: Cap on bracket-shrinking iterations per direction.
        n_probe: Interior points per iteration (``1`` = classical bisection).
        directions: Senses to search, sharing one batch per iteration.
        horizon_crossing_as_capture: Relabel fixed-step horizon crossings
            (module docstring); off by default.

    Returns:
        One :class:`CriticalImpactBisection` per direction, in order.

    Raises:
        ValueError: On an invalid bracket, tolerance or probe count.
        RuntimeError: If a ray ends neither ``CAPTURED`` nor ``ESCAPED``, the
            bracket endpoints are misclassified, or the outcome is not
            monotone in ``b``.
    """
    if not (0.0 <= b_lo < b_hi):
        raise ValueError(f"need 0 <= b_lo < b_hi, got {b_lo!r}, {b_hi!r}")
    if not tol > 0.0:
        raise ValueError(f"tol must be > 0, got {tol!r}")
    if n_probe < 1 or max_iterations < 1:
        raise ValueError("n_probe and max_iterations must be >= 1")
    t0 = time.perf_counter()
    searches = [_Search(p, b_lo, b_hi) for p in directions]
    n_rays = 0
    max_steps = 0
    # Endpoint check: b_lo captured, b_hi escaped, for every direction.
    bs = np.array([b_lo, b_hi] * len(searches))
    pro = np.array([p for s in searches for p in (s.prograde, s.prograde)])
    states, steps, crossed = probe_states(
        st, bs, pro, r0, integ, term, horizon_crossing_as_capture=horizon_crossing_as_capture
    )
    n_rays += bs.size
    n_crossed = crossed
    max_steps = max(max_steps, steps)
    for k, s in enumerate(searches):
        lo_state, hi_state = TerminationState(int(states[2 * k])), TerminationState(int(states[2 * k + 1]))
        if lo_state != TerminationState.CAPTURED or hi_state != TerminationState.ESCAPED:
            raise RuntimeError(
                f"bracket [{b_lo}, {b_hi}] invalid for prograde={s.prograde}: "
                f"{lo_state.name} at b_lo, {hi_state.name} at b_hi"
            )
    fractions = np.arange(1, n_probe + 1) / (n_probe + 1)
    while True:
        open_searches = [s for s in searches if s.width > tol and s.iterations < max_iterations]
        if not open_searches:
            break
        bs = np.concatenate([s.lo + s.width * fractions for s in open_searches])
        pro = np.concatenate([np.full(n_probe, s.prograde) for s in open_searches])
        states, steps, crossed = probe_states(
            st, bs, pro, r0, integ, term, horizon_crossing_as_capture=horizon_crossing_as_capture
        )
        _check_states(bs, states)
        n_rays += bs.size
        n_crossed += crossed
        max_steps = max(max_steps, steps)
        for k, s in enumerate(open_searches):
            sl = slice(k * n_probe, (k + 1) * n_probe)
            s.update(bs[sl], states[sl])
    runtime = time.perf_counter() - t0
    return tuple(
        CriticalImpactBisection(
            spin=st.spin,
            prograde=s.prograde,
            b_c=0.5 * (s.lo + s.hi),
            b_lo=s.lo,
            b_hi=s.hi,
            width=s.width,
            iterations=s.iterations,
            n_rays=n_rays,
            max_steps=max_steps,
            runtime_s=runtime,
            horizon_crossings=n_crossed,
        )
        for s in searches
    )
