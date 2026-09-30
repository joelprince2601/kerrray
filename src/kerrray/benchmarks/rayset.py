"""Fixed benchmark ray set, DOP853 reference solutions and the trajectory-error metric.

Shared by the solver comparison (:mod:`kerrray.benchmarks.solver`, PROJECT.md
section 19) and the step-size study (:mod:`kerrray.experiments.step_size`,
section 20).

**Ray set.** :func:`build_ray_set` returns ``n_rays`` inward-moving photons
launched from ``launch_radius``. Ray ``k`` cycles deterministically through
four *kinds* (``capture``, ``escape``, ``near_critical_out``,
``near_critical_in``), the configured spins (Schwarzschild and Kerr) and the
equatorial / off-equatorial families, so that any prefix of the set covers
all categories; the magnitudes are drawn from the seeded generator that
``experiment.seed`` supplies. Every ray is built from exact constants of
motion, so its true fate is fixed by the critical curve, not by the launch
radius:

* equatorial rays: ``theta = pi/2``, ``Q = 0``, impact parameter
  ``b = f b_c`` with ``b_c`` the computed prograde or retrograde critical
  impact parameter (:func:`kerrray.photons.orbits.critical_impact_parameters`)
  and ``f < 1`` (capture), ``f > 1`` (escape) or ``f = 1 -+ delta`` with
  ``delta = 10^U(-4, -2)`` (near-critical);
* off-equatorial rays: launched at the observer inclination ``theta_o``
  with ``(alpha, beta) = f (alpha_c, beta_c)``, a random point of the
  analytic shadow curve (:func:`kerrray.photons.orbits.shadow_curve`) scaled
  by the same factors; the constants follow from Bardeen's relations
  ``xi = -alpha sin(theta_o)`` and ``eta = beta^2 - a^2 cos^2(theta_o) +
  xi^2 cot^2(theta_o)`` (docs/architecture.md section 1.3 inverted), which
  give ``Theta(theta_o) = beta^2 >= 0`` identically. The scaling acts in the
  ``(xi, eta)`` plane, so ``f < 1`` is inside the critical curve (captured)
  and ``f > 1`` outside (escapes), exactly and independently of the launch
  radius.

**Reference.** :func:`reference_solution` integrates a ray with
``scipy.integrate.solve_ivp(method="DOP853")`` (Hairer's eighth-order
Dormand-Prince code with its seventh-order dense output; Hairer, Norsett and
Wanner 1993, section II.5 and the DOP853 code description) at the reference
tolerances, with the same terminal events as the ``dop853`` path of
:mod:`kerrray.geodesics.integrators` (capture at ``r_+ + horizon_epsilon``,
escape at ``escape_radius`` with outward motion, the polar-angle domain). The
dense output is what allows a comparison *at matched affine parameter*
without interpolating the solver's own points; :meth:`ReferenceSolution.extend`
continues an escaped reference beyond its escape event so that a solver that
overshoots the escape sphere by a large last step is still covered.

**Trajectory error** (:func:`trajectory_error`): the Euclidean distance, in
units of ``M``, between the plotting Cartesian embeddings
(:func:`kerrray.geometry.bl_to_cartesian`) of the solver state and the
reference state at the solver's final affine parameter. Comparing at the
end state with matched ``lambda`` combines the radial, polar and azimuthal
deviations with their natural scale factors and makes a phase error at the
escape sphere show up as ``r_escape * delta_phi``, the quantity that matters
for imaging. Approximation, PROJECT.md section 43: WHAT, captured rays are
compared at ``r = r_+ + PLUNGE_MARGIN`` (the solver is re-run with that
capture radius) instead of at ``r_+ + horizon_epsilon``; WHY, in
Boyer-Lindquist coordinates ``phi`` diverges logarithmically at the horizon
for ``a != 0`` (``dphi/dlambda ~ 1/Delta``), so a radial error ``delta r``
at ``r - r_+ = 1e-6`` is amplified into an azimuthal error of order
``a delta r / [(r_+ - r_-)(r - r_+)]``, a property of the coordinates and
not of the solver; LIMITATION, the plunge from ``r_+ + 0.1 M`` to the
horizon is excluded from the trajectory error (it is still covered by the
null error, the drifts and the failure rate of the production run).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from kerrray.geodesics import (
    IDX_PH,
    IDX_PPH,
    IDX_R,
    IDX_TH,
    TerminationOptions,
    equatorial_photon,
    geodesic_rhs,
    photon_from_constants,
)
from kerrray.geometry import Spacetime, bl_to_cartesian, outer_horizon
from kerrray.photons import TerminationState
from kerrray.photons.classification import THETA_DOMAIN_TOLERANCE
from kerrray.photons.orbits import critical_impact_parameters, shadow_curve

__all__ = [
    "CAPTURE_SCALE",
    "ESCAPE_SCALE",
    "NEAR_CRITICAL_LOG10_RANGE",
    "PLUNGE_MARGIN",
    "RAY_KINDS",
    "BenchmarkRay",
    "ReferenceSolution",
    "build_ray_set",
    "comparison_termination",
    "position_error",
    "reference_solution",
    "trajectory_error",
]

PLUNGE_MARGIN: Final[float] = 0.1
"""Captured rays are compared with the reference at ``r = r_+ + PLUNGE_MARGIN`` (units of M)."""

RAY_KINDS: Final[tuple[str, ...]] = ("capture", "escape", "near_critical_out", "near_critical_in")
CAPTURE_SCALE: Final[tuple[float, float]] = (0.2, 0.9)
"""``b / b_c`` (or the celestial-plane scale factor) of captured rays, drawn uniformly."""
ESCAPE_SCALE: Final[tuple[float, float]] = (1.1, 3.0)
"""Scale factor of escaping rays, drawn uniformly (capped so that ``b < 0.8 r_0``)."""
NEAR_CRITICAL_LOG10_RANGE: Final[tuple[float, float]] = (-4.0, -2.0)
"""``log10`` of the relative offset ``delta`` of near-critical rays, drawn uniformly."""
SHADOW_CURVE_POINTS: Final[int] = 360
LAUNCH_RADIUS_FRACTION: Final[float] = 0.8
"""An escaping ray's impact parameter never exceeds this fraction of the launch radius."""
FAILURE_STATES: Final[frozenset[int]] = frozenset(
    {int(TerminationState.NUMERICAL_FAILURE), int(TerminationState.OUT_OF_DOMAIN)}
)
"""Termination codes counted as solver failures (PROJECT.md section 19)."""


@dataclass(frozen=True)
class BenchmarkRay:
    """One ray of the fixed set (see the module docstring for the construction)."""

    index: int
    spacetime: Spacetime
    kind: str
    equatorial: bool
    scale: float
    prograde: bool
    y0: NDArray[np.float64]

    @property
    def spin(self) -> float:
        """Dimensionless spin of the ray's spacetime."""
        return self.spacetime.spin

    @property
    def label(self) -> str:
        """Short text such as ``a=0.9/eq/near_critical_in/pro``."""
        family = "eq" if self.equatorial else "off"
        sense = "pro" if self.prograde else "retro"
        return f"a={self.spin:g}/{family}/{self.kind}/{sense}"


def _scale_factor(kind: str, rng: np.random.Generator, f_escape_max: float) -> float:
    if kind == "capture":
        return float(rng.uniform(*CAPTURE_SCALE))
    if kind == "escape":
        return float(rng.uniform(ESCAPE_SCALE[0], max(ESCAPE_SCALE[0], min(ESCAPE_SCALE[1], f_escape_max))))
    delta = 10.0 ** float(rng.uniform(*NEAR_CRITICAL_LOG10_RANGE))
    return 1.0 + delta if kind == "near_critical_out" else 1.0 - delta


def _off_equatorial_state(
    st: Spacetime, r0: float, theta_o: float, alpha: float, beta: float, sign_theta: int
) -> NDArray[np.float64]:
    """Photon with the constants of celestial point ``(alpha, beta)`` at inclination ``theta_o``."""
    sin_o, cos_o = math.sin(theta_o), math.cos(theta_o)
    xi = -alpha * sin_o
    eta = beta * beta - st.a**2 * cos_o**2 + xi * xi * (cos_o / sin_o) ** 2
    return photon_from_constants(st, r0, theta_o, 1.0, xi, eta, sign_r=-1, sign_theta=sign_theta)


def build_ray_set(
    n_rays: int,
    spins: list[float] | tuple[float, ...],
    *,
    mass: float,
    launch_radius: float,
    inclination_deg: float,
    rng: np.random.Generator,
) -> list[BenchmarkRay]:
    """Build the fixed ray set (module docstring).

    Ray ``k`` has kind ``RAY_KINDS[k % 4]``, spin ``spins[(k // 2) % len(spins)]``
    and is equatorial when ``(k // 4) % 2 == 0``; with the default two spins
    every block of eight rays covers all kinds, both spins and both families.
    The launch radius must exceed ``ESCAPE_SCALE[0] * b_c / LAUNCH_RADIUS_FRACTION``
    (about 10 M) so that escaping rays can be launched inward.
    """
    if n_rays < 1:
        raise ValueError(f"n_rays must be >= 1, got {n_rays!r}")
    if not spins:
        raise ValueError("spins must not be empty")
    spacetimes = [Spacetime(mass=mass, spin=float(s)) for s in spins]
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rays: list[BenchmarkRay] = []
    theta_o = math.radians(inclination_deg)
    for k in range(n_rays):
        kind = RAY_KINDS[k % len(RAY_KINDS)]
        spin_index = (k // 2) % len(spacetimes)
        st = spacetimes[spin_index]
        equatorial = (k // len(RAY_KINDS)) % 2 == 0
        prograde = bool(rng.random() < 0.5)
        b_pro, b_ret = critical_impact_parameters(st)
        b_c = b_pro if prograde else b_ret
        f_escape_max = LAUNCH_RADIUS_FRACTION * launch_radius / max(b_pro, b_ret)
        if f_escape_max < ESCAPE_SCALE[0]:
            raise ValueError(
                f"launch_radius = {launch_radius!r} M is too small for escaping rays at spin {st.spin:g} "
                f"(needs >= {ESCAPE_SCALE[0] * max(b_pro, b_ret) / LAUNCH_RADIUS_FRACTION:.3g} M)"
            )
        f = _scale_factor(kind, rng, f_escape_max)
        if equatorial:
            y0 = equatorial_photon(st, launch_radius, f * b_c, inward=True, prograde=prograde)
        else:
            if spin_index not in curves:
                curves[spin_index] = shadow_curve(st, inclination_deg, SHADOW_CURVE_POINTS)
            alpha_c, beta_c = curves[spin_index]
            j = int(rng.integers(alpha_c.size))
            sign_theta = 1 if rng.random() < 0.5 else -1
            y0 = _off_equatorial_state(
                st, launch_radius, theta_o, float(f * alpha_c[j]), float(f * beta_c[j]), sign_theta
            )
            axis = st.a if st.a != 0.0 else 1.0
            prograde = bool(axis * y0[IDX_PPH] > 0.0)
        rays.append(BenchmarkRay(k, st, kind, equatorial, f, prograde, y0))
    return rays


def comparison_termination(term: TerminationOptions) -> TerminationOptions:
    """Termination used for the trajectory-error runs: capture at ``r_+ + PLUNGE_MARGIN``."""
    return TerminationOptions(
        horizon_epsilon=max(term.horizon_epsilon, PLUNGE_MARGIN), escape_radius=term.escape_radius
    )


@dataclass
class ReferenceSolution:
    """DOP853 dense-output reference for one ray (see :func:`reference_solution`).

    Attributes:
        spacetime: The spacetime of the ray.
        rtol, atol: Reference tolerances (reused by :meth:`extend`).
        state: Termination state of the reference run (production rules).
        lam_end: Affine parameter of the termination point.
        y_end: State at the termination point.
        n_steps: Accepted DOP853 steps of the first segment.
        runtime_s: Wall-clock seconds spent on the first segment.
        segments: ``(lam_stop, dense_solution)`` pairs; segment ``i`` covers
            affine parameters from the previous stop up to ``lam_stop``.
    """

    spacetime: Spacetime
    rtol: float
    atol: float
    state: TerminationState
    lam_end: float
    y_end: NDArray[np.float64]
    n_steps: int
    runtime_s: float
    segments: list[tuple[float, Any]] = field(default_factory=list)

    @property
    def lam_covered(self) -> float:
        """Largest affine parameter at which :meth:`at` is valid."""
        return self.segments[-1][0]

    def covers(self, lam: float) -> bool:
        """Whether ``lam`` lies inside the dense-output range."""
        return 0.0 <= lam <= self.lam_covered * (1.0 + 1e-12)

    def at(self, lam: float) -> NDArray[np.float64]:
        """Reference state at affine parameter ``lam`` (raises outside the covered range)."""
        if not self.covers(lam):
            raise ValueError(f"lambda = {lam!r} is outside the reference range [0, {self.lam_covered!r}]")
        for lam_stop, sol in self.segments:
            if lam <= lam_stop * (1.0 + 1e-12):
                return np.asarray(sol(min(lam, lam_stop)), dtype=np.float64)
        raise AssertionError("unreachable")  # pragma: no cover

    def extend(self, lam_needed: float) -> None:
        """Continue an ``ESCAPED`` reference (no escape event) up to ``lam_needed``."""
        if lam_needed <= self.lam_covered or self.state != TerminationState.ESCAPED:
            return
        lam_stop, sol = self.segments[-1]
        y_start = np.asarray(sol(lam_stop), dtype=np.float64)
        st = self.spacetime
        res = solve_ivp(
            lambda _l, y: geodesic_rhs(st, y),
            (lam_stop, lam_needed),
            y_start,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
        )
        if res.status != 0:
            raise RuntimeError(f"reference extension failed: {res.message}")
        self.segments.append((float(res.t[-1]), res.sol))


def reference_solution(
    st: Spacetime,
    y0: NDArray[np.float64],
    term: TerminationOptions,
    *,
    rtol: float,
    atol: float,
    lambda_max: float,
    first_step: float,
) -> ReferenceSolution:
    """Integrate ``y0`` with DOP853 dense output under the production termination rules."""
    r_cap = outer_horizon(st) + term.horizon_epsilon

    def fun(_lam: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return geodesic_rhs(st, y)

    def captured(_lam: float, y: NDArray[np.float64]) -> float:
        return float(y[IDX_R] - r_cap)

    def escaped(_lam: float, y: NDArray[np.float64]) -> float:
        return float(y[IDX_R] - term.escape_radius)

    def theta_low(_lam: float, y: NDArray[np.float64]) -> float:
        return float(y[IDX_TH] + THETA_DOMAIN_TOLERANCE)

    def theta_high(_lam: float, y: NDArray[np.float64]) -> float:
        return float(math.pi + THETA_DOMAIN_TOLERANCE - y[IDX_TH])

    events = [captured, escaped, theta_low, theta_high]
    outcomes = [
        TerminationState.CAPTURED,
        TerminationState.ESCAPED,
        TerminationState.OUT_OF_DOMAIN,
        TerminationState.OUT_OF_DOMAIN,
    ]
    for ev, direction in zip(events, (-1, 1, -1, -1), strict=True):
        ev.terminal = True  # type: ignore[attr-defined]
        ev.direction = direction  # type: ignore[attr-defined]
    t0 = time.perf_counter()
    res = solve_ivp(
        fun,
        (0.0, lambda_max),
        np.asarray(y0, dtype=np.float64),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        first_step=min(first_step, lambda_max),
        events=events,
        dense_output=True,
    )
    runtime = time.perf_counter() - t0
    if res.status == 1:
        hit = [k for k, t_ev in enumerate(res.t_events) if len(t_ev) > 0]
        state = outcomes[hit[0]]
    elif res.status == 0:
        state = TerminationState.MAX_AFFINE_PARAMETER
    else:
        state = TerminationState.NUMERICAL_FAILURE
    lam_end = float(res.t[-1])
    return ReferenceSolution(
        spacetime=st,
        rtol=rtol,
        atol=atol,
        state=state,
        lam_end=lam_end,
        y_end=np.asarray(res.y[:, -1], dtype=np.float64),
        n_steps=int(len(res.t) - 1),
        runtime_s=runtime,
        segments=[(lam_end, res.sol)],
    )


def position_error(st: Spacetime, y_a: NDArray[np.floating], y_b: NDArray[np.floating]) -> float:
    """Euclidean distance (units of M) between the Cartesian embeddings of two states."""
    xa = np.array(bl_to_cartesian(st, y_a[IDX_R], y_a[IDX_TH], y_a[IDX_PH]), dtype=np.float64)
    xb = np.array(bl_to_cartesian(st, y_b[IDX_R], y_b[IDX_TH], y_b[IDX_PH]), dtype=np.float64)
    return float(np.sqrt(np.sum((xa - xb) ** 2)))


def trajectory_error(ref: ReferenceSolution, lam: float, y: NDArray[np.floating], state: int) -> float:
    """Trajectory error of a solver end state ``(lam, y)`` against ``ref`` (module docstring).

    Returns ``nan`` when the solver failed (``NUMERICAL_FAILURE`` or
    ``OUT_OF_DOMAIN``) or when ``lam`` lies beyond the reference range after
    an attempted extension (a solver that escapes where the reference is
    captured, or vice versa).
    """
    if int(state) in FAILURE_STATES or not np.all(np.isfinite(y)):
        return math.nan
    ref.extend(float(lam))
    if not ref.covers(float(lam)):
        return math.nan
    return position_error(ref.spacetime, np.asarray(y, dtype=np.float64), ref.at(float(lam)))
