"""Observer camera, ZAMO tetrad and backward-ray initial states.

Implements PROJECT.md sections 15 and 16 and docs/architecture.md sections
1.3 and 5 (camera role). The derivation, the sign conventions, the
finite-distance correction and the limitations are written out in
``docs/raytracing.md``; this docstring states what is implemented.

Observer frame
--------------
The observer is a zero-angular-momentum observer (ZAMO; the "locally
non-rotating frame" of Bardeen, Press and Teukolsky 1972, ApJ 178, 347,
section III; Frolov and Novikov 1998, *Black Hole Physics*, section 3.3) at
Boyer-Lindquist ``(r_o, theta_o, phi_o)``. Writing the ``(t, phi)`` block of
the metric in the BPT form ``-e^{2 nu} dt^2 + e^{2 psi} (dphi - omega dt)^2``
(BPT eq. 2.1) and matching coefficients with the Boyer-Lindquist components
``g_tt = -e^{2 nu} + omega^2 e^{2 psi}``, ``g_tphi = -omega e^{2 psi}``,
``g_phph = e^{2 psi}`` gives::

    omega   = -g_tphi / g_phph                       (frame-dragging angular velocity)
    lapse^2 = e^{2 nu} = -(g_tt - g_tphi^2 / g_phph) = Sigma Delta / A

and the orthonormal tetrad, contravariant components as the rows of
:func:`zamo_tetrad`::

    e_(t)     = (1 / lapse) (1, 0, 0, omega)
    e_(r)     = (0, 1 / sqrt(g_rr), 0, 0)
    e_(theta) = (0, 0, 1 / sqrt(g_thth), 0)
    e_(phi)   = (0, 0, 0, 1 / sqrt(g_phph))

Orthonormality ``e_(a)^mu e_(b)^nu g_mu_nu = eta_ab`` (``eta = diag(-1, 1, 1,
1)``) follows from ``g_tphi + omega g_phph = 0`` and is verified numerically
in ``tests/test_camera.py`` together with ``lapse^2 = -1 / g^tt`` and
``omega = g^tphi / g^tt``. The tetrad is singular on the axis (``g_phph = 0``)
and on the horizon (``Delta = 0``): both are rejected.

Image plane and photon momentum
-------------------------------
Pixel ``(i, j)`` of the square ``resolution x resolution`` image has celestial
coordinates ``(alpha, beta)`` in units of ``M`` with ``alpha`` increasing to
the right and ``beta`` decreasing down the rows (row 0 is the top, ``beta >
0``); pixel centres carry half-pixel offsets and the resolution is even, so
``alpha = 0`` and ``beta = 0`` are never sampled. ``fov`` is the half-width of
the image plane at the outer pixel edge. For a photon arriving at the
observer from the hole side the local tetrad components are (docs/
architecture.md section 1.3, Cunha and Herdeiro 2018, Gen. Rel. Grav. 50, 42)::

    p^(t) = 1,  p^(phi) = -alpha / r_o,  p^(theta) = beta / r_o,
    p^(r) = +sqrt(1 - (alpha^2 + beta^2) / r_o^2)

i.e. ``alpha = -r_o p^(phi) / p^(t)`` and ``beta = r_o p^(theta) / p^(t)``,
which reduce to Bardeen's ``alpha = -xi / sin(theta_o)``, ``beta = p_theta /
E`` as ``r_o -> infinity``. The coordinate momentum is ``p^mu = p^(a)
e_(a)^mu`` and ``p_mu = g_mu_nu p^nu``; it is exactly null because ``eta_ab
p^(a) p^(b) = 0``. For backward tracing the covariant momentum is reversed,
``p_mu -> -p_mu`` (exact: the geodesic equation is invariant under
``(lambda, p) -> (-lambda, -p)``), so the returned states have ``E = -p_t < 0``
and ``p_r < 0`` (inward in the integration direction); the ratios ``xi = L_z /
E`` and ``eta = Q / E^2`` are unchanged by the reversal. The null condition is
then re-imposed numerically by solving the quadratic
``g^{mu nu} p_mu p_nu = 0`` for ``p_t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from kerrray.geometry import (
    Array,
    InverseMetricComponents,
    Spacetime,
    ergosphere_radius,
    inverse_metric_components,
    metric,
    metric_components,
    outer_horizon,
)
from kerrray.utils.config import INCLINATION_EPSILON_DEG, ObserverConfig, RaytraceConfig
from kerrray.utils.logging import get_logger

__all__ = ["NULL_CHECK_TOLERANCE", "Camera", "initial_states", "zamo_tetrad"]

logger = get_logger(__name__)

NULL_CHECK_TOLERANCE: Final[float] = 1e-10
"""Largest ``abs(g^{mu nu} p_mu p_nu)`` accepted after the tetrad projection.

The local components are normalised to ``p^(t) = 1``, so the residual of the
exactly null local momentum is pure round-off (order 1e-16). A larger value
can only come from a broken tetrad or metric, so :func:`initial_states`
raises instead of hiding the defect behind the re-normalisation step.
"""


@dataclass(frozen=True)
class Camera:
    """Pinhole camera of a ZAMO at ``(radius, inclination_deg, phi_deg)``.

    Attributes:
        radius: Boyer-Lindquist radius ``r_o`` of the observer in units of
            ``M`` (finite, > 0). It must exceed the outer horizon of the
            spacetime the camera is used with (checked in :func:`initial_states`).
        inclination_deg: Boyer-Lindquist polar angle ``theta_o`` of the
            observer in degrees (docs/decisions.md D-001): 0 on the spin
            axis, 90 in the equatorial plane. Values are clamped into
            ``[INCLINATION_EPSILON_DEG, 180 - INCLINATION_EPSILON_DEG]`` with
            a logged warning (D-008). Approximation label: WHAT, the
            observer is moved off the axis by at most 1e-3 degrees; WHY,
            the ZAMO tetrad is singular on the axis (``g_phph = 0``);
            LIMITATION, an "on-axis" image is the image seen from
            ``theta_o = 1e-3`` degrees and is axisymmetric only to that
            accuracy.
        phi_deg: Observer azimuth ``phi_o`` in degrees (any finite value).
        fov: Half-width of the image plane in units of ``M``: ``abs(alpha)``
            and ``abs(beta)`` at the outer edge of the outermost pixels.
        resolution: Pixels per side of the square image; an even integer
            ``>= 2`` so that the half-pixel-offset centres never sample
            ``alpha = 0`` or ``beta = 0``.
    """

    radius: float
    inclination_deg: float
    phi_deg: float = 0.0
    fov: float = 12.0
    resolution: int = 64

    def __post_init__(self) -> None:
        radius = float(self.radius)
        inclination = float(self.inclination_deg)
        phi = float(self.phi_deg)
        fov = float(self.fov)
        if not (math.isfinite(radius) and radius > 0.0):
            raise ValueError(f"radius must be a finite number > 0, got {self.radius!r}")
        if not (math.isfinite(inclination) and 0.0 <= inclination <= 180.0):
            raise ValueError(
                f"inclination_deg must lie in [0, 180], got {self.inclination_deg!r}"
            )
        if not math.isfinite(phi):
            raise ValueError(f"phi_deg must be finite, got {self.phi_deg!r}")
        if not (math.isfinite(fov) and fov > 0.0):
            raise ValueError(f"fov must be a finite number > 0, got {self.fov!r}")
        resolution = self.resolution
        if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
            raise ValueError(f"resolution must be an integer, got {resolution!r}")
        resolution = int(resolution)
        if resolution < 2 or resolution % 2 != 0:
            raise ValueError(
                f"resolution must be an even integer >= 2 so that alpha = 0 is never "
                f"sampled (docs/architecture.md section 1.3), got {resolution!r}"
            )
        low, high = INCLINATION_EPSILON_DEG, 180.0 - INCLINATION_EPSILON_DEG
        clamped = min(max(inclination, low), high)
        if clamped != inclination:
            logger.warning(
                "Camera inclination_deg = %g lies on the spin axis where the ZAMO tetrad "
                "is singular; clamped to %g deg (docs/decisions.md, D-008)",
                inclination,
                clamped,
            )
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "inclination_deg", clamped)
        object.__setattr__(self, "phi_deg", phi)
        object.__setattr__(self, "fov", fov)
        object.__setattr__(self, "resolution", resolution)

    @classmethod
    def from_config(cls, observer: ObserverConfig, raytrace: RaytraceConfig) -> Camera:
        """Build the camera from the ``observer`` and ``raytrace`` configuration blocks."""
        return cls(
            radius=observer.radius,
            inclination_deg=observer.inclination_deg,
            phi_deg=observer.phi_deg,
            fov=raytrace.fov,
            resolution=raytrace.resolution,
        )

    @property
    def n_rays(self) -> int:
        """Number of pixels, ``resolution ** 2`` (one ray per pixel)."""
        return self.resolution * self.resolution

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(H, W) = (resolution, resolution)``."""
        return (self.resolution, self.resolution)

    @property
    def pixel_size(self) -> float:
        """Pixel side ``2 fov / resolution`` in units of ``M``."""
        return 2.0 * self.fov / self.resolution

    @property
    def inclination_rad(self) -> float:
        """The (clamped) observer polar angle ``theta_o`` in radians."""
        return math.radians(self.inclination_deg)

    @property
    def phi_rad(self) -> float:
        """The observer azimuth ``phi_o`` in radians."""
        return math.radians(self.phi_deg)

    def pixel_coordinates(self) -> tuple[Array, Array]:
        """Celestial coordinates ``(alpha, beta)`` of the pixel centres, shape ``(H, W)``.

        With ``n = resolution`` and pixel size ``d = 2 fov / n`` the centres
        are ``alpha_j = -fov + (j + 1/2) d = fov (2 j + 1 - n) / n`` for column
        ``j`` and ``beta_i = fov - (i + 1/2) d = -fov (2 i + 1 - n) / n`` for
        row ``i``: ``alpha`` increases to the right, ``beta`` decreases down
        the rows, row 0 is the top (``beta > 0``) and the outer pixel edges lie
        at ``+-fov``. The integer form makes ``alpha[:, ::-1] == -alpha`` and
        ``beta[::-1, :] == -beta`` hold exactly in floating point, and for even
        ``n`` the value ``0`` never occurs (``2 j + 1 - n`` is odd). The arrays
        display correctly with ``matplotlib.pyplot.imshow(..., origin="upper")``.
        """
        n = self.resolution
        centres = self.fov * (2.0 * np.arange(n, dtype=np.float64) + 1.0 - n) / n
        alpha, beta = np.meshgrid(centres, -centres)  # (H, W): alpha[i, j] = centres[j]
        return alpha, beta


def zamo_tetrad(st: Spacetime, r: ArrayLike, theta: ArrayLike) -> Array:
    """Orthonormal ZAMO tetrad ``e_(a)^mu`` as an array of shape ``(..., 4, 4)``.

    Row ``a`` (``0 = t, 1 = r, 2 = theta, 3 = phi``) holds the contravariant
    Boyer-Lindquist components of ``e_(a)``; see the module docstring for the
    derivation (BPT 1972 section III; Frolov and Novikov 1998 section 3.3).
    ``r`` and ``theta`` broadcast; scalars give a plain ``(4, 4)`` array.

    Raises:
        ValueError: If any point lies on or inside the outer horizon
            (``r <= r_plus``), on the axis or outside ``0 < theta < pi``, or is
            non-finite; the tetrad does not exist there.
    """
    r_ = np.asarray(r, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    if not (np.all(np.isfinite(r_)) and np.all(np.isfinite(th))):
        raise ValueError("r and theta must be finite")
    if np.any(r_ <= outer_horizon(st)):
        raise ValueError(
            f"the ZAMO tetrad exists only outside the outer horizon r_plus = "
            f"{outer_horizon(st):.6g}; got r <= r_plus"
        )
    if np.any(th <= 0.0) or np.any(th >= math.pi):
        raise ValueError("the ZAMO tetrad is singular on the spin axis; need 0 < theta < pi")
    g = metric_components(st, r_, th)
    omega = -g.g_tphi / g.g_phph
    lapse2 = -(g.g_tt - g.g_tphi**2 / g.g_phph)
    if not np.all(lapse2 > 0.0):
        raise ValueError("lapse^2 = Sigma Delta / A is not positive: point is not outside the horizon")
    lapse = np.sqrt(lapse2)
    shape = np.broadcast(g.g_tt, g.g_tphi, g.g_rr, g.g_thth, g.g_phph).shape
    e = np.zeros(shape + (4, 4), dtype=np.float64)
    e[..., 0, 0] = 1.0 / lapse
    e[..., 0, 3] = omega / lapse
    e[..., 1, 1] = 1.0 / np.sqrt(g.g_rr)
    e[..., 2, 2] = 1.0 / np.sqrt(g.g_thth)
    e[..., 3, 3] = 1.0 / np.sqrt(g.g_phph)
    return e


def _null_pt(gi: InverseMetricComponents, p: Array) -> Array:
    """Solve ``g^{mu nu} p_mu p_nu = 0`` for ``p_t``, returning the root nearest ``p[:, 0]``.

    The quadratic ``A p_t^2 + B p_t + C = 0`` with ``A = g^tt``, ``B = 2 g^tphi
    p_phi`` and ``C = g^rr p_r^2 + g^thth p_theta^2 + g^phph p_phi^2`` has
    ``A < 0`` outside the horizon. Outside the ergosphere ``g^phph > 0`` too,
    so ``C > 0`` and the two roots have opposite signs (the negative one is
    the physical, future-directed photon with ``E = -p_t > 0``, the positive
    one its reversal); inside the ergosphere ``g^phph`` can be negative and
    both roots may share a sign (negative-energy photons), so the sign cannot
    select the branch. The supplied ``p_t`` is already null to round-off, so
    the root nearest to it is the one it belongs to in every case, and the
    re-normalisation only removes round-off. The roots are evaluated with the
    cancellation-free form ``q = -(B + sign(B) sqrt(B^2 - 4 A C)) / 2``,
    ``p_t = q / A`` and ``C / q`` (Press et al., *Numerical Recipes*, section
    5.6). A discriminant that is negative beyond round-off means no null
    momentum exists with the given spatial components and raises.
    """
    p_t, p_r, p_th, p_ph = (p[:, k] for k in range(4))
    a = np.broadcast_to(np.asarray(gi.gtt, dtype=np.float64), p_t.shape)
    b = 2.0 * gi.gtphi * p_ph
    c = gi.grr * p_r**2 + gi.gthth * p_th**2 + gi.gphph * p_ph**2
    if np.any(a >= 0.0):
        raise RuntimeError("null re-normalisation needs g^tt < 0 (a point outside the horizon)")
    disc = b * b - 4.0 * a * c
    if np.any(disc < -NULL_CHECK_TOLERANCE * (b * b + np.abs(4.0 * a * c))):
        raise RuntimeError("no null momentum exists with the given spatial components")
    q = -0.5 * (b + np.copysign(np.sqrt(np.maximum(disc, 0.0)), b))
    root_1 = q / a
    root_2 = np.divide(c, q, out=root_1.copy(), where=q != 0.0)  # q = 0 only for the double root 0
    return np.where(np.abs(root_1 - p_t) <= np.abs(root_2 - p_t), root_1, root_2)


def initial_states(cam: Camera, st: Spacetime, *, reverse: bool = True) -> Array:
    """Null photon states for every pixel, shape ``(N, 8)``, row-major over ``(H, W)``.

    Each row is ``[t, r, theta, phi, p_t, p_r, p_theta, p_phi]`` (docs/
    architecture.md section 1) with ``t = 0``, the observer position, and the
    momentum built from the pixel's celestial coordinates as described in the
    module docstring. Row ``i * W + j`` belongs to pixel ``(i, j)`` of
    :meth:`Camera.pixel_coordinates`.

    Args:
        cam: The camera (observer position, field of view, resolution).
        st: The spacetime.
        reverse: ``True`` (the default, what the ray tracer integrates)
            reverses ``p_mu`` for backward tracing, so ``E = -p_t < 0`` and
            ``p_r < 0``: the ray moves inward in the integration direction and
            the ESCAPED test (``dr/dlambda > 0``) is never met at launch
            (D-002). ``False`` returns the physical, future-directed photon
            (``E > 0``, ``p_r > 0``) for diagnostics and tests.

    Raises:
        ValueError: If the camera radius is not outside the outer horizon or
            if a pixel has ``alpha^2 + beta^2 >= r_o^2`` (no such direction
            exists on the observer's sky; reduce ``fov`` below
            ``radius / sqrt(2)``).
        RuntimeError: If the tetrad projection does not produce a null
            momentum to :data:`NULL_CHECK_TOLERANCE` (internal defect).
    """
    r_o, th_o, ph_o = cam.radius, cam.inclination_rad, cam.phi_rad
    if r_o <= outer_horizon(st):
        raise ValueError(
            f"camera radius {r_o:.6g} must exceed the outer horizon r_plus = "
            f"{outer_horizon(st):.6g}"
        )
    r_ergo = float(ergosphere_radius(st, th_o))
    if r_o <= r_ergo:
        logger.warning(
            "camera radius %g lies inside the ergosphere (r_E(theta_o) = %g): the ZAMO "
            "frame exists but E = -p_t may have either sign for the physical photon "
            "(docs/raytracing.md, section 7)",
            r_o,
            r_ergo,
        )
    alpha, beta = cam.pixel_coordinates()
    alpha, beta = alpha.ravel(), beta.ravel()  # C order: row-major over (H, W)
    rho2 = (alpha**2 + beta**2) / r_o**2
    if np.any(rho2 >= 1.0):
        raise ValueError(
            f"fov = {cam.fov:.6g} is too large for radius = {r_o:.6g}: pixels with "
            f"alpha^2 + beta^2 >= radius^2 are not directions on the observer's sky "
            f"(need fov < radius / sqrt(2))"
        )
    n = alpha.size
    p_local = np.empty((n, 4), dtype=np.float64)  # p^(a), a = t, r, theta, phi
    p_local[:, 0] = 1.0
    p_local[:, 1] = np.sqrt(1.0 - rho2)
    p_local[:, 2] = beta / r_o
    p_local[:, 3] = -alpha / r_o
    e = zamo_tetrad(st, r_o, th_o)  # rows e_(a)^mu
    p_up = p_local @ e  # p^mu = sum_a p^(a) e_(a)^mu
    p_dn = p_up @ metric(st, r_o, th_o)  # p_mu = g_mu_nu p^nu (g symmetric)
    gi = inverse_metric_components(st, r_o, th_o)
    residual = (
        gi.gtt * p_dn[:, 0] ** 2
        + 2.0 * gi.gtphi * p_dn[:, 0] * p_dn[:, 3]
        + gi.grr * p_dn[:, 1] ** 2
        + gi.gthth * p_dn[:, 2] ** 2
        + gi.gphph * p_dn[:, 3] ** 2
    )
    worst = float(np.max(np.abs(residual)))
    if not worst < NULL_CHECK_TOLERANCE:
        raise RuntimeError(
            f"tetrad projection is not null: max |g^mn p_m p_n| = {worst:.3e} "
            f"exceeds {NULL_CHECK_TOLERANCE:.0e}"
        )
    if reverse:
        p_dn = -p_dn
    p_dn[:, 0] = _null_pt(gi, p_dn)
    y = np.empty((n, 8), dtype=np.float64)
    y[:, 0] = 0.0
    y[:, 1] = r_o
    y[:, 2] = th_o
    y[:, 3] = ph_o
    y[:, 4:8] = p_dn
    return y
