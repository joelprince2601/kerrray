"""Camera, ZAMO tetrad and initial-state tests (PROJECT.md sections 15 and 16;
docs/architecture.md sections 1.3 and 5; derivation in docs/raytracing.md).

Everything is checked against quantities computed independently of
``camera.py``: the full inverse metric for the null residual, the closed-form
inverse-metric components for ``lapse`` and ``omega``, the co-frame closed
form of ``p_mu``, the constants of motion ``E``, ``L_z`` and ``Q`` of
docs/architecture.md section 1, and the exact Schwarzschild impact-parameter
relation ``b = sqrt(alpha^2 + beta^2) / sqrt(1 - 2 M / r_o)``.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from kerrray.geometry import (
    Spacetime,
    inverse_metric,
    inverse_metric_components,
    kerr,
    metric,
    metric_components,
    outer_horizon,
    schwarzschild,
)
from kerrray.raytracing.camera import Camera, initial_states, zamo_tetrad
from kerrray.utils.config import INCLINATION_EPSILON_DEG, ObserverConfig, RaytraceConfig

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
IDX_T, IDX_R, IDX_TH, IDX_PH, IDX_PT, IDX_PR, IDX_PTH, IDX_PPH = range(8)  # architecture.md section 1
SPINS = (0.0, 0.9, -0.7)
INCLINATIONS = (INCLINATION_EPSILON_DEG, 17.0, 60.0, 90.0, 150.0)


def null_residual(st: Spacetime, y: np.ndarray) -> np.ndarray:
    """``g^{mu nu} p_mu p_nu`` from the full inverse metric (independent of camera.py)."""
    gi = inverse_metric(st, y[:, IDX_R], y[:, IDX_TH])
    p = y[:, IDX_PT:]
    return np.einsum("nij,ni,nj->n", gi, p, p)


def constants(st: Spacetime, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``E = -p_t``, ``L_z = p_phi`` and the null Carter constant of architecture.md section 1."""
    energy, lz, p_th, th = -y[:, IDX_PT], y[:, IDX_PPH], y[:, IDX_PTH], y[:, IDX_TH]
    carter = p_th**2 + np.cos(th) ** 2 * (lz**2 / np.sin(th) ** 2 - st.a**2 * energy**2)
    return energy, lz, carter


def frame_quantities(st: Spacetime, r: float, theta: float) -> tuple[float, float]:
    """``(lapse, omega)`` read off the tetrad rows ``e_(t) = (1/lapse)(1, 0, 0, omega)``."""
    e = zamo_tetrad(st, r, theta)
    return 1.0 / e[0, 0], e[0, 3] / e[0, 0]


# --- Camera dataclass ---------------------------------------------------------


def test_camera_defaults_and_derived_quantities() -> None:
    cam = Camera(radius=1000, inclination_deg=60)
    assert (cam.phi_deg, cam.fov, cam.resolution) == (0.0, 12.0, 64)
    assert cam.n_rays == 64 * 64 and cam.shape == (64, 64)
    assert cam.pixel_size == pytest.approx(2.0 * 12.0 / 64)
    assert cam.inclination_rad == math.radians(60.0) and cam.phi_rad == 0.0
    assert isinstance(cam.radius, float) and isinstance(cam.inclination_deg, float)


@pytest.mark.parametrize("resolution", [0, 1, 3, 5, 63, -2, 2.0, True])
def test_camera_rejects_odd_or_non_integer_resolution(resolution: object) -> None:
    with pytest.raises(ValueError):
        Camera(radius=1000.0, inclination_deg=60.0, resolution=resolution)  # type: ignore[arg-type]


@pytest.mark.parametrize("resolution", [2, 4, 64, np.int64(8)])
def test_camera_accepts_even_resolution(resolution: int) -> None:
    cam = Camera(radius=1000.0, inclination_deg=60.0, resolution=resolution)
    assert cam.resolution == int(resolution) and type(cam.resolution) is int


@pytest.mark.parametrize(
    "override",
    [
        {"radius": 0.0},
        {"radius": -1.0},
        {"radius": math.inf},
        {"inclination_deg": -1.0},
        {"inclination_deg": 180.5},
        {"inclination_deg": math.nan},
        {"phi_deg": math.inf},
        {"fov": 0.0},
        {"fov": -3.0},
        {"fov": math.nan},
    ],
)
def test_camera_rejects_invalid_parameters(override: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Camera(**{"radius": 1000.0, "inclination_deg": 60.0, **override})


def test_camera_from_config_blocks() -> None:
    observer = ObserverConfig(radius=500.0, inclination_deg=45.0, phi_deg=30.0)
    raytrace = RaytraceConfig(resolution=16, fov=10.0)
    assert Camera.from_config(observer, raytrace) == Camera(
        radius=500.0, inclination_deg=45.0, phi_deg=30.0, fov=10.0, resolution=16
    )


@pytest.mark.parametrize(
    "given, expected",
    [
        (0.0, INCLINATION_EPSILON_DEG),
        (INCLINATION_EPSILON_DEG / 10.0, INCLINATION_EPSILON_DEG),
        (180.0, 180.0 - INCLINATION_EPSILON_DEG),
        (180.0 - INCLINATION_EPSILON_DEG / 2.0, 180.0 - INCLINATION_EPSILON_DEG),
    ],
)
def test_axis_inclination_is_clamped_and_gives_finite_null_states(
    caplog: pytest.LogCaptureFixture, given: float, expected: float
) -> None:
    """D-008: the tetrad is singular on the axis; the clamped camera must still work."""
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        cam = Camera(radius=1000.0, inclination_deg=given, resolution=4)
    assert cam.inclination_deg == expected
    assert any("D-008" in record.getMessage() for record in caplog.records)
    for st in (schwarzschild(), kerr(1.0, 0.9)):
        y = initial_states(cam, st)
        assert np.all(np.isfinite(y))
        assert np.abs(null_residual(st, y)).max() < 1e-12


def test_off_axis_inclination_is_not_clamped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        cam = Camera(radius=1000.0, inclination_deg=INCLINATION_EPSILON_DEG)
        Camera(radius=1000.0, inclination_deg=90.0)
    assert cam.inclination_deg == INCLINATION_EPSILON_DEG
    assert not [r for r in caplog.records if "D-008" in r.getMessage()]


# --- Pixel grid ---------------------------------------------------------------


@pytest.mark.parametrize("resolution, fov", [(2, 1.0), (8, 12.0), (64, 12.0), (10, 5.5)])
def test_pixel_coordinates_layout(resolution: int, fov: float) -> None:
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=fov, resolution=resolution)
    alpha, beta = cam.pixel_coordinates()
    n, d = resolution, 2.0 * fov / resolution
    assert alpha.shape == beta.shape == (n, n) == cam.shape
    assert cam.pixel_size == pytest.approx(d)
    # alpha depends on the column only, beta on the row only
    assert np.array_equal(alpha, np.broadcast_to(alpha[0], (n, n)))
    assert np.array_equal(beta, np.broadcast_to(beta[:, :1], (n, n)))
    # alpha increases to the right; beta decreases down the rows; row 0 is the top (beta > 0)
    assert np.all(np.diff(alpha[0]) > 0) and np.all(np.diff(beta[:, 0]) < 0)
    assert beta[0, 0] > 0 > beta[-1, 0] and alpha[0, 0] < 0 < alpha[0, -1]
    # half-pixel offsets: centres at -fov + (j + 1/2) d, uniform spacing, outer edges at +-fov
    expected = -fov + (np.arange(n) + 0.5) * d
    np.testing.assert_allclose(alpha[0], expected, rtol=0, atol=1e-13 * fov)
    np.testing.assert_allclose(beta[:, 0], -expected, rtol=0, atol=1e-13 * fov)
    np.testing.assert_allclose(np.diff(alpha[0]), d, rtol=1e-12)
    assert alpha[0, 0] - d / 2 == pytest.approx(-fov) and alpha[0, -1] + d / 2 == pytest.approx(fov)
    # alpha = 0 and beta = 0 are never sampled: the smallest magnitude is half a pixel
    assert not np.any(alpha == 0.0) and not np.any(beta == 0.0)
    assert np.abs(alpha).min() == pytest.approx(d / 2) and np.abs(beta).min() == pytest.approx(d / 2)
    # the grid is exactly reflection symmetric
    assert np.array_equal(alpha[:, ::-1], -alpha) and np.array_equal(beta[::-1, :], -beta)


# --- ZAMO tetrad --------------------------------------------------------------


@pytest.mark.parametrize("spin", [0.0, 0.5, -0.9, 0.998])
@pytest.mark.parametrize(
    "r_offset, theta", [(0.1, 0.3), (3.0, math.pi / 2), (10.0, 2.7), (1000.0, 1.0), (1.0e4, 0.05)]
)
def test_zamo_tetrad_is_orthonormal_and_complete(spin: float, r_offset: float, theta: float) -> None:
    """``e_(a)^mu e_(b)^nu g_mu_nu = eta_ab`` and ``eta^ab e_(a)^mu e_(b)^nu = g^{mu nu}``."""
    st = kerr(1.0, spin)
    r = outer_horizon(st) + r_offset if r_offset < 1.0 else r_offset
    e = zamo_tetrad(st, r, theta)
    assert e.shape == (4, 4) and np.all(np.isfinite(e))
    np.testing.assert_allclose(e @ metric(st, r, theta) @ e.T, ETA, rtol=0, atol=1e-12)
    ginv = inverse_metric(st, r, theta)
    np.testing.assert_allclose(e.T @ ETA @ e, ginv, rtol=1e-12, atol=1e-12 * np.abs(ginv).max())
    # only the (t, phi) row couples: e_(r), e_(theta), e_(phi) are diagonal
    assert np.count_nonzero(e) == (5 if spin != 0.0 else 4)


@pytest.mark.parametrize("spin", [0.0, 0.5, -0.9, 0.998])
@pytest.mark.parametrize("r, theta", [(2.5, 0.4), (3.0, math.pi / 2), (50.0, 2.0)])
def test_zamo_frame_quantities(spin: float, r: float, theta: float) -> None:
    """``lapse^2 = -1/g^tt``, ``omega = g^tphi/g^tt = -g_tphi/g_phph``, ``u_phi = 0``."""
    st = kerr(1.0, spin)
    lapse, omega = frame_quantities(st, r, theta)
    gi = inverse_metric_components(st, r, theta)
    g = metric_components(st, r, theta)
    assert lapse > 0.0
    assert lapse**2 == pytest.approx(-1.0 / float(gi.gtt), rel=1e-12)
    assert omega == pytest.approx(float(gi.gtphi / gi.gtt), rel=1e-12, abs=1e-15)
    assert omega == pytest.approx(float(-g.g_tphi / g.g_phph), rel=1e-12, abs=1e-15)
    assert np.sign(omega) == np.sign(spin)  # frame dragging follows the hole's rotation
    u = zamo_tetrad(st, r, theta)[0]  # ZAMO four-velocity e_(t)
    u_low = metric(st, r, theta) @ u
    assert abs(u_low[3]) <= 1e-12 * max(1.0, abs(float(g.g_tphi) * u[0]))  # zero angular momentum
    assert u_low @ u == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize("r, theta", [(3.0, 1.0), (1000.0, math.pi / 2), (2.1, 3.0)])
def test_zamo_tetrad_schwarzschild_limit(r: float, theta: float) -> None:
    """For ``a = 0`` the ZAMO is the static observer with the textbook diagonal frame."""
    e = zamo_tetrad(schwarzschild(), r, theta)
    f = 1.0 - 2.0 / r
    expected = np.diag([1.0 / math.sqrt(f), math.sqrt(f), 1.0 / r, 1.0 / (r * math.sin(theta))])
    np.testing.assert_allclose(e, expected, rtol=1e-14, atol=0)


def test_zamo_tetrad_small_spin_is_linear_in_a() -> None:
    r, theta = 4.0, 1.1
    e0 = zamo_tetrad(schwarzschild(), r, theta)
    ratios = [np.abs(zamo_tetrad(kerr(1.0, a), r, theta) - e0).max() / a for a in (1e-3, 1e-4)]
    assert ratios[0] > 0.0 and abs(ratios[0] / ratios[1] - 1.0) < 1e-2


def test_zamo_tetrad_broadcasts() -> None:
    st = kerr(1.0, 0.7)
    r = np.array([3.0, 10.0, 100.0])
    e = zamo_tetrad(st, r, 1.0)
    assert e.shape == (3, 4, 4)
    for k, r_k in enumerate(r):
        np.testing.assert_array_equal(e[k], zamo_tetrad(st, r_k, 1.0))
    assert zamo_tetrad(st, r[:, None], np.array([[0.5, 2.0]])).shape == (3, 2, 4, 4)


@pytest.mark.parametrize(
    "r, theta",
    [
        ("r_plus", 1.0),
        ("inside", 1.0),
        (3.0, 0.0),
        (3.0, math.pi),
        (3.0, -0.1),
        (3.0, 3.5),
        (math.nan, 1.0),
        (3.0, math.inf),
    ],
)
def test_zamo_tetrad_rejects_horizon_axis_and_non_finite(r: object, theta: float) -> None:
    st = kerr(1.0, 0.9)
    r_value = {"r_plus": outer_horizon(st), "inside": 0.5 * outer_horizon(st)}.get(r, r)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        zamo_tetrad(st, r_value, theta)  # type: ignore[arg-type]


# --- Initial states -----------------------------------------------------------


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("inclination", INCLINATIONS)
def test_initial_states_are_null_reversed_and_laid_out(spin: float, inclination: float) -> None:
    st = kerr(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=inclination, phi_deg=20.0, resolution=8)
    y = initial_states(cam, st)
    assert y.shape == (cam.n_rays, 8) and y.dtype == np.float64 and np.all(np.isfinite(y))
    assert np.all(y[:, IDX_T] == 0.0) and np.all(y[:, IDX_R] == cam.radius)
    assert np.all(y[:, IDX_TH] == cam.inclination_rad) and np.all(y[:, IDX_PH] == cam.phi_rad)
    residual = null_residual(st, y)
    energy = -y[:, IDX_PT]
    assert np.abs(residual).max() < 1e-12
    assert np.abs(residual / energy**2).max() < 1e-12
    # reversed ray: E < 0 and inward motion in the integration direction (never ESCAPED at launch)
    gi = inverse_metric_components(st, cam.radius, cam.inclination_rad)
    assert np.all(energy < 0.0) and np.all(gi.grr * y[:, IDX_PR] < 0.0)
    # the physical photon: E > 0 (camera outside the ergosphere), outward, null
    y_phys = initial_states(cam, st, reverse=False)
    assert np.all(-y_phys[:, IDX_PT] > 0.0) and np.all(gi.grr * y_phys[:, IDX_PR] > 0.0)
    assert np.abs(null_residual(st, y_phys)).max() < 1e-12
    # reversal is an exact sign flip up to the round-off removed by the re-normalisation
    np.testing.assert_allclose(y[:, IDX_PT:], -y_phys[:, IDX_PT:], rtol=1e-13, atol=0)
    # xi = L_z / E and eta = Q / E^2 are invariant under the reversal
    e1, l1, q1 = constants(st, y)
    e2, l2, q2 = constants(st, y_phys)
    np.testing.assert_allclose(l1 / e1, l2 / e2, rtol=1e-13)
    np.testing.assert_allclose(q1 / e1**2, q2 / e2**2, rtol=1e-13, atol=1e-13 * np.abs(q2).max())


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("inclination", (30.0, 90.0))
def test_initial_states_match_coframe_closed_form(spin: float, inclination: float) -> None:
    """``p_mu = eta_ab p^(b) e^(a)_mu`` with the co-frame ``e^(t) = lapse dt``,
    ``e^(phi) = sqrt(g_phph) (dphi - omega dt)``, ``e^(r) = sqrt(g_rr) dr``,
    ``e^(theta) = sqrt(g_thth) dtheta`` (docs/raytracing.md section 4)."""
    st = kerr(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=inclination, resolution=6)
    r_o, th_o = cam.radius, cam.inclination_rad
    y = initial_states(cam, st, reverse=False)
    alpha, beta = (x.ravel() for x in cam.pixel_coordinates())
    lapse, omega = frame_quantities(st, r_o, th_o)
    g = metric_components(st, r_o, th_o)
    p_phi_loc, p_th_loc = -alpha / r_o, beta / r_o
    p_r_loc = np.sqrt(1.0 - (alpha**2 + beta**2) / r_o**2)
    np.testing.assert_allclose(y[:, IDX_PT], -lapse - omega * np.sqrt(g.g_phph) * p_phi_loc, rtol=1e-12)
    np.testing.assert_allclose(y[:, IDX_PR], np.sqrt(g.g_rr) * p_r_loc, rtol=1e-12)
    np.testing.assert_allclose(y[:, IDX_PTH], np.sqrt(g.g_thth) * p_th_loc, rtol=1e-12)
    np.testing.assert_allclose(y[:, IDX_PPH], np.sqrt(g.g_phph) * p_phi_loc, rtol=1e-12)
    # row-major layout: p_theta depends on the row (beta), p_phi on the column (alpha)
    p_th = y[:, IDX_PTH].reshape(cam.shape)
    p_ph = y[:, IDX_PPH].reshape(cam.shape)
    assert np.array_equal(p_th, np.broadcast_to(p_th[:, :1], cam.shape))
    assert np.array_equal(p_ph, np.broadcast_to(p_ph[:1], cam.shape))


@pytest.mark.parametrize("radius, tol", [(1.0e4, 1e-3), (1.0e6, 2e-6)])
def test_schwarzschild_impact_parameter_is_recovered(radius: float, tol: float) -> None:
    """``b = sqrt(L_z^2 + Q) / E`` equals ``sqrt(alpha^2 + beta^2)`` up to the
    finite-distance factor ``(1 - 2 M / r_o)^(-1/2) = 1 + M / r_o + ...``; the
    leading correction is exactly ``M / r_o`` (1e-6 at r_o = 1e6), hence the
    tolerance ``2 M / r_o`` there (docs/raytracing.md section 5)."""
    st = schwarzschild()
    cam = Camera(radius=radius, inclination_deg=60.0, resolution=8)
    y = initial_states(cam, st)
    energy, lz, carter = constants(st, y)
    b = np.sqrt(lz**2 + carter) / np.abs(energy)
    alpha, beta = (x.ravel() for x in cam.pixel_coordinates())
    b0 = np.sqrt(alpha**2 + beta**2)
    rel = np.abs(b / b0 - 1.0)
    assert rel.max() < tol
    np.testing.assert_allclose(b, b0 / math.sqrt(1.0 - 2.0 * st.mass / radius), rtol=1e-12)
    np.testing.assert_allclose(rel, st.mass / radius, rtol=2e-4)  # (1-2x)^(-1/2) - 1 = x (1 + 1.5 x + ...)


def test_finite_distance_correction_scales_as_one_over_radius() -> None:
    st = schwarzschild()
    errors = []
    for radius in (1.0e3, 1.0e4, 1.0e5):
        cam = Camera(radius=radius, inclination_deg=45.0, resolution=4)
        energy, lz, carter = constants(st, initial_states(cam, st))
        alpha, beta = (x.ravel() for x in cam.pixel_coordinates())
        errors.append(np.abs(np.sqrt(lz**2 + carter) / np.abs(energy) / np.hypot(alpha, beta) - 1.0).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=2e-3)
    assert errors[1] / errors[2] == pytest.approx(10.0, rel=2e-4)


@pytest.mark.parametrize("spin", [0.9, -0.5])
def test_bardeen_celestial_coordinates_in_the_far_limit(spin: float) -> None:
    """``alpha -> -xi / sin(theta_o)`` and ``beta -> p_theta / E`` for ``r_o -> infinity``
    (Bardeen 1973; docs/architecture.md section 1.3), with ``xi`` and ``eta`` invariant
    under the reversal; the prograde side (physical ``a L_z > 0``) is ``alpha < 0``."""
    st = kerr(1.0, spin)
    theta_o = math.radians(60.0)
    cam = Camera(radius=1.0e6, inclination_deg=60.0, resolution=8)
    y = initial_states(cam, st)
    energy, lz, carter = constants(st, y)
    xi, eta = lz / energy, carter / energy**2
    alpha, beta = (x.ravel() for x in cam.pixel_coordinates())
    np.testing.assert_allclose(alpha, -xi / math.sin(theta_o), rtol=1e-5)
    np.testing.assert_allclose(beta, y[:, IDX_PTH] / energy, rtol=1e-5)
    beta_sq = eta + st.a**2 * math.cos(theta_o) ** 2 - xi**2 / math.tan(theta_o) ** 2
    np.testing.assert_allclose(beta**2, beta_sq, rtol=1e-5)
    y_phys = initial_states(cam, st, reverse=False)
    e_phys, lz_phys, _ = constants(st, y_phys)
    np.testing.assert_allclose(lz_phys / e_phys, xi, rtol=1e-13)
    assert np.all(np.sign(lz_phys) == -np.sign(alpha))
    assert np.all(np.sign(st.a * lz_phys) == -np.sign(spin * alpha))


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("inclination", (30.0, 90.0))
def test_mirror_symmetries(spin: float, inclination: float) -> None:
    st = kerr(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=inclination, resolution=8)
    y = initial_states(cam, st).reshape(cam.shape + (8,))
    _, omega = frame_quantities(st, cam.radius, cam.inclination_rad)
    # alpha -> -alpha (columns reversed): L_z flips, p_r and p_theta unchanged,
    # and the ZAMO-frame energy lapse p^(t) = -(p_t + omega p_phi) unchanged
    mirrored = y[:, ::-1, :]
    np.testing.assert_allclose(mirrored[..., IDX_PPH], -y[..., IDX_PPH], rtol=1e-12)
    np.testing.assert_allclose(mirrored[..., IDX_PR], y[..., IDX_PR], rtol=1e-14)
    np.testing.assert_allclose(mirrored[..., IDX_PTH], y[..., IDX_PTH], rtol=1e-14)
    e_zamo = -(y[..., IDX_PT] + omega * y[..., IDX_PPH])
    np.testing.assert_allclose(e_zamo[:, ::-1], e_zamo, rtol=1e-12)
    if spin != 0.0:  # frame dragging: p_t itself differs by 2 omega L_z between the mirror images
        np.testing.assert_allclose(
            mirrored[..., IDX_PT] - y[..., IDX_PT], 2.0 * omega * y[..., IDX_PPH], rtol=1e-6
        )
    else:
        np.testing.assert_allclose(mirrored[..., IDX_PT], y[..., IDX_PT], rtol=1e-14)
    # beta -> -beta (rows reversed): p_theta flips, everything else unchanged
    flipped = y[::-1, :, :]
    np.testing.assert_allclose(flipped[..., IDX_PTH], -y[..., IDX_PTH], rtol=1e-12)
    for k in (IDX_PT, IDX_PR, IDX_PPH):
        np.testing.assert_allclose(flipped[..., k], y[..., k], rtol=1e-13)


@pytest.mark.parametrize("spin", SPINS)
def test_equatorial_observer_rows_nearest_the_equator(spin: float) -> None:
    st = kerr(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=90.0, resolution=8)
    alpha, beta = cam.pixel_coordinates()
    assert not np.any(beta == 0.0)  # even resolution: no beta = 0 row
    y = initial_states(cam, st)
    p_th = y[:, IDX_PTH].reshape(cam.shape)
    p_ph = y[:, IDX_PPH].reshape(cam.shape)
    row = int(np.argmin(np.abs(beta[:, 0])))
    assert abs(beta[row, 0]) == pytest.approx(cam.pixel_size / 2)
    # at theta_o = pi/2, sqrt(g_thth) = r_o, so |p_theta| = |beta| = half a pixel on that row,
    # small compared with the outer rows and non-zero; the row below the equator mirrors it
    half = np.abs(p_th[row])
    np.testing.assert_allclose(half, cam.pixel_size / 2, rtol=1e-12)
    assert half.max() <= np.abs(p_th).max() / (cam.resolution - 1) * (1.0 + 1e-12)
    np.testing.assert_allclose(p_th[cam.resolution - 1 - row], -p_th[row], rtol=1e-12)
    # alpha of either sign gives opposite L_z: reversed ray sign(L_z) = sign(alpha),
    # physical photon sign(L_z) = -sign(alpha)
    assert np.all(np.sign(p_ph[row]) == np.sign(alpha[row]))
    assert np.all(p_ph[row][alpha[row] < 0] < 0) and np.all(p_ph[row][alpha[row] > 0] > 0)
    p_ph_phys = initial_states(cam, st, reverse=False)[:, IDX_PPH].reshape(cam.shape)
    assert np.all(np.sign(p_ph_phys[row]) == -np.sign(alpha[row]))


def test_initial_states_rejects_camera_on_or_inside_the_horizon() -> None:
    st = kerr(1.0, 0.5)
    for radius in (outer_horizon(st), 0.9 * outer_horizon(st)):
        with pytest.raises(ValueError):
            initial_states(Camera(radius=radius, inclination_deg=60.0, resolution=2), st)


def test_initial_states_rejects_directions_off_the_sky() -> None:
    st = schwarzschild()
    with pytest.raises(ValueError):  # corner pixel at (7.5, 7.5): alpha^2 + beta^2 > r_o^2 = 100
        initial_states(Camera(radius=10.0, inclination_deg=60.0, fov=10.0, resolution=4), st)
    y = initial_states(Camera(radius=10.0, inclination_deg=60.0, fov=7.0, resolution=4), st)
    assert np.abs(null_residual(st, y)).max() < 1e-12


def test_camera_inside_the_ergosphere_warns_but_stays_null(caplog: pytest.LogCaptureFixture) -> None:
    """The ZAMO exists down to the horizon; inside the ergosphere the physical
    photon's ``E`` may be negative, so only frame-based statements are tested."""
    st = kerr(1.0, 0.9)
    cam = Camera(radius=1.6, inclination_deg=90.0, fov=0.8, resolution=4)
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        y_phys = initial_states(cam, st, reverse=False)
        y = initial_states(cam, st)
    assert any("ergosphere" in record.getMessage() for record in caplog.records)
    assert np.abs(null_residual(st, y)).max() < 1e-12
    assert np.abs(null_residual(st, y_phys)).max() < 1e-12
    e = zamo_tetrad(st, cam.radius, cam.inclination_rad)
    p_t_local = -(e[0] @ y_phys[:, IDX_PT:].T)  # p^(t) = -e_(t)^mu p_mu, must be +1
    np.testing.assert_allclose(p_t_local, 1.0, rtol=1e-12)
    np.testing.assert_allclose(y[:, IDX_PT:], -y_phys[:, IDX_PT:], rtol=1e-13)
