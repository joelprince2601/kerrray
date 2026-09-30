"""Tests for shadow-boundary extraction and comparison (raytracing role).

Synthetic masks (a disk, a shifted ellipse) exercise the sub-pixel
extraction without ray tracing; the analytic curve tests use the computed
critical impact parameter of ``kerrray.photons.orbits``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult
from kerrray.photons.orbits import critical_impact_parameters
from kerrray.raytracing.boundary import (
    analytic_boundary,
    boundary_error,
    curve_summary,
    extract_boundary,
    extract_boundary_from_mask,
    finite_distance_scale,
    mask_centroid,
    polygon_centroid,
    polygon_radii,
    summarise_shadow,
)
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.shadow import ShadowImage, state_counts

CAM = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=64)
ALPHA, BETA = CAM.pixel_coordinates()
D = CAM.pixel_size
SCHWARZSCHILD = Spacetime(mass=1.0, spin=0.0)


def _synthetic_image(mask: np.ndarray, st: Spacetime = SCHWARZSCHILD, cam: Camera = CAM) -> ShadowImage:
    """A ShadowImage whose only meaningful content is the captured mask."""
    state = np.where(mask, 2, 1).astype(np.int64)
    n = mask.size
    result = BatchResult(
        Y=np.zeros((n, 8)), state=state.ravel(), n_steps=np.zeros(n, dtype=np.int64), lam=np.zeros(n),
        max_null_error=np.zeros(n), max_energy_drift=np.zeros(n), max_lz_drift=np.zeros(n),
        max_carter_drift=np.zeros(n), runtime_s=1.0,
    )
    alpha, beta = cam.pixel_coordinates()
    return ShadowImage(alpha=alpha, beta=beta, state=state, captured=mask, counts=state_counts(state),
                       result=result, camera=cam, spacetime=st)


# --- extraction on synthetic masks ------------------------------------------------


def test_disk_mask_boundary_is_within_half_a_pixel_of_the_true_radius() -> None:
    radius = 5.0
    mask = ALPHA**2 + BETA**2 < radius**2
    angles, radii = extract_boundary_from_mask(ALPHA, BETA, mask)
    assert angles.shape == radii.shape == (360,)
    assert angles[0] == 0.0 and angles[-1] < 2.0 * math.pi
    assert np.max(np.abs(radii - radius)) <= 0.5 * D
    assert abs(radii.mean() - radius) < 0.25 * D
    assert mask_centroid(ALPHA, BETA, mask) == (0.0, 0.0)


def test_shifted_ellipse_mask_boundary_matches_its_polar_equation() -> None:
    ca, cb, a, b = 1.3, -0.7, 4.0, 2.5
    mask = ((ALPHA - ca) / a) ** 2 + ((BETA - cb) / b) ** 2 < 1.0
    angles, radii = extract_boundary_from_mask(ALPHA, BETA, mask, n_angles=720, centre=(ca, cb))
    exact = 1.0 / np.sqrt((np.cos(angles) / a) ** 2 + (np.sin(angles) / b) ** 2)
    assert np.max(np.abs(radii - exact)) <= 0.5 * D
    centroid = mask_centroid(ALPHA, BETA, mask)
    assert math.hypot(centroid[0] - ca, centroid[1] - cb) < 0.25 * D
    default_angles, default_radii = extract_boundary_from_mask(ALPHA, BETA, mask, n_angles=720)
    assert np.array_equal(default_angles, angles)
    assert not np.array_equal(default_radii, radii)  # a different centre gives different radii


def test_extraction_rejects_empty_border_touching_and_off_centre_masks() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_boundary_from_mask(ALPHA, BETA, np.zeros_like(ALPHA, dtype=bool))
    touching = ALPHA**2 + BETA**2 < 8.5**2
    with pytest.raises(ValueError, match="border"):
        extract_boundary_from_mask(ALPHA, BETA, touching)
    disk = ALPHA**2 + BETA**2 < 4.0
    with pytest.raises(ValueError, match="outside the captured region"):
        extract_boundary_from_mask(ALPHA, BETA, disk, centre=(6.0, 0.0))
    with pytest.raises(ValueError, match="shape"):
        extract_boundary_from_mask(ALPHA, BETA, disk[:10, :10])


# --- analytic curve and polygon geometry -----------------------------------------


def test_analytic_boundary_of_schwarzschild_is_the_critical_circle() -> None:
    alpha, beta = analytic_boundary(SCHWARZSCHILD, 60.0, n=361)
    b_c = critical_impact_parameters(SCHWARZSCHILD)[0]
    assert alpha.shape == beta.shape == (360,)
    assert np.allclose(np.hypot(alpha, beta), b_c, rtol=0, atol=1e-12)
    assert b_c == pytest.approx(3.0 * math.sqrt(3.0), rel=1e-10)


def test_polygon_radii_on_the_circle_and_off_centre() -> None:
    alpha, beta = analytic_boundary(SCHWARZSCHILD, 60.0, n=720)
    b_c = critical_impact_parameters(SCHWARZSCHILD)[0]
    angles = 2.0 * math.pi * np.arange(360) / 360  # every ray passes through a vertex
    assert np.allclose(polygon_radii(alpha, beta, (0.0, 0.0), angles), b_c, rtol=0, atol=1e-12)
    ca, cb = 0.5, 0.2
    radii = polygon_radii(alpha, beta, (ca, cb), np.array([0.0, math.pi / 2, math.pi]))
    expected = [math.sqrt(b_c**2 - cb**2) - ca, math.sqrt(b_c**2 - ca**2) - cb, math.sqrt(b_c**2 - cb**2) + ca]
    sagitta = b_c * (1.0 - math.cos(math.pi / 720))  # the inscribed 720-gon lies inside the circle by this much
    assert radii == pytest.approx(expected, abs=sagitta)
    assert np.all(radii <= np.asarray(expected) + 1e-12)
    with pytest.raises(ValueError, match="does not intersect"):
        polygon_radii(alpha, beta, (2.0 * b_c, 0.0), np.array([0.0]))


def test_polygon_centroid_and_curve_summary() -> None:
    square_x, square_y = np.array([1.0, -1.0, -1.0, 1.0]) + 0.3, np.array([1.0, 1.0, -1.0, -1.0]) - 0.4
    assert polygon_centroid(square_x, square_y) == pytest.approx((0.3, -0.4))
    angles = 2.0 * math.pi * np.arange(360) / 360
    summary = curve_summary(angles, np.full(360, 2.0), (1.0, 0.0))
    assert summary["radius_mean"] == 2.0 and summary["asymmetry"] == 0.0
    assert summary["centroid_alpha"] == pytest.approx(1.0, abs=1e-12)
    assert summary["centroid_displacement"] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        polygon_centroid([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])


def test_kerr_analytic_curve_is_displaced_and_beta_symmetric() -> None:
    alpha, beta = analytic_boundary(Spacetime(1.0, 0.9), 60.0)
    cx, cy = polygon_centroid(alpha, beta)
    assert cx > 0.0 and abs(cy) < 1e-12
    assert np.isclose(beta.max(), -beta.min())


# --- comparison -------------------------------------------------------------------


def test_boundary_error_of_a_disk_against_the_critical_circle() -> None:
    radius = 5.0
    mask = ALPHA**2 + BETA**2 < radius**2
    numeric = extract_boundary_from_mask(ALPHA, BETA, mask)
    curve = analytic_boundary(SCHWARZSCHILD, 60.0)
    b_c = critical_impact_parameters(SCHWARZSCHILD)[0]
    err = boundary_error(numeric, curve, (0.0, 0.0), pixel_size=D)
    assert err["mean"] == pytest.approx(radius - b_c, abs=0.25 * D)
    assert err["rms"] <= abs(radius - b_c) + 0.5 * D
    assert err["max"] >= err["rms"] >= err["mean_abs"] >= abs(err["mean"])
    assert err["analytic_radius_mean"] == pytest.approx(b_c, rel=1e-10)
    assert err["analytic_asymmetry"] == pytest.approx(0.0, abs=1e-10)
    assert err["centroid_shift_norm"] == pytest.approx(0.0, abs=1e-10)
    for key in ("max", "rms", "mean", "numeric_radius_mean", "centroid_shift_norm"):
        assert err[key + "_px"] == pytest.approx(err[key] / D)
    assert "scale_px" not in err and err["scale"] == 1.0


def test_boundary_error_scale_rescales_the_numeric_curve() -> None:
    mask = ALPHA**2 + BETA**2 < 25.0
    numeric = extract_boundary_from_mask(ALPHA, BETA, mask)
    curve = analytic_boundary(SCHWARZSCHILD, 60.0)
    raw = boundary_error(numeric, curve, (0.0, 0.0))
    scaled = boundary_error(numeric, curve, (0.0, 0.0), scale=1.02)
    assert scaled["numeric_radius_mean"] == pytest.approx(1.02 * raw["numeric_radius_mean"])
    assert scaled["mean"] == pytest.approx(raw["mean"] + 0.02 * raw["numeric_radius_mean"])
    assert scaled["scale"] == 1.02


def test_finite_distance_scale() -> None:
    assert finite_distance_scale(SCHWARZSCHILD, 1000.0) == pytest.approx(1.0 / math.sqrt(1.0 - 2.0e-3))
    assert finite_distance_scale(Spacetime(2.0, 0.5), 8.0) == pytest.approx(1.0 / math.sqrt(0.5))
    with pytest.raises(ValueError):
        finite_distance_scale(SCHWARZSCHILD, 2.0)


def test_summarise_shadow_on_a_synthetic_image() -> None:
    b_c = critical_impact_parameters(SCHWARZSCHILD)[0]
    img = _synthetic_image(ALPHA**2 + BETA**2 < b_c**2)
    summary = summarise_shadow(img)
    assert summary["centre"] == (0.0, 0.0)
    angles, radii = extract_boundary(img)
    assert np.array_equal(summary["angles"], angles) and np.array_equal(summary["radii"], radii)
    assert summary["error"]["max_px"] <= 0.5
    assert summary["finite_distance_scale"] == pytest.approx(finite_distance_scale(SCHWARZSCHILD, 1000.0))
    corrected = summary["error_corrected"]
    assert corrected["numeric_radius_mean"] == pytest.approx(
        summary["finite_distance_scale"] * summary["error"]["numeric_radius_mean"]
    )
