"""Tests for numerical shadow reconstruction (PROJECT.md section 17; raytracing role).

The shadows are computed once per module at 16x16 and 24x24 (a = 0) and
24x24 (a = 0.9, i = 60 deg) with rtol 1e-7 and fov 8 M. The reference values
(the Schwarzschild critical impact parameter 3 sqrt(3) M, the Bardeen curve)
are computed by ``kerrray.photons.orbits`` and only *checked* here.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime
from kerrray.photons import TerminationState
from kerrray.photons.orbits import critical_impact_parameters
from kerrray.raytracing.boundary import extract_boundary, summarise_shadow
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.shadow import ShadowImage, compute_shadow, state_counts

INTEG = IntegratorOptions(method="rk45", rtol=1e-7, atol=1e-9)
TERM = TerminationOptions(horizon_epsilon=1e-6, escape_radius=1000.0)
OBSERVER_RADIUS = 1000.0
FOV = 8.0


def _shadow(spin: float, inclination: float, resolution: int) -> ShadowImage:
    st = Spacetime(mass=1.0, spin=spin)
    cam = Camera(radius=OBSERVER_RADIUS, inclination_deg=inclination, fov=FOV, resolution=resolution)
    return compute_shadow(st, cam, INTEG, TERM)


@pytest.fixture(scope="module")
def schwarzschild_16() -> ShadowImage:
    return _shadow(0.0, 60.0, 16)


@pytest.fixture(scope="module")
def schwarzschild_24() -> ShadowImage:
    return _shadow(0.0, 60.0, 24)


@pytest.fixture(scope="module")
def kerr_24() -> ShadowImage:
    return _shadow(0.9, 60.0, 24)


ALL_IMAGES = ("schwarzschild_16", "schwarzschild_24", "kerr_24")
SCHWARZSCHILD_IMAGES = ("schwarzschild_16", "schwarzschild_24")


@pytest.mark.parametrize("name", ALL_IMAGES)
def test_every_pixel_is_captured_or_escaped_and_counts_sum(name: str, request: pytest.FixtureRequest) -> None:
    img: ShadowImage = request.getfixturevalue(name)
    assert sum(img.counts.values()) == img.n_rays == img.camera.resolution ** 2
    assert img.count(TerminationState.CAPTURED) + img.count(TerminationState.ESCAPED) == img.n_rays
    assert img.n_failed == 0 and img.n_other == 0
    assert img.count(TerminationState.CAPTURED) > 0
    assert np.array_equal(img.captured, img.state == int(TerminationState.CAPTURED))
    assert img.counts == state_counts(img.state)


@pytest.mark.parametrize("name", SCHWARZSCHILD_IMAGES)
def test_schwarzschild_captured_region_is_a_disk(name: str, request: pytest.FixtureRequest) -> None:
    """Spherical symmetry: capture depends only on the pixel radius, so the mask is a disk."""
    img: ShadowImage = request.getfixturevalue(name)
    radius = np.hypot(img.alpha, img.beta)
    assert radius[img.captured].max() < radius[~img.captured].min()


@pytest.mark.parametrize("name", SCHWARZSCHILD_IMAGES)
def test_schwarzschild_radius_matches_critical_impact_parameter(
    name: str, request: pytest.FixtureRequest
) -> None:
    """Mean extracted radius = b_c within one pixel plus the O(M/r_o) finite-distance offset."""
    img: ShadowImage = request.getfixturevalue(name)
    st = img.spacetime
    b_c = critical_impact_parameters(st)[0]
    assert b_c == pytest.approx(3.0 * math.sqrt(3.0) * st.mass, rel=1e-10)  # reference check only
    angles, radii = extract_boundary(img)
    assert angles.shape == radii.shape == (360,)
    finite_distance_offset = b_c * (1.0 - math.sqrt(1.0 - 2.0 * st.mass / OBSERVER_RADIUS))
    assert abs(radii.mean() - b_c) < img.pixel_size + finite_distance_offset
    assert radii.max() - radii.min() < img.pixel_size  # a disk to within the pixel size


def test_kerr_shadow_boundary_matches_bardeen_curve(kerr_24: ShadowImage) -> None:
    """a = 0.9, i = 60 deg: boundary within 1.5 px rms of the analytic curve.

    A mirrored image orientation would fail this: the analytic curve is
    displaced by about 1.7 M and asymmetric by 0.4 M, both several pixels.
    """
    assert kerr_24.count(TerminationState.CAPTURED) > 0
    summary = summarise_shadow(kerr_24)
    err = summary["error"]
    assert err["rms_px"] < 1.5
    assert err["max_px"] < 3.0
    # The shadow of a > 0 is displaced to alpha > 0 (prograde, flattened side at alpha < 0).
    assert err["analytic_centroid_alpha"] > 0.0
    assert math.copysign(1.0, err["numeric_centroid_alpha"]) == math.copysign(1.0, err["analytic_centroid_alpha"])
    assert err["centroid_shift_norm_px"] < 1.0
    assert summary["error_corrected"]["scale"] == pytest.approx(1.0 / math.sqrt(1.0 - 2.0 / OBSERVER_RADIUS))


@pytest.mark.parametrize("name", ALL_IMAGES)
def test_captured_mask_is_symmetric_under_beta_reflection(name: str, request: pytest.FixtureRequest) -> None:
    """beta -> -beta flips p_theta only; E, L_z and Q are unchanged, so capture is invariant."""
    img: ShadowImage = request.getfixturevalue(name)
    assert np.array_equal(img.captured[::-1, :], img.captured)
    assert np.array_equal(img.beta[::-1, :], -img.beta)


@pytest.mark.parametrize("name", ALL_IMAGES)
def test_conserved_momenta_are_exact_and_layout_is_row_major(name: str, request: pytest.FixtureRequest) -> None:
    """p_t and p_phi are exact constants of the Hamiltonian form (their rhs is identically 0)."""
    img: ShadowImage = request.getfixturevalue(name)
    assert img.result.max_energy_drift.max() == 0.0
    assert img.result.max_lz_drift.max() == 0.0
    assert np.array_equal(img.result.state.reshape(img.shape), img.state)
    assert img.result.Y.reshape(img.shape + (8,)).shape == img.shape + (8,)
    escaped = img.result.state == int(TerminationState.ESCAPED)
    assert img.result.max_null_error[escaped].max() < 1e-5


def test_save_and_load_round_trip(schwarzschild_16: ShadowImage, tmp_path: Path) -> None:
    path = schwarzschild_16.save(tmp_path / "nested" / "shadow.npz")
    loaded = ShadowImage.load(path)
    assert loaded.camera == schwarzschild_16.camera
    assert loaded.spacetime == schwarzschild_16.spacetime
    assert loaded.counts == schwarzschild_16.counts
    for name in ("alpha", "beta", "state", "captured"):
        assert np.array_equal(getattr(loaded, name), getattr(schwarzschild_16, name))
    for name in ("Y", "state", "n_steps", "lam", "max_null_error", "n_rejected"):
        assert np.array_equal(getattr(loaded.result, name), getattr(schwarzschild_16.result, name))
    assert loaded.result.runtime_s == schwarzschild_16.result.runtime_s
    assert loaded.result.event_Y is None


def test_compute_shadow_reports_progress_and_respects_chunks() -> None:
    fractions: list[float] = []
    img = compute_shadow(
        Spacetime(1.0, 0.0),
        Camera(radius=OBSERVER_RADIUS, inclination_deg=60.0, fov=FOV, resolution=8),
        INTEG,
        TERM,
        progress=fractions.append,
        chunk_size=24,
    )
    assert fractions == [24 / 64, 48 / 64, 1.0]
    assert img.shape == (8, 8) and sum(img.counts.values()) == 64
