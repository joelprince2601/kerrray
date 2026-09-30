"""Tests for kerrray.physics.lensing, the lensing experiment and the lens command
(PROJECT.md section 22; docs/lensing.md).

Reference values used only as checks: b_c = 3 sqrt(3) M, the weak-field
deflection 4M/b, the Darwin / Bozza strong-field coefficients (a_bar = 1,
b_bar = log[216 (7 - 4 sqrt 3)] - pi). Everything compared with them is
computed by the code under test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from kerrray.commands import lens as lens_cmd
from kerrray.experiments import lensing as lensing_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions, equatorial_photon, integrate
from kerrray.geometry import kerr, schwarzschild
from kerrray.photons import Trajectory, deflection_angle
from kerrray.physics.lensing import (
    darwin_strong_field_coefficients,
    deflection_from_trajectory,
    schwarzschild_closest_approach,
    schwarzschild_critical_impact_parameter,
    schwarzschild_deflection_exact,
    strong_field_fit,
    weak_field_deflection,
)
from kerrray.reporting.report import REPORT_HEADINGS
from kerrray.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
ST = schwarzschild()
M = ST.mass


def _trajectory(st, b: float, r0: float, rtol: float, *, prograde: bool = True) -> Trajectory:
    integ = IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2, lambda_max=4.0 * r0, max_steps=400_000)
    term = TerminationOptions(escape_radius=r0)
    return integrate(st, equatorial_photon(st, r0, b, inward=True, prograde=prograde), integ, term)


def _closest_approach_trigonometric(b: float) -> float:
    """r_0 = (2 b / sqrt 3) cos[(1/3) arccos(-3 sqrt(3) M / b)]: the r_0 > 3M root of r^3 - b^2 (r - 2M)."""
    return (2.0 * b / math.sqrt(3.0)) * math.cos(math.acos(-3.0 * math.sqrt(3.0) * M / b) / 3.0)


# ---------------------------------------------------------------- exact integral


def test_critical_impact_parameter_is_computed_and_matches_reference() -> None:
    b_c = schwarzschild_critical_impact_parameter(ST)
    assert abs(b_c - 3.0 * math.sqrt(3.0) * M) < 1e-12


@pytest.mark.parametrize("b", [6.0, 10.0, 100.0])
def test_closest_approach_matches_trigonometric_form_and_turning_point(b: float) -> None:
    r0 = schwarzschild_closest_approach(ST, b)
    assert abs(r0 - _closest_approach_trigonometric(b)) < 1e-12 * b
    assert abs(b * b - r0**3 / (r0 - 2.0 * M)) < 1e-10 * b * b
    assert 3.0 * M < r0 < b


def test_closest_approach_and_exact_integral_reject_bad_input() -> None:
    b_c = schwarzschild_critical_impact_parameter(ST)
    with pytest.raises(ValueError):
        schwarzschild_closest_approach(ST, b_c)
    with pytest.raises(ValueError):
        schwarzschild_deflection_exact(ST, 0.5 * b_c)
    with pytest.raises(ValueError):
        schwarzschild_deflection_exact(kerr(1.0, 0.5), 10.0)


def test_exact_deflection_recovers_weak_field_limit() -> None:
    """At b = 1000 M the exact value equals 4M/b to 0.5 %; the relative residual scales as 1/b."""
    b1, b2 = 1000.0, 10_000.0
    a1 = schwarzschild_deflection_exact(ST, b1)
    a2 = schwarzschild_deflection_exact(ST, b2)
    rel1 = (a1 - 4.0 * M / b1) / (4.0 * M / b1)
    rel2 = (a2 - 4.0 * M / b2) / (4.0 * M / b2)
    assert 0.0 < rel1 < 5e-3
    assert 8.0 < rel1 / rel2 < 12.0  # next order: relative residual (15 pi / 16) M / b


def test_exact_deflection_grows_monotonically_toward_critical_and_winds() -> None:
    b_c = schwarzschild_critical_impact_parameter(ST)
    offsets = np.logspace(-6, 3, 40)
    alpha = schwarzschild_deflection_exact(ST, b_c * (1.0 + offsets))
    assert isinstance(alpha, np.ndarray) and alpha.shape == offsets.shape
    assert np.all(np.diff(alpha) < 0.0)
    assert alpha[0] > 2.0 * math.pi  # more than one full winding at b/b_c - 1 = 1e-6
    assert np.all(alpha > 4.0 * M / (b_c * (1.0 + offsets)))
    assert abs(schwarzschild_deflection_exact(ST, 10.0) - alpha[np.argmin(np.abs(b_c * (1 + offsets) - 10.0))]) < 1.0


def test_weak_field_deflection_is_vectorised() -> None:
    out = weak_field_deflection(ST, [10.0, 20.0, 40.0])
    assert np.allclose(out, [0.4, 0.2, 0.1])


# ---------------------------------------------------------------- trajectories


def test_exact_integral_matches_numerical_trajectory_at_b_10() -> None:
    """b = 10 M, launch radius 1e4, rtol 1e-11: agreement to 1e-6 rad (measured ~1.5e-10)."""
    traj = _trajectory(ST, 10.0, 1.0e4, 1e-11)
    numerical = deflection_from_trajectory(traj)
    exact = schwarzschild_deflection_exact(ST, 10.0)
    assert abs(numerical - exact) < 1e-6
    assert abs(deflection_angle(traj) - exact) < 1e-6


def test_deflection_converges_with_launch_radius() -> None:
    """b = 20 M, r_0 = 1e3, 1e4, 1e5.

    docs/lensing.md section 2: the end-point-direction estimate is short of the
    exact value by 1.5 M b^3 / r_0^4 (1.2e-8 at r_0 = 1e3) and the straight-line
    estimate exceeds it by M b^3 / (2 r_0^4) (4e-9); at r_0 >= 1e4 both residuals
    (1e-12 and below) are hidden by the integration error (~2e-10 at rtol 1e-11).
    """
    b = 20.0
    exact = schwarzschild_deflection_exact(ST, b)
    errors = {}
    straight = {}
    for r0 in (1.0e3, 1.0e4, 1.0e5):
        traj = _trajectory(ST, b, r0, 1e-11)
        errors[r0] = deflection_from_trajectory(traj) - exact
        straight[r0] = deflection_angle(traj) - exact
    predicted_direction = -1.5 * M * b**3 / 1.0e3**4
    predicted_straight = 0.5 * M * b**3 / 1.0e3**4
    assert 0.7 * predicted_direction > errors[1.0e3] > 1.3 * predicted_direction
    assert 0.7 * predicted_straight < straight[1.0e3] < 1.3 * predicted_straight
    assert abs(errors[1.0e4]) < abs(errors[1.0e3])
    assert abs(errors[1.0e4]) < 1e-9
    assert abs(errors[1.0e5]) < 1e-9


def test_deflection_from_trajectory_rejects_captured_and_non_equatorial_rays() -> None:
    captured = _trajectory(ST, 4.0, 200.0, 1e-8)
    with pytest.raises(ValueError):
        deflection_from_trajectory(captured)
    escaped = _trajectory(ST, 12.0, 200.0, 1e-8)
    tilted = Trajectory(
        spacetime=escaped.spacetime, lam=escaped.lam, y=escaped.y.copy(), state=escaped.state,
        n_steps=escaped.n_steps, n_rejected=escaped.n_rejected, runtime_s=escaped.runtime_s,
        diagnostics=escaped.diagnostics,
    )
    tilted.y[:, 2] += 1e-3
    with pytest.raises(ValueError):
        deflection_from_trajectory(tilted)


def test_kerr_prograde_is_less_deflected_and_frame_dragging_residual_is_derived() -> None:
    """a = 0.9, b = 8 M, r_0 = 500 M.

    docs/lensing.md section 2: the two estimates differ by 4 M |a| / r_0^2 (the
    end-point direction counts the frame-dragging drift 2 M a / r^2 per leg that
    the straight-line estimate ignores); the prograde sense is deflected less.
    """
    st = kerr(1.0, 0.9)
    r0 = 500.0
    values = {}
    for prograde in (True, False):
        traj = _trajectory(st, 8.0, r0, 1e-9, prograde=prograde)
        values[prograde] = (deflection_from_trajectory(traj), deflection_angle(traj))
    assert values[True][0] < values[False][0]
    assert values[True][0] > float(weak_field_deflection(st, 8.0))
    predicted = 4.0 * st.mass * abs(st.a) / r0**2
    for direction, straight in values.values():
        assert 0.75 * predicted < abs(direction - straight) < 1.25 * predicted


# ---------------------------------------------------------------- strong-field fit


def test_strong_field_fit_recovers_synthetic_coefficients() -> None:
    b_c = 5.0
    b = b_c * (1.0 + np.logspace(-4, -1, 10))
    alpha = -1.3 * np.log(b / b_c - 1.0) + 0.7
    fit = strong_field_fit(np.concatenate([[4.0], b]), np.concatenate([[np.nan], alpha]), b_c=b_c)
    assert abs(fit.a_bar - 1.3) < 1e-12 and abs(fit.b_bar - 0.7) < 1e-12
    assert fit.n_points == 10 and fit.rms_residual < 1e-12
    assert np.allclose(fit.evaluate(b), alpha)
    with pytest.raises(ValueError):
        strong_field_fit([4.0, 4.5], [1.0, 2.0], b_c=b_c)


def test_strong_field_fit_of_exact_integral_matches_darwin_bozza_loosely() -> None:
    """Fit over b/b_c - 1 in [1e-6, 1e-3]; measured a_bar = 0.9997, b_bar = -0.396 vs (1, -0.4002)."""
    b_c = schwarzschild_critical_impact_parameter(ST)
    x = np.logspace(-6, -3, 12)
    alpha = schwarzschild_deflection_exact(ST, b_c * (1.0 + x))
    fit = strong_field_fit(b_c * (1.0 + x), alpha, b_c=b_c)
    a_ref, b_ref = darwin_strong_field_coefficients()
    assert abs(fit.a_bar - a_ref) < 0.01
    assert abs(fit.b_bar - b_ref) < 0.02
    assert abs(b_ref - (-0.4002)) < 1e-4  # the closed form reproduces the quoted -0.4002


# ---------------------------------------------------------------- experiment and CLI


def test_sample_impact_parameters_resolves_critical_region() -> None:
    params = lensing_experiment.LensingParams(impact_min=3.0, impact_max=20.0, n_rays=16, launch_radius=100.0)
    b = lensing_experiment.sample_impact_parameters(params, 5.0)
    assert b.shape == (16,) and np.all(np.diff(b) > 0)
    assert np.count_nonzero(b < 5.0) == 2
    assert np.isclose(b[2], 5.0 * (1.0 + lensing_experiment.CRITICAL_OFFSET_MIN))
    assert np.isclose(b[-1], 20.0)
    above = lensing_experiment.sample_impact_parameters(
        lensing_experiment.LensingParams(impact_min=6.0, impact_max=20.0, n_rays=5, launch_radius=100.0), 5.0
    )
    assert np.isclose(above[0], 6.0) and np.isclose(above[-1], 20.0) and above.shape == (5,)
    below = lensing_experiment.sample_impact_parameters(
        lensing_experiment.LensingParams(impact_min=1.0, impact_max=4.0, n_rays=3, launch_radius=100.0), 5.0
    )
    assert np.allclose(below, [1.0, 2.5, 4.0])


def _lensing_config(tmp_path: Path, **overrides):
    base = {
        "experiment.output_dir": str(tmp_path / "runs"),
        "experiment.report_dir": str(tmp_path / "reports"),
        "experiment.parameters.lensing.n_rays": 6,
        "experiment.parameters.lensing.launch_radius": 200.0,
        "experiment.parameters.lensing.impact_min": 4.0,
        "experiment.parameters.lensing.impact_max": 12.0,
        "integration.lambda_max": 1000.0,
    }
    base.update(overrides)
    return load_config(REPO_ROOT / "configs" / "lensing.yaml", base)


def test_lensing_experiment_schwarzschild_run(tmp_path: Path) -> None:
    cfg = _lensing_config(tmp_path, **{"black_hole.spin": 0.0})
    record = lensing_experiment.run(cfg)
    res = record.results
    scan = res["scans"]["schwarzschild"]
    assert set(res["scans"]) == {"schwarzschild"}
    assert scan["counts"] == {"CAPTURED": 1, "ESCAPED": 5}
    assert scan["max_abs_dev_exact"] < 1e-4  # rtol 1e-9 rays from r_0 = 200 M, incl. b/b_c - 1 = 1e-3
    assert scan["strong_field_fit"] is not None and 0.8 < scan["strong_field_fit"]["a_bar"] < 1.2
    assert res["darwin_coefficients"]["a_bar"] == 1.0
    assert res["escape_radius"] == 200.0
    turns = np.asarray(scan["turns"], dtype=float)
    assert np.all(np.abs(turns[1:]) > 0.5)  # escaped rays sweep more than half a turn
    report = Path(res["report"])
    text = report.read_text(encoding="utf-8")
    for _, heading in REPORT_HEADINGS:
        assert heading in text
    for fig in res["figures"]:
        assert Path(fig).is_file()
    assert (record.run_dir / "manifest.json").is_file() and record.summary_path.is_file()


def _group_app() -> typer.Typer:
    """A fresh Typer app in sub-command mode, as in ``kerrray.cli``.

    A Typer app with a single command and no callback runs that command as the
    application itself, so the command name would be an unexpected argument;
    the no-op callback makes the app a command group like the real CLI.
    """
    app = typer.Typer(add_completion=False)

    @app.callback()
    def _root() -> None:
        """Test application root (forces sub-command mode)."""

    return app


def test_lens_command_runs_and_prints_computed_values(tmp_path: Path) -> None:
    cfg = _lensing_config(tmp_path)
    cfg_path = tmp_path / "lens.yaml"
    cfg_path.write_text(cfg.to_yaml(), encoding="utf-8")
    app = _group_app()
    lens_cmd.register(app)
    result = CliRunner().invoke(
        app, ["lens", "--config", str(cfg_path), "--spin", "0.0", "--impact-range", "6,12", "--n", "4", "--launch-radius", "150"]
    )
    assert result.exit_code == 0, result.output
    assert "schwarzschild b_c" in result.output and "5.19615" in result.output
    assert "escaped" in result.output
    bad = CliRunner().invoke(app, ["lens", "--config", str(cfg_path), "--impact-range", "12,6"])
    assert bad.exit_code == 1


def test_parse_impact_range_validation() -> None:
    assert lens_cmd.parse_impact_range("3.5, 20") == (3.5, 20.0)
    for text in ("3", "a,b", "5,5", "-1,4"):
        with pytest.raises(Exception):
            lens_cmd.parse_impact_range(text)
