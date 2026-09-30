"""Tests for kerrray.validation.schwarzschild (EXP-001, EXP-002; PROJECT.md section 12).

Rays start and escape at 50 M and the searches use rtol 1e-8 and a bracket
tolerance of 1e-6 M so that the module runs in seconds; the documented
references 3M and 3 sqrt(3) M (MTW 1973 section 25.6) are only compared with
the computed values.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.validation.schwarzschild import (
    CRITICAL_IMPACT_REFERENCE,
    PHOTON_SPHERE_REFERENCE,
    RK4_FORMAL_ORDER,
    SchwarzschildValidationParams,
    critical_impact_experiment,
    photon_sphere_experiment,
    rk4_step_convergence,
    rtol_convergence,
)

INTEG = IntegratorOptions(rtol=1e-8, atol=1e-10)
TERM = TerminationOptions(horizon_epsilon=1e-6, escape_radius=50.0)
TOL = 1e-6
R0 = 50.0


def test_references_are_the_documented_values() -> None:
    assert PHOTON_SPHERE_REFERENCE == 3.0
    assert CRITICAL_IMPACT_REFERENCE == pytest.approx(math.sqrt(27.0), rel=1e-15)


def test_photon_sphere_bisection_recovers_3M() -> None:
    res = photon_sphere_experiment(INTEG, TERM, r_lo=2.5, r_hi=4.0, tol=TOL, n_probe=15)
    assert res.width <= TOL
    assert res.r_lo <= res.radius <= res.r_hi
    assert abs(res.radius - PHOTON_SPHERE_REFERENCE) <= TOL
    assert res.iterations >= 1 and res.n_rays > 2 and res.runtime_s > 0.0


def test_critical_impact_bisection_recovers_3sqrt3M() -> None:
    res = critical_impact_experiment(INTEG, TERM, b_lo=4.0, b_hi=7.0, tol=TOL, r0=R0, n_probe=15)
    assert res.width <= TOL
    assert abs(res.b_c - CRITICAL_IMPACT_REFERENCE) <= TOL
    assert res.launch_radius == R0 and res.horizon_crossings == 0


def test_classical_bisection_halves_the_bracket() -> None:
    """n_probe = 1 is plain bisection: one ray per iteration, width 3 / 2^k (loose tolerance, fast)."""
    one = critical_impact_experiment(INTEG, TERM, b_lo=4.0, b_hi=7.0, tol=0.05, r0=R0, n_probe=1)
    assert one.n_rays == 2 + one.iterations
    assert one.width == pytest.approx(3.0 / 2**one.iterations, rel=1e-12)
    assert one.b_lo < CRITICAL_IMPACT_REFERENCE < one.b_hi


@pytest.mark.parametrize("bracket", [(5.5, 7.0), (4.0, 5.0)])
def test_invalid_impact_bracket_raises(bracket: tuple[float, float]) -> None:
    with pytest.raises(RuntimeError, match="bracket"):
        critical_impact_experiment(INTEG, TERM, b_lo=bracket[0], b_hi=bracket[1], tol=TOL, r0=R0)


def test_invalid_photon_sphere_bracket_raises() -> None:
    with pytest.raises(RuntimeError, match="bracket"):
        photon_sphere_experiment(INTEG, TERM, r_lo=3.2, r_hi=4.0, tol=TOL)
    with pytest.raises(ValueError):
        photon_sphere_experiment(INTEG, TERM, r_lo=4.0, r_hi=3.0, tol=TOL)


def test_rk4_step_convergence_is_fourth_order() -> None:
    params = SchwarzschildValidationParams(step_sizes=[0.4, 0.2], convergence_launch_radius=20.0,
                                           convergence_halfwidth=1e-4, convergence_tol=1e-9)
    study = rk4_step_convergence(params, INTEG, TERM, center=CRITICAL_IMPACT_REFERENCE)
    assert all(e > params.convergence_tol for e in study.errors)
    assert study.errors[1] < study.errors[0]
    assert abs(study.fitted_order - RK4_FORMAL_ORDER) <= params.order_tolerance
    # fixed-step captured rays cross the horizon in one step and are relabelled (kerr_bisection docstring)
    assert sum(d["horizon_crossings"] for d in study.extra) > 0


def test_rtol_convergence_error_decreases() -> None:
    params = SchwarzschildValidationParams(rtol_levels=[1e-5, 1e-7], convergence_launch_radius=20.0,
                                           convergence_halfwidth=1e-4, convergence_tol=1e-9)
    study = rtol_convergence(params, replace(INTEG, rtol=1e-9, atol=1e-11), TERM, center=CRITICAL_IMPACT_REFERENCE)
    assert study.levels == (1e-5, 1e-7)
    assert study.errors[1] < study.errors[0]
    assert study.fitted_order > 0.5
