"""Tests for kerrray.validation.kerr, kerr_bisection and kerr_report (PROJECT.md section 13).

One small validation run (module fixture) is inspected check by check: rays
start and escape at 50 M, rtol 1e-8, bisection to 1e-6 M, one spin per
family. The only hard-coded physics number is the documented reference
3 sqrt(3) M; every other reference is computed by the orbit code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import kerr
from kerrray.photons.orbits import critical_impact_parameters
from kerrray.utils.config import load_config
from kerrray.validation import ValidationReport
from kerrray.validation.kerr import KerrValidationParams, check_horizon, run_kerr_validation
from kerrray.validation.kerr_bisection import bisect_critical_impacts
from kerrray.validation.schwarzschild import CRITICAL_IMPACT_REFERENCE

REPO = Path(__file__).resolve().parents[1]
R0 = 50.0
FAST = {
    "integration.rtol": 1e-8,
    "integration.atol": 1e-10,
    "termination.escape_radius": R0,
    "experiment.parameters.validation": {
        "small_spins": [1e-2, 1e-3],
        "critical_spins": [0.9],
        "conservation_spins": [0.99],
        "near_extremal_spins": [0.999],
        "critical_b_tol": 1e-5,
        "conservation_tol": 1e-7,
        "launch_radius": R0,
        "impact_bracket": [2.0, 8.0],
        "reversibility_tol": 1e-5,
    },
}


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> ValidationReport:
    out = tmp_path_factory.mktemp("kerr")
    cfg = load_config(REPO / "configs" / "kerr.yaml",
                      {**FAST, "experiment.output_dir": str(out / "runs"), "experiment.report_dir": str(out / "reports")})
    return run_kerr_validation(cfg)


def test_all_checks_pass_and_are_recorded(report: ValidationReport) -> None:
    assert [c.name for c in report.checks] == ["horizon", "schwarzschild_limit", "critical_impact", "conservation",
                                               "near_extremal", "reversibility"]
    assert report.all_passed, {c.name: c.error for c in report.checks if not c.passed}
    run = report.run
    assert run is not None
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment"] == "validate_kerr" and manifest["status"] == "completed"
    assert manifest["results"]["all_passed"] is True
    for name in ("report.md", "summary.json", "critical_impact.png", "conservation.png", "near_extremal.png",
                 "schwarzschild_limit.png"):
        assert (run.report_dir / name).is_file(), name


def test_horizon_check_covers_every_spin(report: ValidationReport) -> None:
    rows = report.check("horizon").computed["rows"]
    assert {r["spin"] for r in rows} >= {0.9, 0.99, 0.999, 1e-2, 1e-3}
    for r in rows:
        a = r["spin"]
        assert r["r_plus"] + r["r_minus"] == pytest.approx(2.0, abs=1e-14)  # Vieta: r+ + r- = 2M
        assert r["r_plus"] * r["r_minus"] == pytest.approx(a * a, abs=1e-14)  # Vieta: r+ r- = a^2


def test_horizon_check_fails_on_impossible_tolerance() -> None:
    check = check_horizon(1.0, [0.5, 0.9], KerrValidationParams(horizon_tol=0.0))
    assert check.error["max"] >= 0.0
    assert check.passed == (check.error["max"] == 0.0)


def test_schwarzschild_limit_is_linear_in_spin(report: ValidationReport) -> None:
    c = report.check("schwarzschild_limit")
    assert c.reference["b_c_schwarzschild"] == pytest.approx(CRITICAL_IMPACT_REFERENCE)
    spins = np.array(c.computed["spins"])
    pro, ret = np.array(c.computed["b_prograde"]), np.array(c.computed["b_retrograde"])
    assert np.all(pro < CRITICAL_IMPACT_REFERENCE) and np.all(ret > CRITICAL_IMPACT_REFERENCE)
    for key in ("prograde", "retrograde"):
        assert abs(c.computed["exponents"][key] - 1.0) <= 0.1
    # the deviation per unit spin is the same at both spins (linear scaling), slope from the derived closed form
    slope = c.reference["odd_part_slope_derived"]
    assert np.allclose(0.5 * (pro - ret) / spins, slope, rtol=1e-2)


def test_critical_impact_matches_derived_values(report: ValidationReport) -> None:
    (row,) = report.check("critical_impact").computed["rows"]
    d_pro, d_ret = critical_impact_parameters(kerr(1.0, 0.9))
    assert row["b_prograde"] < row["b_retrograde"]  # prograde photons get closer before capture
    assert abs(row["b_prograde"] - d_pro) / d_pro <= 1e-5
    assert abs(row["b_retrograde"] - d_ret) / d_ret <= 1e-5
    assert row["bracket_width"] <= 1e-6


def test_conservation_near_extremal_and_reversibility(report: ValidationReport) -> None:
    (cons,) = report.check("conservation").computed["rows"]
    assert cons["state"] == "ESCAPED"
    assert cons["max_drift"]["energy"] <= 1e-12 and cons["max_drift"]["lz"] <= 1e-12  # structural (Hamiltonian form)
    assert 0.0 < cons["max_drift"]["carter"] <= 1e-7
    (near,) = report.check("near_extremal").computed["rows"]
    assert near["r_turn_analytic"] > near["r_plus"]
    last = near["runs"][-1]
    assert last["state"] == "ESCAPED"
    assert near["r_plus"] < last["closest_approach"] < near["r_ph_prograde"] * 1.5
    rev = report.check("reversibility")
    assert rev.computed["forward_state"] == rev.computed["backward_state"] == "MAX_AFFINE_PARAMETER"
    assert max(rev.error.values()) <= 1e-5


def test_bisection_rejects_invalid_bracket_and_arguments() -> None:
    st = kerr(1.0, 0.5)
    integ, term = IntegratorOptions(rtol=1e-8, atol=1e-10), TerminationOptions(escape_radius=R0)
    with pytest.raises(ValueError):
        bisect_critical_impacts(st, integ, term, r0=R0, b_lo=5.0, b_hi=4.0, tol=1e-3, max_iterations=5)
    with pytest.raises(RuntimeError, match="bracket"):
        bisect_critical_impacts(st, integ, term, r0=R0, b_lo=7.0, b_hi=8.0, tol=1e-3, max_iterations=5,
                                directions=(True,))


def test_bisection_iteration_cap() -> None:
    st = kerr(1.0, 0.5)
    integ, term = IntegratorOptions(rtol=1e-8, atol=1e-10), TerminationOptions(escape_radius=R0)
    (res,) = bisect_critical_impacts(st, integ, term, r0=R0, b_lo=2.0, b_hi=8.0, tol=1e-12, max_iterations=1,
                                     n_probe=5, directions=(True,))
    assert res.iterations == 1 and res.width <= 6.0 / 6.0 + 1e-12
    assert res.b_lo < critical_impact_parameters(st)[0] < res.b_hi
    assert math.isfinite(res.b_c)
