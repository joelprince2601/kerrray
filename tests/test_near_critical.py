"""Tests for the near-critical experiment (kerrray.experiments.near_critical, EXP-007)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kerrray.experiments.near_critical import (
    NearCriticalParams,
    bisect_critical_impact,
    build_families,
    run,
    turns_per_decade,
)
from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime
from kerrray.utils.config import load_config
from kerrray.utils.config_parsing import ConfigError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
R0 = 30.0


def test_turns_per_decade_of_synthetic_data() -> None:
    offsets = [1e-1, 1e-2, 1e-3]
    turns = [1.0 + 0.5 * k for k in range(3)]
    assert turns_per_decade(offsets, turns) == pytest.approx(0.5)
    assert math.isnan(turns_per_decade([1e-2], [3.0]))
    assert math.isnan(turns_per_decade([1e-2, 1e-2], [3.0, 3.1]))


def test_params_validation() -> None:
    with pytest.raises(ConfigError):
        NearCriticalParams(offsets=[0.0])
    with pytest.raises(ConfigError):
        NearCriticalParams(tolerances=[2.0])
    with pytest.raises(ConfigError):
        NearCriticalParams(launch_radius=5.0)
    with pytest.raises(ConfigError):
        NearCriticalParams(spins=[1.0])


def test_bisection_brackets_the_schwarzschild_critical_impact_parameter() -> None:
    st = Spacetime(mass=1.0, spin=0.0)
    integ = IntegratorOptions(method="rk45", rtol=1e-7, atol=1e-9)
    term = TerminationOptions(horizon_epsilon=1e-4, escape_radius=R0)
    tol = 2e-2
    bis = bisect_critical_impact(st, R0, integ, term, b_lo=3.0, b_hi=8.0, tol=tol, max_iterations=60)
    assert bis["b_hi"] - bis["b_lo"] <= tol
    # reference value 3 sqrt(3) M (PROJECT.md section 12) checks the computed bracket
    assert bis["b_lo"] - 1e-6 <= 3.0 * math.sqrt(3.0) <= bis["b_hi"] + 1e-6


def test_kerr_families_use_prograde_and_retrograde_critical_values() -> None:
    cfg = load_config(CONFIGS / "near_critical.yaml")
    params = NearCriticalParams(spins=[0.9], launch_radius=R0)
    fams = build_families(params, cfg, IntegratorOptions(), TerminationOptions())
    assert [f.prograde for f in fams] == [True, False]
    assert fams[0].b_c < fams[1].b_c  # prograde photons can approach closer
    assert all(f.b_c_source == "critical_impact_parameters" for f in fams)


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("near_critical")
    cfg = load_config(CONFIGS / "near_critical.yaml", {
        "experiment.output_dir": str(tmp / "runs"),
        "experiment.report_dir": str(tmp / "reports"),
        "termination.escape_radius": R0,
        "integration.rtol": 1e-8,
        "integration.atol": 1e-10,
        "experiment.parameters.near_critical": {
            "offsets": [1e-1, 1e-2, 1e-3], "launch_radius": R0, "tolerances": [1e-6, 1e-8],
            "spins": [0.0], "bisection_tol": 1e-4,
        },
    })
    return run(cfg)


def test_near_critical_outcomes_and_turns(record) -> None:
    res = record.results
    assert len(res["families"]) == 1 and res["families"][0]["b_c_source"] == "bisection"
    rows = res["rows"]
    assert len(rows) == 3 * 2 * 2
    for row in rows:
        assert row["state"] == ("ESCAPED" if row["sign"] > 0 else "CAPTURED")
        assert row["affine_length"] > 0 and row["n_steps"] > 0 and row["runtime_s"] > 0
    for sign in (1, -1):
        sel = sorted((r for r in rows if r["sign"] == sign and r["rtol"] == 1e-8), key=lambda r: -r["offset"])
        turns = [r["turns"] for r in sel]
        assert np.all(np.diff(turns) > 0), turns  # more winding as b approaches b_c


def test_near_critical_fits_come_from_the_data(record) -> None:
    res = record.results
    fits = [f for f in res["fits"] if f["rtol"] == 1e-8]
    assert len(fits) == 2
    for fit in fits:
        assert fit["n_points"] == 3
        refit = np.polyfit(-np.log10(fit["offsets"]), fit["turns"], 1)[0]
        assert fit["turns_per_decade"] == pytest.approx(refit)
        assert fit["slope_turns_vs_log10_delta"] == pytest.approx(-refit)
        # comparison with the strong-deflection-limit value, not an input: within 25 % at these offsets
        assert fit["turns_per_decade"] == pytest.approx(res["schwarzschild_turns_per_decade_comparison"], rel=0.25)
    assert res["n_flips"] == sum(1 for f in res["flips"] if f["flip"])
    assert (record.report_dir / "turns_vs_offset.png").is_file()
    assert (record.report_dir / "report.md").is_file()
