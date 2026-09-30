"""Smoke tests for the scripts that produce the results of research.md.

They run each study at a tiny size so that a change to the engine cannot
silently break the paper's reproduction path. The full studies are run with
scripts/reproduce_paper.py.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_winding_prediction_schwarzschild_and_ordering() -> None:
    w = _load("study_winding")
    gamma, r_ph, xi = w.predicted_gamma(0.0, True)
    assert gamma == pytest.approx(1.0, rel=1e-6)
    assert r_ph == pytest.approx(3.0, rel=1e-12)
    # prograde orbits are less stable per radian than retrograde ones for a > 0
    assert w.predicted_gamma(0.9, True)[0] < 1.0 < w.predicted_gamma(0.9, False)[0]


def test_winding_measurement_matches_prediction_small() -> None:
    w = _load("study_winding")
    m = w.measured_rate(0.0, True, [1e-3, 1e-4, 1e-5], rtol=1e-10)
    assert set(m["states"]) == {"ESCAPED"}
    assert m["rate_per_decade"] == pytest.approx(math.log(10) / (2 * math.pi), rel=5e-3)


@pytest.mark.parametrize("spin", [0.0, 0.9])
def test_oracle_matches_traced_mask_at_tight_tolerance(spin: float) -> None:
    pytest.importorskip("numba")
    o = _load("study_oracle")
    with np.errstate(all="ignore"):
        row, _ = o.run_one(spin, 60.0, 24, 1e-6, 1000.0)
    assert row["disagree"] == 0
    assert row["failed"] == 0
    assert row["traced_edge"]["rms_px"] == pytest.approx(row["oracle_edge"]["rms_px"])
    assert row["traced_edge"]["rms_px"] < 0.5


def test_oracle_long_observer_distance_has_affine_budget() -> None:
    pytest.importorskip("numba")
    o = _load("study_oracle")
    with np.errstate(all="ignore"):
        row, _ = o.run_one(0.9, 60.0, 16, 1e-6, 1.0e4)
    assert "MAX_AFFINE_PARAMETER" not in row["states"]
    assert row["disagree"] == 0


def test_convergence_single_image() -> None:
    pytest.importorskip("numba")
    c = _load("study_shadow_convergence")
    with np.errstate(all="ignore"):
        row = c.run_one(0.5, 60.0, 32, 1e-6, 8.0, 1000.0, "numba")
    assert row["failed_rays"] == 0
    assert 0.0 < row["rms_px"] < 0.5
    assert row["area_rel_err"] < 0.05
