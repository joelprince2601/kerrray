"""Tests for kerrray.physics.frame_dragging and the EXP-005 / EXP-003 drivers (PROJECT.md sections 14, 37).

Expected behaviour is checked through computed structure, not stored numbers:
the L_z = 0 azimuth is exactly antisymmetric in a and zero at a = 0, the
equatorial odd part has the sign and 1/b^2 approach of the leading-order
weak-field result (Sereno and De Luca 2006; docs/experiments_kerr.md), and the
integrated azimuth agrees with the quadrature of the separated equations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kerrray.experiments import frame_dragging as fd_exp
from kerrray.experiments import spin_sweep
from kerrray.experiments.base import RunRecord
from kerrray.geometry import kerr, outer_horizon
from kerrray.photons.orbits import critical_impact_parameters, radial_potential
from kerrray.physics.frame_dragging import (
    azimuth_quadrature,
    leading_order_asymmetry,
    leading_order_polar_drag,
    turning_point_radius,
)
from kerrray.utils.config import load_config
from kerrray.validation.schwarzschild import CRITICAL_IMPACT_REFERENCE

REPO = Path(__file__).resolve().parents[1]


def _cfg(tmp: Path, extra: dict[str, Any] | None = None) -> Any:
    overrides = {"experiment.output_dir": str(tmp / "runs"), "experiment.report_dir": str(tmp / "reports")}
    overrides.update(extra or {})
    return load_config(REPO / "configs" / "kerr.yaml", overrides)


@pytest.fixture(scope="module")
def fd_run(tmp_path_factory: pytest.TempPathFactory) -> RunRecord:
    tmp = tmp_path_factory.mktemp("fd")
    return fd_exp.run(_cfg(tmp, {"experiment.parameters.frame_dragging.scaling_impact_parameters": [20.0, 40.0]}))


def test_turning_point_radius_is_a_root_outside_the_horizon() -> None:
    st = kerr(1.0, 0.9)
    b_pro, _ = critical_impact_parameters(st)
    r_t = turning_point_radius(st, b_pro * 1.01, 0.0)
    assert r_t is not None and r_t > outer_horizon(st)
    assert abs(float(radial_potential(st, r_t, b_pro * 1.01, 0.0))) <= 1e-8 * r_t**4
    assert turning_point_radius(st, b_pro * 0.99, 0.0) is None


def test_azimuth_quadrature_is_antisymmetric_for_polar_rays() -> None:
    plus = azimuth_quadrature(kerr(1.0, 0.9), 0.0, 30.0**2, 1000.0, 1000.0)
    minus = azimuth_quadrature(kerr(1.0, -0.9), 0.0, 30.0**2, 1000.0, 1000.0)
    assert plus > 0.0 and minus == pytest.approx(-plus, rel=1e-12)
    assert azimuth_quadrature(kerr(1.0, 0.0), 0.0, 30.0**2, 1000.0, 1000.0) == 0.0
    with pytest.raises(ValueError):
        azimuth_quadrature(kerr(1.0, 0.0), 1.0, 0.0, 1000.0, 1000.0)  # b = 1 < b_c: captured


def test_leading_order_signs() -> None:
    st = kerr(1.0, 0.5)
    assert leading_order_asymmetry(st, 10.0) < 0.0
    assert leading_order_polar_drag(st, 10.0) > 0.0
    assert leading_order_polar_drag(kerr(1.0, -0.5), 10.0) < 0.0


def test_main_family_capture_asymmetry(fd_run: RunRecord) -> None:
    rays = {(r["family"], r["spin"]): r for r in fd_run.results["rays"]}
    b = fd_run.results["scaling"][0]["b"]
    b_ret = critical_impact_parameters(kerr(1.0, 0.9))[1]
    # b lies between the Schwarzschild and the retrograde critical values: only the retrograde ray is captured
    assert CRITICAL_IMPACT_REFERENCE < b < b_ret
    assert rays[("equatorial", 0.9)]["state"] == "ESCAPED"
    assert rays[("equatorial", 0.0)]["state"] == "ESCAPED"
    assert rays[("equatorial", -0.9)]["state"] == "CAPTURED"
    (pair,) = fd_run.results["pairs"]
    assert math.isnan(float(pair["odd_part"])) and math.isnan(float(pair["even_part"]))
    # prograde photon passes farther out and is deflected less than the Schwarzschild one
    assert rays[("equatorial", 0.9)]["deflection"] < rays[("equatorial", 0.0)]["deflection"]
    assert rays[("equatorial", 0.9)]["closest_approach"] > rays[("equatorial", 0.0)]["closest_approach"]


def test_polar_family_is_pure_frame_dragging(fd_run: RunRecord) -> None:
    (pair,) = fd_run.results["pairs"]
    assert pair["polar_zero"] == 0.0
    assert pair["polar_plus"] > 0.0  # co-rotates with the hole
    assert abs(pair["polar_antisymmetry_residual"]) <= 1e-12 * abs(pair["polar_plus"])
    assert abs(pair["polar_plus"] / pair["polar_estimate"] - 1.0) <= 0.1  # weak field at b ~ 80 M


def test_escaped_rays_match_quadrature_and_conserve(fd_run: RunRecord) -> None:
    for ray in (*fd_run.results["rays"], *fd_run.results["scaling_rays"]):
        if ray["state"] != "ESCAPED":
            continue
        assert abs(float(ray["quadrature_difference"])) <= 1e-6
        assert ray["max_null_error"] <= 1e-7 and ray["max_carter_drift"] <= 1e-7


def test_odd_part_sign_and_leading_order_approach(fd_run: RunRecord) -> None:
    rows = [r for r in fd_run.results["scaling"] if not math.isnan(float(r["odd_part"]))]
    assert [r["b"] for r in rows] == [20.0, 40.0]
    for r in rows:
        assert r["odd_part"] < 0.0  # prograde (a L_z > 0) sweeps less azimuth
        assert r["ratio"] > 1.0
    assert abs(rows[1]["ratio"] - 1.0) < abs(rows[0]["ratio"] - 1.0)  # approaches -8 a M / b^2
    assert -3.5 < fd_run.results["odd_part_exponent"] < -2.0


def test_frame_dragging_outputs(fd_run: RunRecord) -> None:
    for name in ("report.md", "frame_dragging_trajectories.png", "frame_dragging_azimuth.png",
                 "frame_dragging_scaling.png"):
        assert (fd_run.report_dir / name).is_file(), name


def test_spin_sweep(tmp_path: Path) -> None:
    rec = spin_sweep.run(_cfg(tmp_path))
    rows = rec.results["rows"]
    spins = np.array([r["spin"] for r in rows])
    assert np.all(np.diff(spins) > 0)
    col = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    assert np.all(np.diff(col["r_plus"]) < 0) and np.all(np.diff(col["b_c_prograde"]) < 0)
    assert np.all(np.diff(col["b_c_retrograde"]) > 0) and np.all(np.diff(col["r_isco_prograde"]) < 0)
    assert np.all(col["r_ph_prograde"] > col["r_plus"]) and np.all(col["r_isco_prograde"] > col["r_ph_prograde"])
    zero = rows[0]
    assert zero["spin"] == 0.0
    assert zero["b_c_prograde"] == pytest.approx(CRITICAL_IMPACT_REFERENCE, rel=1e-12)
    assert zero["shadow_centroid_alpha"] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(col["shadow_centroid_alpha"]) > 0)
    for name in ("report.md", "spin_sweep_radii.png", "spin_sweep_critical_b.png", "spin_sweep_shadows.png"):
        assert (rec.report_dir / name).is_file()
