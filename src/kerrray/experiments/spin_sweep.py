"""EXP-003 spin sweep (PROJECT.md section 37; docs/experiments_kerr.md section 4).

For each spin of ``experiment.parameters.spin_sweep.spins`` (default ``[0,
0.25, 0.5, 0.75, 0.9, 0.99]``) the driver tabulates and plots the
characteristic radii and impact parameters of the Kerr hole, all computed by
the tested geometry and orbit code:

* horizons ``r_+-`` (:func:`kerrray.geometry.horizon_radii`) and the
  equatorial ergosphere radius (``2M`` for every spin);
* prograde and retrograde equatorial photon-orbit radii and critical impact
  parameters (:mod:`kerrray.photons.orbits`, Bardeen, Press and Teukolsky
  1972, eq. 2.18 and docs/derivations.md section 6);
* prograde and retrograde ISCO radii (BPT 1972, eq. 2.21);
* the analytic shadow boundary (Bardeen 1973) seen from the configured
  observer inclination, its horizontal extent and centroid shift.

The shadow curve here is the analytic ``r_o -> infinity`` curve, labelled as
such; the numerically traced shadow is produced by ``kerrray shadow``. No ray
is integrated by this driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geometry import ergosphere_radius, horizon_radii, kerr
from kerrray.photons.orbits import critical_impact_parameters, equatorial_photon_orbit_radius, isco_radius, shadow_curve
from kerrray.reporting import plots
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import KerrRayConfig

__all__ = ["EXPERIMENT_NAME", "SpinSweepParams", "run", "spin_row"]

EXPERIMENT_NAME: Final[str] = "spin_sweep"
plt = plots.plt


@dataclass(frozen=True)
class SpinSweepParams:
    """``experiment.parameters.spin_sweep`` (D-009)."""

    spins: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 0.9, 0.99])


def spin_row(mass: float, spin: float, inclination_deg: float) -> dict[str, Any]:
    """Characteristic radii, impact parameters and shadow extent for one spin (units of M)."""
    st = kerr(mass, spin)
    r_plus, r_minus = horizon_radii(st)
    b_pro, b_ret = critical_impact_parameters(st)
    alpha, beta = shadow_curve(st, inclination_deg)
    return {
        "spin": spin,
        "r_plus": r_plus,
        "r_minus": r_minus,
        "r_ergo_equator": float(ergosphere_radius(st, 0.5 * math.pi)),
        "r_ph_prograde": equatorial_photon_orbit_radius(st, True),
        "r_ph_retrograde": equatorial_photon_orbit_radius(st, False),
        "b_c_prograde": b_pro,
        "b_c_retrograde": b_ret,
        "r_isco_prograde": isco_radius(st, True),
        "r_isco_retrograde": isco_radius(st, False),
        "shadow_alpha_min": float(np.min(alpha)),
        "shadow_alpha_max": float(np.max(alpha)),
        "shadow_width": float(np.max(alpha) - np.min(alpha)),
        "shadow_height": float(np.max(beta) - np.min(beta)),
        "shadow_centroid_alpha": float(0.5 * (np.max(alpha) + np.min(alpha))),
    }


def _figures(ctx: ExperimentContext, rows: list[dict[str, Any]], inclination: float) -> list[Any]:
    s = np.array([r["spin"] for r in rows])

    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows])

    radii = plots.plot_lines(ctx.report_dir / "spin_sweep_radii.png", {
        "r_+": (s, col("r_plus")), "r_-": (s, col("r_minus")),
        "photon orbit (pro)": (s, col("r_ph_prograde")), "photon orbit (retro)": (s, col("r_ph_retrograde")),
        "ISCO (pro)": (s, col("r_isco_prograde")), "ISCO (retro)": (s, col("r_isco_retrograde")),
    }, xlabel="spin a/M", ylabel="radius [M]", title="Characteristic radii against spin")
    impact = plots.plot_lines(ctx.report_dir / "spin_sweep_critical_b.png", {
        "b_c prograde": (s, col("b_c_prograde")), "b_c retrograde": (s, col("b_c_retrograde")),
    }, xlabel="spin a/M", ylabel="critical impact parameter [M]", title="Equatorial critical impact parameters")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    mass = ctx.cfg.black_hole.mass
    for index, row in enumerate(rows):
        a, b = shadow_curve(kerr(mass, row["spin"]), inclination)
        ax.plot(np.append(a, a[0]), np.append(b, b[0]), label=f"a = {row['spin']:g}", **plots.series_style(index))
    ax.set_aspect("equal")
    plots._style_axes(ax, "alpha [M]", "beta [M]", f"Analytic shadow boundaries, inclination {inclination:g} deg")
    ax.legend(loc="best", frameon=False, fontsize=8)
    shadows = plots._save(fig, ctx.report_dir / "spin_sweep_shadows.png", plots.DEFAULT_DPI)
    return [radii, impact, shadows]


def _body(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    params = ctx.parameters("spin_sweep", SpinSweepParams)
    inclination = cfg.observer.inclination_deg
    rows = [spin_row(cfg.black_hole.mass, float(a), inclination) for a in params.spins]
    figures = _figures(ctx, rows, inclination)
    first, last = rows[0], rows[-1]
    sections = ReportSections(
        objective="EXP-003: how the horizon, photon orbits, ISCO, critical impact parameters and the analytic "
                  "shadow change with spin (PROJECT.md section 37).",
        mathematical_model="Kerr horizons r_+- = M +- sqrt(M^2 - a^2); equatorial photon orbits and critical impact "
                           "parameters from R = R' = 0 (docs/derivations.md sections 5-6); ISCO from BPT 1972 eq. "
                           "2.21; Bardeen 1973 shadow curve for an observer at infinity.",
        numerical_method="Closed forms and bracketed root finding in kerrray.photons.orbits (tested against the "
                         "references in tests/test_orbits.py); no ray integration.",
        parameters=f"- spins: {params.spins}\n- observer inclination: {inclination:g} deg\n- mass: {cfg.black_hole.mass:g}",
        results=f"From a = {first['spin']:g} to a = {last['spin']:g}: r_+ {first['r_plus']:.6g} -> {last['r_plus']:.6g} M, "
                f"prograde b_c {first['b_c_prograde']:.6g} -> {last['b_c_prograde']:.6g} M, retrograde b_c "
                f"{first['b_c_retrograde']:.6g} -> {last['b_c_retrograde']:.6g} M, shadow centroid "
                f"{first['shadow_centroid_alpha']:.4g} -> {last['shadow_centroid_alpha']:.4g} M.",
        error_analysis="All values are closed forms or root-finder results at 1e-14 tolerance; the orbit code is "
                       "validated to 1e-12 against BPT 1972 in tests/test_orbits.py, and the bisected b_c by "
                       "`kerrray validate kerr`.",
        interpretation="Spin separates the prograde and retrograde photon orbits and critical impact parameters "
                       "(the prograde ones shrink towards the horizon), which shifts and flattens the shadow on the "
                       "prograde side; the numbers are in the table.",
        limitations="Analytic observer at infinity; equatorial orbits only in the radius table; the traced, "
                    "finite-distance shadow comes from `kerrray shadow`.",
        reproducibility=f"Run {ctx.run_id}; configuration in {ctx.run_dir / 'config.yaml'}.",
    )
    write_report(ctx.report_dir, sections, figures, [markdown_table(rows, precision=8)], title="EXP-003 spin sweep")
    return {"rows": rows, "inclination_deg": inclination, "figures": [str(f) for f in figures]}


def run(cfg: KerrRayConfig) -> RunRecord:
    """Run EXP-003 for ``cfg`` (docs/architecture.md section 6)."""
    return run_experiment(EXPERIMENT_NAME, cfg, _body)
