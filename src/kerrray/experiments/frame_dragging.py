"""EXP-005 frame-dragging experiment (PROJECT.md sections 14 and 37; docs/experiments_kerr.md).

For every spin of ``experiment.parameters.frame_dragging.spins`` (default
``[-0.9, 0.0, 0.9]``) the *same* photons are integrated: an equatorial ray
with ``L_z = +b E`` launched inward from ``launch_radius`` and a polar
``L_z = 0`` ray launched parallel to the spin axis
(:func:`kerrray.physics.frame_dragging.frame_dragging_experiment`). The
driver measures the azimuth swept, the closest approach, the deflection, the
``+a`` / ``-a`` asymmetry and its comparison with the exact quadrature and the
leading-order estimates, repeats the equatorial family for
``scaling_impact_parameters`` to show the ``1/b^2`` scaling of the odd part,
and writes three figures (trajectories, azimuth and accumulated shift versus
the affine parameter, scaling) plus ``report.md``. Why this setup isolates
frame dragging is explained in docs/experiments_kerr.md section 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geometry import kerr, outer_horizon
from kerrray.photons import to_cartesian
from kerrray.physics.frame_dragging import FrameDraggingResult, frame_dragging_experiment
from kerrray.reporting import plots
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import KerrRayConfig
from kerrray.validation import integrator_options_from_config, termination_options_from_config
from kerrray.validation.convergence import loglog_order

__all__ = ["EXPERIMENT_NAME", "FrameDraggingParams", "run", "scaling_rows"]

EXPERIMENT_NAME: Final[str] = "frame_dragging"
plt = plots.plt  # Agg backend selected by kerrray.reporting.plots


@dataclass(frozen=True)
class FrameDraggingParams:
    """``experiment.parameters.frame_dragging`` (D-009 keys first, then documented extensions)."""

    spins: list[float] = field(default_factory=lambda: [-0.9, 0.0, 0.9])
    impact_parameter: float = 6.0
    launch_radius: float = 1000.0
    scaling_impact_parameters: list[float] = field(default_factory=lambda: [10.0, 20.0, 40.0])
    polar_impact_parameter: float | None = None


def scaling_rows(results: list[FrameDraggingResult]) -> list[dict[str, Any]]:
    """One row per (b, |a|) pair: odd part, leading-order estimate and their ratio."""
    rows = []
    for res in results:
        for pair in res.pairs:
            ratio = pair.odd_part / pair.estimate if pair.estimate else math.nan
            rows.append({"b": res.impact_parameter, "spin": pair.spin, "state_plus": pair.state_plus,
                         "state_minus": pair.state_minus, "odd_part": pair.odd_part, "estimate": pair.estimate,
                         "ratio": ratio, "even_part": pair.even_part})
    return rows


def _trajectory_figure(path: Path, res: FrameDraggingResult) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5))
    half = 3.0 * res.impact_parameter
    for index, ray in enumerate(res.equatorial):
        x, y, _ = to_cartesian(ray.trajectory)
        axes[0].plot(x, y, label=f"a = {ray.spin:+g} ({ray.state})", **plots.series_style(index))
    for index, ray in enumerate(res.polar):
        _, y, z = to_cartesian(ray.trajectory)
        axes[1].plot(z, y, label=f"a = {ray.spin:+g}", **plots.series_style(index))
    mass = res.equatorial[0].trajectory.spacetime.mass
    for spin in sorted({abs(s) for s in res.spins}):  # horizons in the bl_to_cartesian embedding
        r_h = math.hypot(outer_horizon(kerr(mass, spin)), spin * mass)
        axes[0].add_patch(plots.Circle((0.0, 0.0), r_h, fill=False, color=plots.HORIZON_COLOR, linewidth=0.8))
    axes[0].set_xlim(-half, half)
    axes[0].set_ylim(-half, half)
    axes[0].set_aspect("equal")
    plots._style_axes(axes[0], "x [M]", "y [M]", f"Equatorial rays, L_z = +{res.impact_parameter:g} E (horizons r+)")
    plots._style_axes(axes[1], "z [M] (along the spin axis)", "y [M]",
                      f"L_z = 0 rays at {res.polar_impact_parameter:.3g} M from the axis")
    for ax in axes:
        ax.legend(loc="best", frameon=False, fontsize=8)
    return plots._save(fig, path, plots.DEFAULT_DPI)


def _azimuth_figure(path: Path, res: FrameDraggingResult) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    zero = next((r for r in res.equatorial if r.spin == 0.0), None)
    for index, ray in enumerate(res.equatorial):
        tr = ray.trajectory
        phi = tr.y[:, 3] - tr.y[0, 3]
        axes[0].plot(tr.lam, phi, label=f"a = {ray.spin:+g}", **plots.series_style(index))
        if zero is not None and ray.spin != 0.0:
            z = zero.trajectory
            lam = tr.lam[tr.lam <= z.lam[-1]]
            shift = np.interp(lam, tr.lam, phi) - np.interp(lam, z.lam, z.y[:, 3] - z.y[0, 3])
            axes[1].plot(lam, shift, label=f"a = {ray.spin:+g} minus a = 0", **plots.series_style(index))
    for index, ray in enumerate(res.polar):
        tr = ray.trajectory
        axes[2].plot(tr.lam, tr.y[:, 3] - tr.y[0, 3], label=f"a = {ray.spin:+g}", **plots.series_style(index))
    plots._style_axes(axes[0], "affine parameter lambda [M]", "phi - phi_0 [rad]", "Equatorial azimuth")
    plots._style_axes(axes[1], "affine parameter lambda [M]", "shift [rad]", "Accumulated shift relative to a = 0")
    plots._style_axes(axes[2], "affine parameter lambda [M]", "phi - phi_0 [rad]", "L_z = 0 azimuth (pure drag)")
    for ax in axes:
        ax.legend(loc="best", frameon=False, fontsize=8)
    return plots._save(fig, path, plots.DEFAULT_DPI)


def _scaling_figure(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    usable = [r for r in rows if math.isfinite(r["odd_part"])]
    if len(usable) < 2:
        return None
    b = np.array([r["b"] for r in usable])
    series = {"|odd part| (integrated)": (b, np.abs([r["odd_part"] for r in usable])),
              "8 |a| M / b^2 (leading order)": (b, np.abs([r["estimate"] for r in usable]))}
    return plots.plot_lines(path, series, xlabel="impact parameter b [M]", ylabel="|Delta phi(+a) - Delta phi(-a)| [rad]",
                            logx=True, logy=True, title="Odd (frame-dragging) part of the azimuth")


def _sections(ctx: ExperimentContext, params: FrameDraggingParams, main: FrameDraggingResult,
              rows: list[dict[str, Any]], exponent: float) -> ReportSections:
    cfg = ctx.cfg
    pair_lines = []
    for p in main.pairs:
        pair_lines.append(f"- |a| = {p.spin:g}: equatorial outcomes {p.state_plus} (+a) / {p.state_minus} (-a); "
                          f"odd part {p.odd_part:.6g} rad (leading order {p.estimate:.6g}); polar azimuth "
                          f"{p.polar_plus:.6g} / {p.polar_minus:.6g} rad (antisymmetry residual "
                          f"{p.polar_antisymmetry_residual:.3g}, leading order {p.polar_estimate:.6g}).")
    ratio_lines = [f"- b = {r['b']:g}, |a| = {r['spin']:g}: odd/estimate = {r['ratio']:.4g}" for r in rows]
    return ReportSections(
        objective="Demonstrate frame dragging (PROJECT.md section 14, EXP-005) with identical photons in spacetimes "
                  "of negative, zero and positive spin, and isolate the part of the motion caused by g_tphi.",
        mathematical_model="Kerr metric in Boyer-Lindquist coordinates; null geodesics in Hamiltonian form; the "
                           "separated equations of Carter (1968) for the exact azimuth quadrature; the leading-order "
                           "estimates -8aM/b^2 (equatorial odd part) and 4aM/b^2 (L_z = 0 ray) of "
                           "docs/experiments_kerr.md section 2.",
        numerical_method=f"Integrator {cfg.integration.method}, rtol {cfg.integration.rtol:g}, atol "
                         f"{cfg.integration.atol:g}; each ray recorded and compared with the quadrature of the "
                         "separated equations (kerrray.physics.frame_dragging.azimuth_quadrature).",
        parameters="\n".join(f"- `{k}`: {v}" for k, v in params.__dict__.items())
                   + f"\n- polar impact parameter used: {main.polar_impact_parameter:.6g} M",
        results="\n".join(pair_lines) + "\n\nOdd part against the leading-order estimate:\n" + "\n".join(ratio_lines),
        error_analysis="Integrated minus quadrature azimuth per ray (escaped rays): "
                       + ", ".join(f"{r.family} a={r.spin:+g}: {r.quadrature_difference:.3g}"
                                   for r in (*main.equatorial, *main.polar))
                       + f". Fitted exponent of |odd part| against b over the escaped pairs: {exponent:.4g}.",
        interpretation="The odd part changes sign with a and is the only part produced by g_tphi (docs/experiments_kerr.md "
                       "section 2); the L_z = 0 ray has no azimuthal motion at a = 0 and opposite azimuths for +-a, "
                       "so its azimuth is frame dragging alone. The approach of the odd part to the leading-order "
                       "value as b grows is given by the ratios above.",
        limitations="Captured rays have no comparable azimuth: the Boyer-Lindquist phi diverges at the horizon for a != 0 "
                    "(dphi/dlambda contains a P/Delta), so pairs with a captured ray report nan asymmetries. The "
                    "leading-order formulas are weak-field approximations (WHAT/WHY/LIMITATION in "
                    "kerrray.physics.frame_dragging). Finite launch radius: the azimuth beyond r0 is not included.",
        reproducibility=f"Run {ctx.run_id}; seed {cfg.experiment.seed}; configuration in {ctx.run_dir / 'config.yaml'}.",
    )


def _body(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    params = ctx.parameters("frame_dragging", FrameDraggingParams)
    integ, term = integrator_options_from_config(cfg), termination_options_from_config(cfg)
    mass = cfg.black_hole.mass
    kwargs = {"integ": integ, "term": term, "r0": params.launch_radius * mass, "mass": mass,
              "polar_b": params.polar_impact_parameter}
    main = frame_dragging_experiment(params.spins, params.impact_parameter * mass, **kwargs)
    ctx.logger.info("frame_dragging: main family (%.1f s)", ctx.timer.lap("main"))
    extra = [frame_dragging_experiment(params.spins, b * mass, **kwargs) for b in params.scaling_impact_parameters]
    ctx.logger.info("frame_dragging: scaling families (%.1f s)", ctx.timer.lap("scaling"))
    rows = scaling_rows([main, *extra])
    usable = [r for r in rows if math.isfinite(r["odd_part"])]
    exponent = loglog_order([r["b"] for r in usable], np.abs([r["odd_part"] for r in usable])) if usable else math.nan
    figures = [_trajectory_figure(ctx.report_dir / "frame_dragging_trajectories.png", main),
               _azimuth_figure(ctx.report_dir / "frame_dragging_azimuth.png", main)]
    scaling_fig = _scaling_figure(ctx.report_dir / "frame_dragging_scaling.png", rows)
    if scaling_fig is not None:
        figures.append(scaling_fig)
    ray_rows = [r.as_row() for r in (*main.equatorial, *main.polar)]
    tables = [markdown_table(ray_rows, columns=["family", "spin", "impact_parameter", "state", "delta_phi", "turns",
                                                "closest_approach", "deflection", "quadrature_difference", "n_steps",
                                                "max_carter_drift", "max_null_error"], precision=8),
              markdown_table([p.as_row() for p in main.pairs], precision=8),
              markdown_table(rows, precision=6)]
    write_report(ctx.report_dir, _sections(ctx, params, main, rows, exponent), figures, tables,
                 title="EXP-005 frame dragging")
    return {"rays": ray_rows, "pairs": [p.as_row() for p in main.pairs], "scaling": rows,
            "odd_part_exponent": exponent, "polar_impact_parameter": main.polar_impact_parameter,
            "scaling_rays": [r.as_row() for res in extra for r in res.equatorial],
            "figures": [str(f) for f in figures]}


def run(cfg: KerrRayConfig) -> RunRecord:
    """Run EXP-005 for ``cfg`` (docs/architecture.md section 6)."""
    return run_experiment(EXPERIMENT_NAME, cfg, _body)
