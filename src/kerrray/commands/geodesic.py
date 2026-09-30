"""``kerrray geodesic``: integrate one photon and report its diagnostics (PROJECT.md sections 28 and 51).

Launch state. The photon starts at ``(r0, theta0, phi = 0)`` moving inward
with ``E = 1`` and

    L_z / E = xi  = s b sin(theta0),      Q / E^2 = eta = (b^2 - a^2) cos^2(theta0),

where ``s = +1`` for a prograde photon (``a L_z > 0``; ``L_z > 0`` when
``a = 0``) and ``-1`` for a retrograde one. Then the polar potential
``Theta(theta0) = eta + a^2 cos^2 theta0 - xi^2 cot^2 theta0`` (Carter 1968)
vanishes, i.e. the photon starts at a turning point of its polar motion, and
Bardeen's celestial coordinates of an observer at inclination ``theta0``
(docs/architecture.md section 1.3) are ``alpha = -xi / sin(theta0) = -s b``,
``beta = 0``: the ray is the one that reaches (or, traced backwards, leaves)
that observer's image plane on the horizontal axis at distance ``b`` from the
centre. For ``theta0 = 90`` deg it is the equatorial photon with impact
parameter ``b = |L_z|/E``; for ``a = 0`` ``b`` is the ordinary impact
parameter at every ``theta0``.

Defaults come from ``--config`` (``configs/kerr.yaml``): ``theta0`` is
``observer.inclination_deg``, ``r0`` is ``observer.radius`` and ``b`` is
``experiment.parameters.frame_dragging.impact_parameter``. The run goes
through :func:`kerrray.experiments.base.run_experiment` (name ``geodesic``),
so a manifest and the configuration copy are written; ``--plot`` adds
``reports/<run_id>/trajectory.png``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import typer

from kerrray.experiments.base import ExperimentContext, run_experiment
from kerrray.experiments.frame_dragging import FrameDraggingParams
from kerrray.geodesics import integrate, photon_from_constants
from kerrray.geometry import Spacetime, ergosphere_radius, kerr, outer_horizon
from kerrray.photons import TerminationState, Trajectory, azimuthal_winding, closest_approach, deflection_angle, to_cartesian
from kerrray.reporting import plots
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.commands.validate import DEFAULT_KERR_CONFIG, SET_HELP, load_command_config
from kerrray.utils.config import experiment_block
from kerrray.validation import integrator_options_from_config, termination_options_from_config

__all__ = ["FIGURE_FILENAME", "launch_constants", "launch_state", "register", "result_rows"]

FIGURE_FILENAME: Final[str] = "trajectory.png"


def launch_constants(st: Spacetime, theta_deg: float, b: float, prograde: bool) -> tuple[float, float]:
    """``(xi, eta)`` of the launch state (module docstring)."""
    if not 0.0 < theta_deg < 180.0:
        raise ValueError(f"theta must lie strictly between 0 and 180 degrees, got {theta_deg!r}")
    if not b >= 0.0:
        raise ValueError(f"impact parameter must be >= 0, got {b!r}")
    th = math.radians(theta_deg)
    sign = 1.0 if prograde == (st.a >= 0.0) else -1.0
    cos2 = 0.0 if theta_deg == 90.0 else math.cos(th) ** 2
    return sign * b * math.sin(th), (b * b - st.a * st.a) * cos2


def launch_state(st: Spacetime, r0: float, theta_deg: float, b: float, prograde: bool) -> np.ndarray:
    """Initial state ``y0`` (module docstring)."""
    xi, eta = launch_constants(st, theta_deg, b, prograde)
    return photon_from_constants(st, r0, math.radians(theta_deg), 1.0, xi, eta, sign_r=-1, sign_theta=1)


def result_rows(traj: Trajectory) -> list[Row]:
    """*Results* section rows computed from the trajectory."""
    d = traj.diagnostics
    dphi = azimuthal_winding(traj)
    rows: list[Row] = [
        ("Termination", traj.state.name),
        ("Max null error", f"{d.max_null_error:.3e}"),
        ("Energy drift", f"{d.max_energy_drift:.3e}"),
        ("L_z drift", f"{d.max_lz_drift:.3e}"),
        ("Carter Q drift", f"{d.max_carter_drift:.3e}"),
        ("Closest approach", f"{closest_approach(traj):.9g} M (r+ {outer_horizon(traj.spacetime):.6g} M)"),
        ("Delta phi", f"{dphi:.9g} rad"),
        ("Turns", f"{dphi / (2.0 * math.pi):.6g}"),
        ("Final radius", f"{float(traj.y_end[1]):.6g} M"),
        ("Affine length", f"{float(traj.lam[-1]):.6g} M"),
        ("Steps", f"{traj.n_steps:,} accepted, {traj.n_rejected:,} rejected"),
        ("Runtime", f"{traj.runtime_s:.3f} s"),
    ]
    if traj.state == TerminationState.ESCAPED:
        try:
            rows.append(("Deflection", f"{deflection_angle(traj):.9g} rad (equatorial, finite-radius corrected)"))
        except ValueError:
            pass
    return rows


def _figure(path: Path, traj: Trajectory) -> Path:
    """Top view (x, y) and side projection (x, z) in the oblate embedding of
    :func:`kerrray.geometry.bl_to_cartesian` (x = sqrt(r^2 + a^2) sin(theta) cos(phi), z = r cos(theta)),
    with the horizon and ergosphere drawn in the same embedding."""
    st = traj.spacetime
    x, y, z = to_cartesian(traj)
    a2 = st.a * st.a
    r_plus = outer_horizon(st)
    fig, (ax1, ax2) = plots.plt.subplots(1, 2, figsize=(11.0, 5.5))
    phi = np.linspace(0.0, 2.0 * math.pi, 361)
    r_e = math.sqrt(float(ergosphere_radius(st, 0.5 * math.pi)) ** 2 + a2)
    plots.plot_trajectory_xy(ax1, x, y, math.sqrt(r_plus**2 + a2), (r_e * np.cos(phi), r_e * np.sin(phi)),
                             title=f"Top view, a = {st.spin:g}")
    th = np.linspace(0.0, 2.0 * math.pi, 361)  # angle around the (x, z) plane; BL theta = arccos(cos th)
    r_th = ergosphere_radius(st, np.arccos(np.cos(th)))
    plots.plot_trajectory_xy(ax2, x, z, r_plus, (np.sqrt(r_th**2 + a2) * np.sin(th), r_th * np.cos(th)),
                             title="Side projection (x, z)")
    ax2.plot(math.sqrt(r_plus**2 + a2) * np.sin(th), r_plus * np.cos(th), color=plots.HORIZON_COLOR, linewidth=1.0)
    ax2.set_ylabel("z [M]")
    reach = max(4.0 * closest_approach(traj), 10.0 * st.mass)
    for ax in (ax1, ax2):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
    return plots._save(fig, path, plots.DEFAULT_DPI)


def geodesic(
    config: Path = typer.Option(DEFAULT_KERR_CONFIG, "--config", help="Configuration file."),
    spin: float | None = typer.Option(None, "--spin", help="Dimensionless spin a/M (|a/M| < 1)."),
    theta: float | None = typer.Option(None, "--theta", help="Launch polar angle = observer inclination [deg]."),
    impact_parameter: float | None = typer.Option(None, "--impact-parameter", help="Impact parameter b [M]."),
    prograde: bool = typer.Option(True, "--prograde/--retrograde", help="Sense of L_z relative to the spin."),
    method: str | None = typer.Option(None, "--method", help="rk4 | rk45 | dop853."),
    rtol: float | None = typer.Option(None, "--rtol", help="Relative tolerance (adaptive methods)."),
    atol: float | None = typer.Option(None, "--atol", help="Absolute tolerance (adaptive methods)."),
    step_size: float | None = typer.Option(None, "--step-size", help="Fixed step (rk4) / initial step [M]."),
    launch_radius: float | None = typer.Option(None, "--launch-radius", help="Launch radius r0 [M]."),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Save the trajectory figure."),
    sets: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Integrate one photon geodesic and print its termination state and conservation diagnostics."""
    cfg = load_command_config(config, {"black_hole.spin": spin, "integration.method": method,
                                       "integration.rtol": rtol, "integration.atol": atol,
                                       "integration.step_size": step_size}, sets)
    theta0 = cfg.observer.inclination_deg if theta is None else theta
    b = impact_parameter if impact_parameter is not None else \
        experiment_block(cfg, "frame_dragging", FrameDraggingParams).impact_parameter * cfg.black_hole.mass
    r0 = launch_radius if launch_radius is not None else cfg.observer.radius
    console = get_console()
    try:
        st = kerr(cfg.black_hole.mass, cfg.black_hole.spin)
        integ, term = integrator_options_from_config(cfg), termination_options_from_config(cfg)
        xi, eta = launch_constants(st, theta0, b, prograde)
        y0 = launch_state(st, r0, theta0, b, prograde)
    except ValueError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    render_section(console, "Spacetime", [("Metric", "Schwarzschild" if st.is_schwarzschild else "Kerr"),
                                          ("Mass", f"{st.mass:g} M"), ("Spin", f"{st.spin:.6g}"),
                                          ("Horizon r+", f"{outer_horizon(st):.6g} M"), ("Coordinates", "Boyer-Lindquist")])
    render_section(console, "Photon", [("Launch", f"r0 {r0:g} M, theta0 {theta0:g} deg, inward"),
                                       ("Impact parameter", f"{b:g} M ({'prograde' if prograde else 'retrograde'})"),
                                       ("L_z / E", f"{xi:.9g} M"), ("Q / E^2", f"{eta:.9g} M^2")])
    render_section(console, "Integrator", [("Method", integ.method.upper()),
                                           ("rtol / atol", f"{integ.rtol:g} / {integ.atol:g}"),
                                           ("Step size", f"{integ.step_size:g} M"),
                                           ("Budget", f"lambda_max {integ.lambda_max:g} M, {integ.max_steps:,} steps"),
                                           ("Escape radius", f"{term.escape_radius:g} M")])
    holder: dict[str, Any] = {}

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        traj = integrate(st, y0, integ, term, record=True)
        holder["traj"] = traj
        if plot:
            holder["figure"] = _figure(ctx.report_dir / FIGURE_FILENAME, traj)
        d = traj.diagnostics
        return {"state": traj.state.name, "xi": xi, "eta": eta, "theta_deg": theta0, "impact_parameter": b,
                "prograde": prograde, "launch_radius": r0, "max_null_error": d.max_null_error,
                "max_energy_drift": d.max_energy_drift, "max_lz_drift": d.max_lz_drift,
                "max_carter_drift": d.max_carter_drift, "closest_approach": closest_approach(traj),
                "delta_phi": azimuthal_winding(traj), "n_steps": traj.n_steps, "n_rejected": traj.n_rejected,
                "runtime_s": traj.runtime_s, "y0": y0.tolist(), "y_end": traj.y_end.tolist(),
                "figure": str(holder.get("figure", ""))}

    record = run_experiment("geodesic", cfg, body)
    rows = result_rows(holder["traj"])
    rows += [("Run", record.run_id)] + ([("Figure", str(holder["figure"]))] if "figure" in holder else [])
    render_section(console, "Results", rows)


def register(app: typer.Typer) -> None:
    """Add the ``geodesic`` command to ``app`` (docs/architecture.md section 6)."""
    app.command("geodesic")(geodesic)
