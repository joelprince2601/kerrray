"""``kerrray blackhole create --mass M --spin a`` (PROJECT.md section 28).

Prints the characteristic surfaces and orbits of the requested Kerr hole,
each computed by the tested geometry and orbit code (nothing is typed in):

* horizons ``r_+- = M +- sqrt(M^2 - a^2)`` (:func:`kerrray.geometry.horizon_radii`);
* ergosphere ``r_E(theta) = M + sqrt(M^2 - a^2 cos^2 theta)`` at the equator
  and the pole (:func:`kerrray.geometry.ergosphere_radius`);
* prograde and retrograde equatorial circular photon orbits and critical
  impact parameters (:mod:`kerrray.photons.orbits`; Bardeen, Press and
  Teukolsky 1972, eq. 2.18; docs/derivations.md sections 5-6);
* prograde and retrograde ISCO (BPT 1972, eq. 2.21).

The mass and spin default to the ``black_hole`` block of ``--config``
(``configs/kerr.yaml``); ``--mass`` and ``--spin`` override it.
"""

from __future__ import annotations

import math
from pathlib import Path

import typer

from kerrray.geometry import ergosphere_radius, horizon_radii, kerr
from kerrray.photons.orbits import critical_impact_parameters, equatorial_photon_orbit_radius, isco_radius
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.commands.validate import DEFAULT_KERR_CONFIG, SET_HELP, load_command_config

__all__ = ["blackhole_rows", "register"]

blackhole_app = typer.Typer(help="Inspect a Kerr black hole.", no_args_is_help=True)


def blackhole_rows(mass: float, spin: float) -> list[tuple[str, list[Row]]]:
    """Titled sections describing the hole of the given mass and dimensionless spin."""
    st = kerr(mass, spin)
    r_plus, r_minus = horizon_radii(st)
    b_pro, b_ret = critical_impact_parameters(st)
    return [
        ("Spacetime", [("Metric", "Schwarzschild" if st.is_schwarzschild else "Kerr"), ("Mass", f"{mass:g} M"),
                       ("Spin a/M", f"{spin:.6g}"), ("a", f"{st.a:.6g} M"), ("Coordinates", "Boyer-Lindquist")]),
        ("Horizons", [("Outer r+", f"{r_plus:.9f} M"), ("Inner r-", f"{r_minus:.9f} M")]),
        ("Ergosphere", [("r_E equator", f"{float(ergosphere_radius(st, 0.5 * math.pi)):.9f} M"),
                        ("r_E pole", f"{float(ergosphere_radius(st, 0.0)):.9f} M"),
                        ("Equatorial depth", f"{float(ergosphere_radius(st, 0.5 * math.pi)) - r_plus:.9f} M")]),
        ("Photon orbits", [("Prograde r_ph", f"{equatorial_photon_orbit_radius(st, True):.9f} M"),
                           ("Retrograde r_ph", f"{equatorial_photon_orbit_radius(st, False):.9f} M"),
                           ("Prograde b_c", f"{b_pro:.9f} M"), ("Retrograde b_c", f"{b_ret:.9f} M")]),
        ("ISCO", [("Prograde", f"{isco_radius(st, True):.9f} M"), ("Retrograde", f"{isco_radius(st, False):.9f} M")]),
    ]


@blackhole_app.command("create")
def create(
    mass: float | None = typer.Option(None, "--mass", help="Mass M (geometric units; default from the config)."),
    spin: float | None = typer.Option(None, "--spin", help="Dimensionless spin a/M, |a/M| < 1."),
    config: Path = typer.Option(DEFAULT_KERR_CONFIG, "--config", help="Configuration file."),
    sets: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Horizons, ergosphere, photon orbits, ISCO and critical impact parameters of a Kerr hole."""
    cfg = load_command_config(config, {"black_hole.mass": mass, "black_hole.spin": spin}, sets)
    console = get_console()
    try:
        sections = blackhole_rows(cfg.black_hole.mass, cfg.black_hole.spin)
    except ValueError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    for title, rows in sections:
        render_section(console, title, rows)


def register(app: typer.Typer) -> None:
    """Add the ``blackhole`` sub-app to ``app`` (docs/architecture.md section 6)."""
    app.add_typer(blackhole_app, name="blackhole")
