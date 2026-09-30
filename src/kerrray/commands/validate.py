"""``kerrray validate schwarzschild`` and ``kerrray validate kerr`` (PROJECT.md sections 12, 13 and 28).

``schwarzschild`` runs :func:`kerrray.validation.schwarzschild.run_schwarzschild_validation`
(EXP-001, EXP-002 and the convergence of ``b_c`` with ``rtol`` and the RK4
step) and prints the section 12 lines: the analytical reference, the
numerical value and the computed error for the photon sphere and the critical
impact parameter, then the convergence tables and the verdicts. ``kerr`` runs
:func:`kerrray.validation.kerr.run_kerr_validation`. Both exit with code 1 when
any check fails (or a search cannot classify its rays) and 0 otherwise; every
number printed is computed by the run. The helpers :func:`parse_sets` and
:func:`load_command_config` are shared with ``blackhole`` and ``geodesic``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import typer
import yaml
from rich.console import Console

from kerrray.reporting.console import Row, get_console, render_section
from kerrray.utils.config import ConfigError, KerrRayConfig, load_config
from kerrray.validation import ValidationReport

__all__ = ["DEFAULT_KERR_CONFIG", "DEFAULT_SCHWARZSCHILD_CONFIG", "load_command_config", "parse_sets", "register",
           "kerr_rows", "schwarzschild_rows"]

DEFAULT_SCHWARZSCHILD_CONFIG: Final[Path] = Path("configs") / "schwarzschild.yaml"
DEFAULT_KERR_CONFIG: Final[Path] = Path("configs") / "kerr.yaml"
SET_HELP: Final[str] = "Dotted override key.path=value, value parsed as YAML (repeatable), e.g. --set integration.rtol=1e-8."

validate_app = typer.Typer(help="Validate the physics against Schwarzschild and Kerr references.", no_args_is_help=True)


def parse_sets(sets: list[str] | None) -> dict[str, Any]:
    """``["a.b=1e-8", "c=[1, 2]"]`` -> ``{"a.b": 1e-08, "c": [1, 2]}`` (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for item in sets or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects key.path=value, got {item!r}")
        try:
            parsed = yaml.safe_load(value.strip())
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set {key.strip()}: value is not valid YAML: {exc}") from None
        if isinstance(parsed, str):  # YAML 1.1 reads "1e-8" (no dot) as a string
            try:
                parsed = float(parsed) if parsed.strip().lower() not in ("nan", "inf", "-inf", "+inf") else parsed
            except ValueError:
                pass
        out[key.strip()] = parsed
    return out


def load_command_config(config: Path, named: Mapping[str, Any], sets: list[str] | None) -> KerrRayConfig:
    """Load ``config`` with the named flags (``None`` = not given) and ``--set`` overrides; exit 1 on errors."""
    try:
        overrides = {k: v for k, v in named.items() if v is not None}
        overrides.update(parse_sets(sets))
        return load_config(config, overrides)
    except (ConfigError, OSError) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None


def _f(value: float, digits: int = 9) -> str:
    return f"{value:.{digits}f}"


def _e(value: Any) -> str:
    return f"{value:.3e}" if isinstance(value, (int, float)) else str(value)


def _verdict_rows(report: ValidationReport) -> list[Row]:
    rows: list[Row] = []
    for c in report.checks:
        tol = ", ".join(f"{k} {v:g}" if isinstance(v, (int, float)) else f"{k} {v}" for k, v in c.tolerance.items())
        rows.append((c.name, f"{'PASS' if c.passed else 'FAIL'}  ({tol})"))
    rows.append(("All checks", "PASS" if report.all_passed else "FAIL"))
    if report.run is not None:
        rows += [("Run", report.run.run_id), ("Report dir", str(report.run.report_dir)),
                 ("Runtime", f"{report.run.runtime_s:.1f} s")]
    return rows


def _study_rows(check: Any) -> list[Row]:
    c = check.computed
    rows: list[Row] = [(f"{c['parameter']} {lv:g}", f"b_c {_f(v, 12)} M  error {_e(e)}  ({d['runtime_s']:.1f} s)")
                       for lv, v, e, d in zip(c["levels"], c["values"], c["errors"], c["levels_detail"], strict=True)]
    rows.append(("Fitted order", f"{c['fitted_order']:.4f} (errors above {c['noise_floor']:.1e} M)"))
    rich = c["richardson"]
    rows.append(("Richardson order", f"{rich['order']:.4f}" if rich else "n/a (levels not in a constant ratio)"))
    crossings = sum(d.get("horizon_crossings", 0) for d in c["levels_detail"])
    if crossings:
        rows.append(("Horizon crossings", f"{crossings} rays relabelled CAPTURED (fixed step, see kerr_bisection)"))
    return rows


def schwarzschild_rows(report: ValidationReport) -> list[tuple[str, list[Row]]]:
    """Titled sections for the Schwarzschild report (section 12 style)."""
    sections: list[tuple[str, list[Row]]] = []
    for name, label in (("photon_sphere", "photon sphere"), ("critical_impact", "critical impact b_c")):
        c = report.check(name)
        comp = c.computed
        sections.append((label[0].upper() + label[1:], [
            (f"Analytical {label}", f"{_f(c.reference['value'], 6)} M"),
            (f"Numerical {label}", f"{_f(c.computed['value'], 12)} M"),
            ("Absolute error", f"{_e(c.error['absolute'])} M"),
            ("Relative error", _e(c.error["relative"])),
            ("Bracket width", f"{_e(comp['width'])} M after {comp['iterations']} iterations"),
            ("Rays / max steps", f"{comp['n_rays']} / {comp['max_steps']}"),
            ("Runtime", f"{comp['runtime_s']:.2f} s"),
        ]))
    for name, title in (("rtol_convergence", "Convergence of b_c with rtol (rk45)"),
                        ("rk4_convergence", "Convergence of b_c with the RK4 step")):
        try:
            sections.append((title, _study_rows(report.check(name))))
        except KeyError:
            continue
    sections.append(("Verdict", _verdict_rows(report)))
    return sections


def kerr_rows(report: ValidationReport) -> list[tuple[str, list[Row]]]:
    """Titled sections for the Kerr report."""
    hz = report.check("horizon")
    lim = report.check("schwarzschild_limit")
    crit = report.check("critical_impact")
    cons = report.check("conservation")
    near = report.check("near_extremal")
    rev = report.check("reversibility")
    lc = lim.computed
    sections: list[tuple[str, list[Row]]] = [
        ("Horizon and ergosphere", [
            ("Spins checked", ", ".join(f"{r['spin']:g}" for r in hz.computed["rows"])),
            ("Worst error", _e(hz.error["max"])),
        ] + [(f"r+ / r- (a = {r['spin']:g})", f"{_f(r['r_plus'], 12)} / {_f(r['r_minus'], 12)} M")
             for r in hz.computed["rows"] if r["spin"] == report.spin]),
        ("Schwarzschild limit (a -> 0)", [
            (f"a = {a:g}", f"b_pro {_f(bp)}  b_ret {_f(br)} M  (b - 3 sqrt3 M)/a {dp:+.5f} / {dr:+.5f}")
            for a, bp, br, dp, dr in zip(lc["spins"], lc["b_prograde"], lc["b_retrograde"],
                                         lc["deviation_over_spin_prograde"], lc["deviation_over_spin_retrograde"],
                                         strict=True)
        ] + [
            ("Reference 3 sqrt3 M", f"{_f(lim.reference['b_c_schwarzschild'])} M"),
            ("Exponent pro / ret", f"{lc['exponents']['prograde']:.5f} / {lc['exponents']['retrograde']:.5f}"),
            ("Even part (min a)", f"{_e(lim.error['even_part_at_smallest_spin'])} M"),
            ("Max rel. error", _e(lim.error["max_relative_error_vs_derived"])),
        ]),
        ("Critical impact parameters", [
            (f"a = {r['spin']:g}", f"pro {_f(r['b_prograde'])} (err {_e(r['rel_err_prograde'])})  "
                                   f"retro {_f(r['b_retrograde'])} (err {_e(r['rel_err_retrograde'])})")
            for r in crit.computed["rows"]
        ]),
        ("Conservation (off-equatorial rays)", [
            (f"a = {r['spin']:g}", f"{r['state']}  E {_e(r['max_drift']['energy'])}  Lz {_e(r['max_drift']['lz'])}  "
                                   f"Q {_e(r['max_drift']['carter'])}  null {_e(r['max_drift']['null'])}")
            for r in cons.computed["rows"]
        ]),
        ("Near-extremal prograde rays", [
            (f"a = {r['spin']:g}, eps {run['horizon_epsilon']:g}",
             f"{run['state']}  r_min {run['closest_approach']:.6f} (r+ {r['r_plus']:.6f})  turns {run['turns']:.3f}  "
             f"steps {run['n_steps']}  Q {_e(run['max_drift']['carter'])}")
            for r in near.computed["rows"] for run in r["runs"]
        ]),
        ("Forward/backward consistency", [(k, _e(v)) for k, v in rev.error.items()]
         + [("Legs", f"{rev.computed['forward_state']} / {rev.computed['backward_state']}")]),
        ("Verdict", _verdict_rows(report)),
    ]
    return sections


def _render(console: Console, sections: list[tuple[str, list[Row]]]) -> None:
    for title, rows in sections:
        render_section(console, title, rows)


def _run(kind: str, cfg: KerrRayConfig) -> None:
    console = get_console()
    try:
        if kind == "schwarzschild":
            from kerrray.validation.schwarzschild import run_schwarzschild_validation

            report = run_schwarzschild_validation(cfg)
            _render(console, schwarzschild_rows(report))
        else:
            from kerrray.validation.kerr import run_kerr_validation

            report = run_kerr_validation(cfg)
            _render(console, kerr_rows(report))
    except (RuntimeError, ValueError, ConfigError) as exc:
        get_console(stderr=True).print(f"error: validation could not complete: {exc}")
        raise typer.Exit(code=1) from None
    if not report.all_passed:
        raise typer.Exit(code=1)


@validate_app.command("schwarzschild")
def schwarzschild(
    config: Path = typer.Option(DEFAULT_SCHWARZSCHILD_CONFIG, "--config", help="Configuration file."),
    rtol: float | None = typer.Option(None, "--rtol", help="Relative tolerance of the main searches (rk45)."),
    sets: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """EXP-001/EXP-002: photon sphere and critical impact parameter against 3M and 3 sqrt(3) M."""
    _run("schwarzschild", load_command_config(config, {"integration.rtol": rtol}, sets))


@validate_app.command("kerr")
def kerr_command(
    config: Path = typer.Option(DEFAULT_KERR_CONFIG, "--config", help="Configuration file."),
    spin: float | None = typer.Option(None, "--spin", help="Spin of the horizon check (a/M, |a/M| < 1)."),
    rtol: float | None = typer.Option(None, "--rtol", help="Relative tolerance (rk45)."),
    sets: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Kerr checks: horizon, a -> 0 limit, prograde/retrograde b_c, conservation, near-extremal, reversibility."""
    _run("kerr", load_command_config(config, {"black_hole.spin": spin, "integration.rtol": rtol}, sets))


def register(app: typer.Typer) -> None:
    """Add the ``validate`` sub-app to ``app`` (docs/architecture.md section 6)."""
    app.add_typer(validate_app, name="validate")
