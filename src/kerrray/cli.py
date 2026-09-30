"""KerrRay command-line interface (PROJECT.md section 28).

Phase 0 provides the application skeleton: a banner callback with the global
``--log-level`` option, and the ``version`` and ``info`` commands. The physics
commands (``geodesic``, ``trace``, ``shadow``, ``lens``, ``validate``,
``benchmark``, ``experiment``, ``report``) are added by the phases that
implement them. Every command prints through the Rich helpers in
:mod:`kerrray.reporting.console`; nothing here calls ``print`` directly, so
output stays safe on non-UTF-8 consoles. Log records go through the Rich
handler installed by :func:`kerrray.utils.logging.configure_logging` on
standard error, so they never mix with results on standard output.
"""

from __future__ import annotations

import typer

from kerrray import __version__
from kerrray.reporting.console import get_console, render_banner, render_section
from kerrray.utils.logging import LOG_LEVEL_NAMES, configure_logging
from kerrray.utils.manifest import collect_environment

HELP_TEXT = (
    "KerrRay: numerical relativistic ray tracing of photon null geodesics "
    "in Schwarzschild and Kerr spacetime."
)

DEFAULT_LOG_LEVEL = "INFO"

app = typer.Typer(
    name="kerrray",
    help=HELP_TEXT,
    add_completion=False,
    # The built-in --help is replaced by the eager option in `main` so that the
    # banner is rendered before the help text. Subcommands keep the built-in
    # help option and inherit both spellings through `help_option_names`.
    add_help_option=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
)


def _print_help(ctx: typer.Context) -> None:
    """Render the help of the command behind ``ctx`` (Typer's Rich help prints itself)."""
    formatter = ctx.make_formatter()
    ctx.command.format_help(ctx, formatter)
    text = formatter.getvalue().rstrip("\n")
    if text:
        typer.echo(text)


def _help_callback(ctx: typer.Context, value: bool) -> None:
    """Eager ``--help`` handler: banner, then help, then exit."""
    if not value or ctx.resilient_parsing:
        return
    render_banner(get_console())
    _print_help(ctx)
    raise typer.Exit()


def _git_dirty_label(dirty: bool | None) -> str:
    if dirty is None:
        return "unknown"
    return "yes (uncommitted changes)" if dirty else "no"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_help: bool = typer.Option(
        False,
        "--help",
        "-h",
        is_eager=True,
        callback=_help_callback,
        help="Show this message and exit.",
    ),
    log_level: str = typer.Option(
        DEFAULT_LOG_LEVEL,
        "--log-level",
        metavar="LEVEL",
        help=f"kerrray logger level: {', '.join(LOG_LEVEL_NAMES)} (case insensitive).",
    ),
) -> None:
    """KerrRay: numerical relativistic ray tracing of photon null geodesics
    in Schwarzschild and Kerr spacetime."""
    try:
        configure_logging(log_level, console=get_console(stderr=True))
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from exc
    render_banner(get_console())
    if ctx.invoked_subcommand is None:
        _print_help(ctx)


@app.command()
def version() -> None:
    """Print the installed KerrRay version."""
    render_section(get_console(), "Version", [("KerrRay", __version__)])


@app.command()
def info() -> None:
    """Show the Python, package, hardware and git environment KerrRay runs in."""
    env = collect_environment()
    console = get_console()
    render_section(
        console,
        "Python",
        [
            ("KerrRay", __version__),
            ("Python", env.python_version),
            ("Implementation", env.python_implementation),
        ],
    )
    render_section(console, "Packages", list(env.package_versions.items()))
    render_section(
        console,
        "Platform",
        [
            ("Platform", env.platform),
            ("Machine", env.machine),
            ("Processor", env.processor),
            ("CPU count", env.cpu_count),
            ("Git commit", env.git_commit),
            ("Git dirty", _git_dirty_label(env.git_dirty)),
        ],
    )


@app.command()
def gui(
    port: int = typer.Option(8741, "--port", help="Local port for KerrRay Desktop (the next free port is used if taken)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser window."),
) -> None:
    """Launch the interactive KerrRay app in your browser (http://127.0.0.1:PORT)."""
    from kerrray.web.server import serve

    render_section(get_console(), "KerrRay Desktop", [("Requested port", port), ("Stop", "Ctrl+C")])
    serve(port, open_browser=not no_browser)


COMMAND_MODULES = (
    "blackhole",
    "geodesic",
    "trace",
    "shadow",
    "lens",
    "render",
    "validate",
    "benchmark",
    "experiment",
    "report",
)
"""Command modules under kerrray.commands, registered in the order of
docs/architecture.md section 6 (``render`` follows ``lens``)."""


def _register_commands() -> None:
    import importlib

    for name in COMMAND_MODULES:
        importlib.import_module(f"kerrray.commands.{name}").register(app)


_register_commands()
