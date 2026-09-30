"""CLI command modules, one per verb (docs/architecture.md section 6).

Every module in this package exposes::

    def register(app: typer.Typer) -> None: ...

which adds its command (``app.command(...)``) or sub-app
(``app.add_typer(...)``) to the Typer application owned by
:mod:`kerrray.cli`. ``cli.py`` imports the modules and calls each ``register``
in this fixed order so ``kerrray --help`` lists the verbs of PROJECT.md
section 28 predictably: ``blackhole``, ``geodesic``, ``trace``, ``shadow``,
``lens``, ``validate``, ``benchmark``, ``experiment``, ``report``.

Conventions shared by every command:

* load a configuration with ``--config`` (defaulting to the matching file in
  ``configs/``) and apply command-line overrides through
  :func:`kerrray.utils.config.apply_overrides`;
* print through :mod:`kerrray.reporting.console` (banner already rendered by
  the application callback, then titled sections) and print only computed
  values;
* exit with a non-zero code and a one-line message on user errors instead of
  a traceback.

Owners: ``blackhole``, ``geodesic`` (kerr role); ``trace``, ``shadow``
(raytracing role); ``lens`` (rendering role); ``validate`` (schwarzschild
role); ``benchmark``, ``experiment`` (numerical role); ``report`` (infra role,
:mod:`kerrray.commands.report`).
"""
