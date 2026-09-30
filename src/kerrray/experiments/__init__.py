"""Experiment drivers (PROJECT.md sections 18 to 22 and 37).

Every experiment module in this package (``shadow``, ``near_critical``,
``lensing``, ``spin_sweep``, ``frame_dragging``, ``step_size``, ...)
exposes the same entry point::

    def run(cfg: KerrRayConfig) -> RunRecord: ...

implemented with :func:`kerrray.experiments.base.run_experiment`::

    def run(cfg):
        return run_experiment("shadow", cfg, _body)

    def _body(ctx: ExperimentContext) -> dict:
        params = ctx.parameters("spin_sweep", SpinSweepParams)   # D-009 sub-block
        ...                                                       # compute
        plot_shadow(ctx.report_dir / "shadow.png", ...)           # figures next to the report
        return {"boundary_rms": rms, ...}                         # JSON-serialisable results

``run_experiment`` creates ``<output_dir>/<run_id>/`` with ``manifest.json``
and ``config.yaml`` before the body runs, seeds the generator ``ctx.rng`` from
``experiment.seed``, records results, runtime and status afterwards and
writes ``<report_dir>/<run_id>/summary.json``. ``kerrray experiment
--config configs/<name>.yaml`` dispatches on ``experiment.name`` to
``kerrray.experiments.<name>.run``; ``kerrray report --run runs/<run_id>``
prints the stored manifest. Experiment parameters come from
``experiment.parameters`` in the configuration file, never from constants in
the driver (PROJECT.md section 34).
"""
