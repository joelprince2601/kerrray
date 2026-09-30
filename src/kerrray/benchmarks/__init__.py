"""Solver, CPU, GPU and floating-point precision benchmarks (PROJECT.md sections 19, 26 and 27).

Phase 6: :mod:`kerrray.benchmarks.solver` (solver comparison, EXP-006) with
its configuration in :mod:`kerrray.benchmarks.solver_config`, its ray set and
reference machinery in :mod:`kerrray.benchmarks.rayset` and its report in
:mod:`kerrray.benchmarks.solver_report`. Phase 8 (CPU performance and
precision, docs/architecture.md section 9): :mod:`~kerrray.benchmarks.cpu`
(EXP-009), :mod:`~kerrray.benchmarks.precision` (EXP-011),
:mod:`~kerrray.benchmarks.memory` and the :mod:`~kerrray.benchmarks.gpu`
stub (GPU acceleration is optional future work, docs/decisions.md D-007).
Nothing is imported eagerly here so that ``kerrray benchmark`` stays cheap
to start. Measured results are discussed in docs/numerical_analysis.md.
"""
