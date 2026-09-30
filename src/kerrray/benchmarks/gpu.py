"""GPU benchmark stub (PROJECT.md sections 25 to 26 and EXP-010; docs/decisions.md D-007).

GPU acceleration is optional future work: no GPU backend exists in KerrRay
and no CUDA device is used. :func:`run_gpu_benchmark` only checks, inside a
``try``/``except``, whether ``numba.cuda`` reports a usable device, prints
and records the outcome, and returns a :class:`~kerrray.experiments.base.RunRecord`
whose results contain ``gpu_available`` and the message. It never produces a
timing, a speed-up or a memory number: there is nothing to measure without a
GPU path, and PROJECT.md sections 26 and 47 forbid reporting unmeasured
results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from rich.console import Console

from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.reporting.console import get_console
from kerrray.reporting.report import ReportSections, write_report
from kerrray.utils.config import KerrRayConfig

__all__ = ["EXPERIMENT_NAME", "GPU_UNAVAILABLE_MESSAGE", "GpuStatus", "gpu_status", "run_gpu_benchmark"]

EXPERIMENT_NAME: Final[str] = "benchmark_gpu"

GPU_UNAVAILABLE_MESSAGE: Final[str] = (
    "GPU acceleration is optional future work (docs/decisions.md D-007); "
    "no CUDA device available; no GPU numbers are reported"
)
GPU_PRESENT_MESSAGE: Final[str] = (
    "GPU acceleration is optional future work (docs/decisions.md D-007); a CUDA device is "
    "reported by numba.cuda but KerrRay has no GPU backend; no GPU numbers are reported"
)


@dataclass(frozen=True)
class GpuStatus:
    """Outcome of the CUDA availability probe."""

    available: bool
    detail: str


def gpu_status() -> GpuStatus:
    """Probe ``numba.cuda.is_available()`` without ever raising.

    Returns ``available=False`` with the reason when Numba or its CUDA
    support cannot be imported or the probe itself fails.
    """
    try:
        from numba import cuda  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError, or numba present without CUDA support
        return GpuStatus(False, f"numba.cuda not importable: {type(exc).__name__}: {exc}")
    try:
        available = bool(cuda.is_available())
    except Exception as exc:
        return GpuStatus(False, f"numba.cuda.is_available() failed: {type(exc).__name__}: {exc}")
    return GpuStatus(available, "numba.cuda.is_available() returned " + str(available))


def _body(ctx: ExperimentContext, console: Console) -> dict[str, Any]:
    status = gpu_status()
    message = GPU_PRESENT_MESSAGE if status.available else GPU_UNAVAILABLE_MESSAGE
    console.print(message)
    ctx.logger.info("%s (%s)", message, status.detail)
    sections = ReportSections(
        objective="EXP-010, GPU performance: report whether a GPU path exists and can be measured.",
        mathematical_model="Not applicable: no computation is performed by this stub.",
        numerical_method="A guarded probe of numba.cuda.is_available(); no kernel is launched.",
        parameters=f"CUDA probe detail: {status.detail}",
        results=message + "\n\nNo runtime, speed-up or memory figure is reported because none was measured.",
        error_analysis="Not applicable.",
        interpretation=(
            "Phase 8 is CPU performance and precision (docs/architecture.md section 9); the CPU "
            "benchmark (EXP-009) and the precision study (EXP-011) carry the measured numbers."
        ),
        limitations="A future GPU backend must satisfy the same agreement test as the numba backend before any speed-up is reported.",
        reproducibility=f"Run {ctx.run_id}; configuration in {ctx.run_dir}.",
    )
    write_report(ctx.report_dir, sections, title="KerrRay EXP-010: GPU benchmark (not available)")
    return {
        "gpu_available": status.available,
        "gpu_numbers_reported": False,
        "message": message,
        "cuda_probe": status.detail,
    }


def run_gpu_benchmark(cfg: KerrRayConfig, *, console: Console | None = None) -> RunRecord:
    """Record that GPU acceleration is optional future work (never a fabricated number).

    Args:
        cfg: Any configuration; only the ``experiment`` block (directories,
            seed) is used.
        console: Console to print the message to (standard output by default).
    """
    out = console if console is not None else get_console()
    return run_experiment(EXPERIMENT_NAME, cfg, lambda ctx: _body(ctx, out))
