"""Tests for the experiment driver scaffolding (PROJECT.md sections 35 and 36)."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kerrray.experiments.base import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    SUMMARY_FILENAME,
    ExperimentContext,
    RunRecord,
    Timer,
    run_experiment,
)
from kerrray.utils.config import KerrRayConfig, load_config
from kerrray.utils.manifest import (
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_KEYS,
    EnvironmentInfo,
    collect_environment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "schwarzschild.yaml"


@pytest.fixture(scope="module")
def env() -> EnvironmentInfo:
    return collect_environment()


@pytest.fixture
def cfg(tmp_path: Path) -> KerrRayConfig:
    return load_config(
        CONFIG_PATH,
        {
            "experiment.output_dir": str(tmp_path / "runs"),
            "experiment.report_dir": str(tmp_path / "reports"),
            "experiment.seed": 3,
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_experiment_writes_manifest_config_and_summary(
    cfg: KerrRayConfig, env: EnvironmentInfo
) -> None:
    def body(ctx: ExperimentContext) -> dict[str, Any]:
        assert ctx.run_dir.is_dir() and ctx.report_dir.is_dir()
        assert ctx.cfg is cfg
        return {"value": np.float64(1.5), "array": np.arange(3), "nan": float("nan")}

    record = run_experiment("demo", cfg, body, environment=env)

    assert isinstance(record, RunRecord)
    assert record.run_dir == Path(cfg.experiment.output_dir) / record.run_id
    assert record.report_dir == Path(cfg.experiment.report_dir) / record.run_id
    assert record.manifest_path == record.run_dir / MANIFEST_FILENAME
    assert record.summary_path == record.report_dir / SUMMARY_FILENAME
    assert record.results == {"value": 1.5, "array": [0, 1, 2], "nan": "nan"}
    assert record.runtime_s >= 0.0

    manifest = _read_json(record.manifest_path)
    for key in MANIFEST_KEYS:
        assert key in manifest
    assert manifest["run_id"] == record.run_id
    assert manifest["experiment"] == "demo"
    assert manifest["status"] == STATUS_COMPLETED
    assert manifest["results"] == record.results
    assert manifest["random_seed"] == 3
    assert manifest["error"] is None
    assert manifest["runtime_s"] == pytest.approx(record.runtime_s)
    assert manifest["finished"] is not None
    assert manifest["git_commit"] == env.git_commit
    assert load_config(record.run_dir / CONFIG_FILENAME) == cfg

    summary = _read_json(record.summary_path)
    assert summary["run_id"] == record.run_id
    assert summary["experiment"] == "demo"
    assert summary["status"] == STATUS_COMPLETED
    assert summary["results"] == record.results
    assert summary["random_seed"] == 3
    assert Path(summary["manifest_path"]) == record.manifest_path


def test_manifest_exists_with_running_status_before_body_runs(
    cfg: KerrRayConfig, env: EnvironmentInfo
) -> None:
    seen: dict[str, Any] = {}

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        manifest = _read_json(ctx.run_dir / MANIFEST_FILENAME)
        seen["status"] = manifest["status"]
        seen["results"] = manifest["results"]
        seen["config_written"] = (ctx.run_dir / CONFIG_FILENAME).is_file()
        return {}

    run_experiment("demo", cfg, body, environment=env)
    assert seen == {"status": STATUS_RUNNING, "results": {}, "config_written": True}


def test_failure_marks_manifest_failed_and_reraises(
    cfg: KerrRayConfig, env: EnvironmentInfo
) -> None:
    def body(ctx: ExperimentContext) -> dict[str, Any]:
        raise RuntimeError("boom at step 3")

    with pytest.raises(RuntimeError, match="boom at step 3"):
        run_experiment("demo", cfg, body, run_id="fixed-id", environment=env)

    manifest = _read_json(Path(cfg.experiment.output_dir) / "fixed-id" / MANIFEST_FILENAME)
    assert manifest["status"] == STATUS_FAILED
    assert manifest["error"] == "RuntimeError: boom at step 3"
    assert manifest["results"] == {}
    assert manifest["runtime_s"] >= 0.0
    summary = _read_json(Path(cfg.experiment.report_dir) / "fixed-id" / SUMMARY_FILENAME)
    assert summary["status"] == STATUS_FAILED
    assert "boom" in summary["error"]


def test_non_mapping_result_is_a_failure(cfg: KerrRayConfig, env: EnvironmentInfo) -> None:
    with pytest.raises(TypeError, match="mapping"):
        run_experiment("demo", cfg, lambda ctx: [1, 2], run_id="bad-result", environment=env)
    manifest = _read_json(Path(cfg.experiment.output_dir) / "bad-result" / MANIFEST_FILENAME)
    assert manifest["status"] == STATUS_FAILED
    assert manifest["error"].startswith("TypeError")


def test_rng_is_seeded_from_the_configuration(cfg: KerrRayConfig, env: EnvironmentInfo) -> None:
    def body(ctx: ExperimentContext) -> dict[str, Any]:
        return {"draws": ctx.rng.random(4)}

    first = run_experiment("demo", cfg, body, environment=env).results["draws"]
    second = run_experiment("demo", cfg, body, environment=env).results["draws"]
    assert first == second
    other = cfg.with_overrides({"experiment.seed": 4})
    third = run_experiment("demo", other, body, environment=env).results["draws"]
    assert third != first
    assert first == list(np.random.default_rng(3).random(4))


def test_context_exposes_logger_timer_and_parameters(
    cfg: KerrRayConfig, env: EnvironmentInfo, caplog: pytest.LogCaptureFixture
) -> None:
    @dataclasses.dataclass(frozen=True)
    class Validation:
        photon_sphere_tol: float = 1.0
        critical_b_tol: float = 1.0
        bisection_iterations: int = 1

    raw = cfg.experiment.parameters["validation"]

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        params = ctx.parameters("validation", Validation)
        assert params.bisection_iterations == raw["bisection_iterations"]
        assert params.photon_sphere_tol == float(raw["photon_sphere_tol"])
        ctx.logger.info("inside body")
        assert ctx.logger.name == "kerrray.experiments.demo"
        return {"elapsed": ctx.timer.elapsed_s(), "lap": ctx.timer.lap("body")}

    with caplog.at_level(logging.INFO, logger="kerrray"):
        record = run_experiment("demo", cfg, body, environment=env)
    assert record.results["elapsed"] >= 0.0
    assert record.results["lap"] >= record.results["elapsed"]
    assert any("inside body" in r.getMessage() for r in caplog.records)
    assert any(record.run_id in r.getMessage() for r in caplog.records)


def test_run_ids_are_unique_and_directories_are_created(
    cfg: KerrRayConfig, env: EnvironmentInfo
) -> None:
    a = run_experiment("demo", cfg, lambda ctx: {}, environment=env)
    b = run_experiment("demo", cfg, lambda ctx: {}, environment=env)
    assert a.run_id != b.run_id
    assert a.run_dir.is_dir() and b.run_dir.is_dir()
    assert a.report_dir.is_dir() and b.report_dir.is_dir()


def test_timer_is_monotone() -> None:
    timer = Timer()
    first = timer.elapsed_s()
    time.sleep(0.01)
    lap = timer.lap("wait")
    second = timer.elapsed_s()
    assert 0.0 <= first <= second
    assert lap > 0.0 and timer.laps == {"wait": lap}
    assert timer.lap("again") < lap + 0.5
