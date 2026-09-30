"""Run manifest tests (PROJECT.md section 35)."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kerrray import __version__
from kerrray.utils.config import load_config
from kerrray.utils.manifest import (
    MANIFEST_KEYS,
    NOT_INSTALLED,
    TRACKED_PACKAGES,
    UNKNOWN_COMMIT,
    build_manifest,
    collect_environment,
    git_commit,
    git_dirty,
    new_run_id,
    package_versions,
    sanitise_for_json,
    utc_timestamp,
    write_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "kerr.yaml"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
SECTION_35_KEYS = (
    "run_id",
    "timestamp",
    "git_commit",
    "config",
    "python_version",
    "package_versions",
    "hardware",
    "solver",
    "tolerances",
    "random_seed",
    "results",
)


def _git(*args: str) -> str:
    """Run git in the repository or skip the test when that is impossible."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    if proc.returncode != 0:
        pytest.skip(f"not a git checkout: {proc.stderr.strip()}")
    return proc.stdout


def _fail_on_json_constant(name: str) -> None:
    raise AssertionError(f"non-standard JSON constant {name!r} in manifest")


def test_new_run_id_format_and_uniqueness() -> None:
    first, second = new_run_id(), new_run_id()
    assert RUN_ID_PATTERN.match(first)
    assert RUN_ID_PATTERN.match(second)
    assert first != second


def test_new_run_id_uses_given_utc_time() -> None:
    now = datetime(2026, 9, 28, 10, 15, 30, tzinfo=timezone.utc)
    assert new_run_id(now).startswith("20260928T101530Z-")


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError):
        new_run_id(datetime(2026, 9, 28, 10, 15, 30))


def test_utc_timestamp_is_iso8601_utc() -> None:
    parsed = datetime.fromisoformat(utc_timestamp())
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def test_package_versions_match_importlib_metadata() -> None:
    versions = package_versions()
    assert set(versions) == set(TRACKED_PACKAGES)
    for name in TRACKED_PACKAGES:
        assert versions[name] == metadata.version(name)
    assert TRACKED_PACKAGES[0] == "kerrray"
    assert versions["kerrray"] == __version__


def test_missing_package_is_reported_not_raised() -> None:
    assert package_versions(["kerrray-no-such-distribution"]) == {
        "kerrray-no-such-distribution": NOT_INSTALLED
    }


def test_git_commit_matches_git_rev_parse() -> None:
    expected = _git("rev-parse", "HEAD").strip()
    assert git_commit() == expected
    assert git_commit(REPO_ROOT) == expected


def test_git_commit_outside_repository(tmp_path: Path) -> None:
    assert git_commit(tmp_path) == UNKNOWN_COMMIT


def test_git_dirty_matches_git_status() -> None:
    expected = bool(_git("status", "--porcelain").strip())
    assert git_dirty() is expected
    assert git_dirty(REPO_ROOT) is expected


def test_git_dirty_outside_repository(tmp_path: Path) -> None:
    assert git_dirty(tmp_path) is None


def test_collect_environment_reports_real_values() -> None:
    env = collect_environment()
    assert env.python_version == platform.python_version()
    assert env.python_implementation == platform.python_implementation()
    assert env.machine == platform.machine()
    assert env.cpu_count == os.cpu_count()
    assert env.package_versions["numpy"] == metadata.version("numpy")
    assert env.package_versions["kerrray"] == __version__
    assert env.git_commit == git_commit()
    assert env.git_dirty == git_dirty()
    assert env.git_dirty is None or isinstance(env.git_dirty, bool)
    assert set(env.hardware()) == {"platform", "system", "machine", "processor", "cpu_count"}


def test_manifest_keys_are_section_35_then_git_dirty() -> None:
    assert MANIFEST_KEYS == (*SECTION_35_KEYS, "git_dirty")


def test_build_manifest_has_every_key_in_order() -> None:
    cfg = load_config(CONFIG_PATH)
    manifest = build_manifest("run-x", cfg, random_seed=0)
    assert list(manifest) == list(MANIFEST_KEYS)
    assert manifest["config"] == cfg.to_mapping()
    assert manifest["solver"]["method"] == cfg.integration.method
    assert manifest["tolerances"]["rtol"] == cfg.integration.rtol
    assert manifest["tolerances"]["atol"] == cfg.integration.atol
    assert manifest["random_seed"] == 0
    assert manifest["results"] == {}
    assert manifest["package_versions"]["kerrray"] == __version__
    assert manifest["git_dirty"] is None or isinstance(manifest["git_dirty"], bool)


def test_write_run_creates_manifest_and_config(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_PATH)
    run_id = new_run_id()
    run_dir = tmp_path / "runs" / run_id
    manifest = build_manifest(run_id, cfg, random_seed=7, results={"note": "phase 0 smoke test"})

    paths = write_run(run_dir, cfg, manifest)

    assert paths.manifest_path == run_dir / "manifest.json"
    assert paths.config_path == run_dir / "config.yaml"
    assert paths.manifest_path.is_file() and paths.config_path.is_file()
    data = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    for key in MANIFEST_KEYS:
        assert key in data, key
    assert data["run_id"] == run_id
    assert data["git_commit"] == git_commit()
    assert data["git_dirty"] is None or isinstance(data["git_dirty"], bool)
    assert data["package_versions"]["kerrray"] == __version__
    assert datetime.fromisoformat(data["timestamp"]).tzinfo is not None
    assert data["python_version"] == platform.python_version()
    assert data["random_seed"] == 7
    assert data["results"] == {"note": "phase 0 smoke test"}
    assert load_config(paths.config_path) == cfg
    assert b"\r\n" not in paths.manifest_path.read_bytes()


def test_write_run_rejects_missing_keys(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_PATH)
    manifest = build_manifest("run-y", cfg, random_seed=0)
    del manifest["results"]
    with pytest.raises(ValueError, match="results"):
        write_run(tmp_path / "run-y", cfg, manifest)


def test_numpy_results_are_json_serialisable(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_PATH)
    results = {"scalar": np.float64(1.5), "array": np.arange(3)}
    manifest = build_manifest("run-z", cfg, random_seed=0, results=results)
    paths = write_run(tmp_path / "run-z", cfg, manifest)
    data = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert data["results"] == {"scalar": 1.5, "array": [0, 1, 2]}


def test_non_finite_results_are_written_as_strict_json(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_PATH)
    results = {
        "x": float("nan"),
        "y": np.inf,
        "z": np.array([1.0, -np.inf, np.nan]),
        "w": np.float32("nan"),
    }
    manifest = build_manifest("run-n", cfg, random_seed=0, results=results)
    paths = write_run(tmp_path / "run-n", cfg, manifest)
    text = paths.manifest_path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text, parse_constant=_fail_on_json_constant)
    assert data["results"] == {"x": "nan", "y": "inf", "z": [1.0, "-inf", "nan"], "w": "nan"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.bool_(True), True),
        (np.int64(3), 3),
        ((1, 2.5), [1, 2.5]),
        ({1: Path("a") / "b"}, {"1": str(Path("a") / "b")}),
        (datetime(2026, 9, 28, tzinfo=timezone.utc), "2026-09-28T00:00:00+00:00"),
        ({"nested": [np.array([[1, 2]]), {"f": -float("inf")}]}, {"nested": [[[1, 2]], {"f": "-inf"}]}),
    ],
)
def test_sanitise_for_json(value: Any, expected: Any) -> None:
    result = sanitise_for_json(value)
    assert result == expected
    assert type(result) is type(expected)
