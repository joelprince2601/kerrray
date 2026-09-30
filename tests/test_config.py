"""Configuration schema tests: file loading, validation, round trip, overrides, sub-blocks."""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from kerrray.utils.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    INCLINATION_EPSILON_DEG,
    SUPPORTED_BACKENDS,
    SUPPORTED_DTYPES,
    SUPPORTED_INTEGRATION_METHODS,
    ConfigError,
    KerrRayConfig,
    apply_overrides,
    dump_config,
    experiment_block,
    load_config,
    load_yaml_mapping,
)
from kerrray.utils.config_parsing import parse_block, validate_block_type
from kerrray.utils.seeds import DEFAULT_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_IDS = [path.stem for path in CONFIG_FILES]
EXPECTED_STEMS = {
    "schwarzschild", "kerr", "shadow", "lensing", "convergence", "near_critical", "gpu",
    "benchmark",
}
DEVELOPMENT_RESOLUTION = 64  # docs/architecture.md section 8

SECTION_34_EXAMPLE = """\
black_hole:
  mass: 1.0
  spin: 0.9

observer:
  radius: 1000.0
  inclination_deg: 60.0

integration:
  method: rk45
  rtol: 1e-9
  atol: 1e-11
  max_steps: 100000

raytrace:
  resolution: 512

termination:
  horizon_epsilon: 1e-6
  escape_radius: 1000.0
"""
"""The PROJECT.md section 34 example, copied verbatim (checked against the file below)."""


# --- Experiment parameter sub-blocks of docs/architecture.md section 7 -------------
# These dataclasses mirror the section 7 defaults so the YAML files are checked
# against the contract the experiment drivers implement.


@dataclasses.dataclass(frozen=True)
class _Lensing:
    impact_min: float = 3.0
    impact_max: float = 20.0
    n_rays: int = 40
    launch_radius: float = 1000.0


@dataclasses.dataclass(frozen=True)
class _Convergence:
    resolutions: list[int] = dataclasses.field(default_factory=lambda: [64, 128, 256, 512])
    tolerances: list[float] = dataclasses.field(
        default_factory=lambda: [1e-6, 1e-8, 1e-10, 1e-12]
    )
    step_sizes: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.01, 0.001, 0.0001])


@dataclasses.dataclass(frozen=True)
class _NearCritical:
    offsets: list[float] = dataclasses.field(
        default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    )
    launch_radius: float = 1000.0


@dataclasses.dataclass(frozen=True)
class _FrameDragging:
    spins: list[float] = dataclasses.field(default_factory=lambda: [-0.9, 0.0, 0.9])
    impact_parameter: float = 6.0
    launch_radius: float = 1000.0


@dataclasses.dataclass(frozen=True)
class _SpinSweep:
    spins: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
    )


@dataclasses.dataclass(frozen=True)
class _InclinationSweep:
    inclinations_deg: list[float] = dataclasses.field(
        default_factory=lambda: [0, 30, 45, 60, 75, 90]
    )


@dataclasses.dataclass(frozen=True)
class _Disk:
    enabled: bool = False
    r_in: float | None = None
    r_out: float = 20.0
    emissivity_index: float = 3.0
    intensity_law: str = "g4"
    prograde: bool = True


@dataclasses.dataclass(frozen=True)
class _Benchmark:
    ray_counts: list[int] = dataclasses.field(
        default_factory=lambda: [1000, 10000, 100000, 1000000]
    )
    backends: list[str] = dataclasses.field(default_factory=lambda: ["numpy", "numba"])
    dtypes: list[str] = dataclasses.field(default_factory=lambda: ["float32", "float64"])
    repeats: int = 3


@dataclasses.dataclass(frozen=True)
class _SchwarzschildValidation:
    photon_sphere_tol: float = 1e-6
    critical_b_tol: float = 1e-6
    bisection_iterations: int = 60


@dataclasses.dataclass(frozen=True)
class _KerrValidation:
    horizon_tol: float = 1e-12
    small_spins: list[float] = dataclasses.field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    schwarzschild_limit_tol: float = 1e-6
    conservation_tol: float = 1e-8
    critical_b_tol: float = 1e-6
    bisection_iterations: int = 60
    near_extremal_spins: list[float] = dataclasses.field(default_factory=lambda: [0.99, 0.999])


EXPECTED_SUB_BLOCKS: dict[str, dict[str, type]] = {
    "shadow": {
        "spin_sweep": _SpinSweep,
        "inclination_sweep": _InclinationSweep,
        "convergence": _Convergence,
    },
    "lensing": {"lensing": _Lensing},
    "convergence": {"convergence": _Convergence},
    "near_critical": {"near_critical": _NearCritical},
    "kerr": {
        "frame_dragging": _FrameDragging,
        "spin_sweep": _SpinSweep,
        "validation": _KerrValidation,
    },
    "schwarzschild": {"validation": _SchwarzschildValidation},
    "gpu": {"benchmark": _Benchmark},
    "benchmark": {"benchmark": _Benchmark},
}


@pytest.fixture
def base() -> dict[str, Any]:
    """Raw mapping of configs/kerr.yaml, mutated by the validation tests."""
    return load_yaml_mapping(CONFIG_DIR / "kerr.yaml")


def test_section_30_config_files_exist() -> None:
    assert {path.stem for path in CONFIG_FILES} == EXPECTED_STEMS


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_config_file_loads_and_validates(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.experiment.name == path.stem
    assert cfg.integration.method in SUPPORTED_INTEGRATION_METHODS
    assert isinstance(cfg.integration.rtol, float) and cfg.integration.rtol > 0
    assert isinstance(cfg.integration.atol, float) and cfg.integration.atol > 0
    assert isinstance(cfg.integration.max_steps, int)
    assert cfg.integration.step_size > 0 and cfg.integration.lambda_max > 0
    assert isinstance(cfg.raytrace.resolution, int) and cfg.raytrace.resolution >= 1
    assert cfg.raytrace.fov > 0
    assert cfg.raytrace.backend in SUPPORTED_BACKENDS
    assert cfg.raytrace.dtype in SUPPORTED_DTYPES
    assert cfg.black_hole.mass > 0 and abs(cfg.black_hole.spin) < 1
    assert 0 <= cfg.observer.inclination_deg <= 180
    assert cfg.experiment.seed >= 0
    assert cfg.experiment.output_dir and cfg.experiment.report_dir


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_config_files_state_every_key_explicitly(path: Path) -> None:
    """The architecture section 7 keys have defaults, but the shipped files spell them out."""
    raw = load_yaml_mapping(path)
    assert set(raw["integration"]) == {"method", "rtol", "atol", "max_steps", "step_size", "lambda_max"}
    assert set(raw["raytrace"]) == {"resolution", "fov", "backend", "dtype"}
    assert set(raw["observer"]) == {"radius", "inclination_deg", "phi_deg"}
    assert {"name", "description", "seed", "output_dir", "report_dir", "parameters"} <= set(
        raw["experiment"]
    )


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_config_files_use_development_resolution(path: Path) -> None:
    assert load_config(path).raytrace.resolution == DEVELOPMENT_RESOLUTION


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_config_files_define_section_7_sub_blocks(path: Path) -> None:
    cfg = load_config(path)
    expected = EXPECTED_SUB_BLOCKS[path.stem]
    assert set(cfg.experiment.parameters) == set(expected)
    for name, block_type in expected.items():
        block = experiment_block(cfg, name, block_type)
        assert isinstance(block, block_type)
        for spec in dataclasses.fields(block):
            assert getattr(block, spec.name) is not None or spec.name == "r_in"


def test_gpu_and_benchmark_files_share_the_benchmark_block() -> None:
    gpu = load_config(CONFIG_DIR / "gpu.yaml")
    bench = load_config(CONFIG_DIR / "benchmark.yaml")
    assert gpu.experiment.parameters["benchmark"] == bench.experiment.parameters["benchmark"]
    assert experiment_block(gpu, "benchmark", _Benchmark) == experiment_block(
        bench, "benchmark", _Benchmark
    )


def test_schwarzschild_config_has_zero_spin() -> None:
    """Spin 0 is what defines the Schwarzschild configuration (PROJECT.md section 12)."""
    assert load_config(CONFIG_DIR / "schwarzschild.yaml").black_hole.spin == 0.0


def test_section_34_example_loads_verbatim() -> None:
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE)
    assert cfg.black_hole.mass == 1.0
    assert cfg.black_hole.spin == 0.9
    assert cfg.observer.radius == 1000.0
    assert cfg.observer.inclination_deg == 60.0
    assert cfg.observer.phi_deg == 0.0
    assert cfg.integration.method == "rk45"
    assert cfg.integration.rtol == pytest.approx(1e-9)
    assert cfg.integration.atol == pytest.approx(1e-11)
    assert cfg.integration.max_steps == 100000
    assert cfg.raytrace.resolution == 512
    assert cfg.termination.horizon_epsilon == pytest.approx(1e-6)
    assert cfg.termination.escape_radius == 1000.0
    assert cfg.experiment.name == DEFAULT_EXPERIMENT_NAME
    assert cfg.experiment.description == ""
    assert dict(cfg.experiment.parameters) == {}


def test_section_7_keys_have_documented_defaults() -> None:
    """docs/architecture.md section 7: older files without the new keys still load."""
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE)
    assert cfg.integration.step_size == 0.01
    assert cfg.integration.lambda_max == 1.0e4
    assert cfg.raytrace.fov == 12.0
    assert cfg.raytrace.backend == "numpy"
    assert cfg.raytrace.dtype == "float64"
    assert cfg.experiment.seed == DEFAULT_SEED == 0
    assert cfg.experiment.output_dir == DEFAULT_OUTPUT_DIR == "runs"
    assert cfg.experiment.report_dir == DEFAULT_REPORT_DIR == "reports"


def test_section_7_keys_are_read_from_files() -> None:
    cfg = load_config(CONFIG_DIR / "kerr.yaml")
    raw = load_yaml_mapping(CONFIG_DIR / "kerr.yaml")
    assert cfg.integration.step_size == float(raw["integration"]["step_size"])
    assert cfg.integration.lambda_max == float(raw["integration"]["lambda_max"])
    assert cfg.raytrace.fov == float(raw["raytrace"]["fov"])
    assert cfg.raytrace.backend == raw["raytrace"]["backend"]
    assert cfg.raytrace.dtype == raw["raytrace"]["dtype"]
    assert cfg.experiment.seed == raw["experiment"]["seed"]
    assert cfg.experiment.output_dir == raw["experiment"]["output_dir"]
    assert cfg.experiment.report_dir == raw["experiment"]["report_dir"]


def test_section_34_example_matches_project_md() -> None:
    project = REPO_ROOT / "PROJECT.md"
    if not project.is_file():
        pytest.skip("PROJECT.md not present")
    text = project.read_text(encoding="utf-8")
    match = re.search(r"^# 34\. Configuration.*?```yaml\n(.*?)```", text, re.DOTALL | re.MULTILINE)
    assert match is not None, "section 34 YAML example not found in PROJECT.md"
    assert match.group(1).strip() == SECTION_34_EXAMPLE.strip()


def test_experiment_name_derived_from_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "my_run.yaml"
    path.write_text(SECTION_34_EXAMPLE, encoding="utf-8")
    assert load_config(path).experiment.name == "my_run"


def test_experiment_block_without_name_uses_derived_name(tmp_path: Path) -> None:
    path = tmp_path / "described.yaml"
    path.write_text(SECTION_34_EXAMPLE + "experiment:\n  description: hello\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.experiment.name == "described"
    assert cfg.experiment.description == "hello"


def test_empty_experiment_block_uses_defaults() -> None:
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE + "experiment:\n")
    assert cfg.experiment.name == DEFAULT_EXPERIMENT_NAME
    assert cfg.experiment.description == ""


def test_missing_experiment_block_allowed_in_mapping(base: dict[str, Any]) -> None:
    del base["experiment"]
    cfg = KerrRayConfig.from_mapping(base, default_name="from-caller")
    assert cfg.experiment.name == "from-caller"


def test_unknown_experiment_key_rejected() -> None:
    with pytest.raises(ConfigError, match="owner"):
        KerrRayConfig.from_yaml(SECTION_34_EXAMPLE + "experiment:\n  owner: me\n")


def test_experiment_parameters_are_preserved(tmp_path: Path) -> None:
    text = SECTION_34_EXAMPLE + (
        "experiment:\n"
        "  name: sweep\n"
        "  parameters:\n"
        "    spins: [0.0, 0.5]\n"
        "    grid: {n: 4, label: coarse}\n"
    )
    cfg = KerrRayConfig.from_yaml(text)
    assert cfg.experiment.name == "sweep"
    assert cfg.experiment.parameters["spins"] == [0.0, 0.5]
    assert cfg.experiment.parameters["grid"] == {"n": 4, "label": "coarse"}
    written = dump_config(cfg, tmp_path / "sweep.yaml")
    assert load_config(written) == cfg
    assert load_config(written).to_mapping()["experiment"]["parameters"] == {
        "spins": [0.0, 0.5],
        "grid": {"n": 4, "label": "coarse"},
    }


def test_experiment_parameters_are_read_only() -> None:
    cfg = KerrRayConfig.from_yaml(
        SECTION_34_EXAMPLE + "experiment:\n  parameters:\n    spins: [0.0, 0.5]\n"
    )
    params = cfg.experiment.parameters
    assert isinstance(params, Mapping)
    with pytest.raises(TypeError):
        params["spins"] = []  # type: ignore[index]
    with pytest.raises(TypeError):
        del params["spins"]  # type: ignore[attr-defined]
    exported = cfg.to_mapping()
    exported["experiment"]["parameters"]["spins"].append(0.9)
    exported["experiment"]["parameters"]["extra"] = True
    assert cfg.experiment.parameters["spins"] == [0.0, 0.5]
    assert "extra" not in cfg.experiment.parameters


# --- experiment_block ---------------------------------------------------------


def _cfg_with_parameters(yaml_parameters: str) -> KerrRayConfig:
    return KerrRayConfig.from_yaml(
        SECTION_34_EXAMPLE + "experiment:\n  parameters:\n" + yaml_parameters
    )


def test_experiment_block_absent_returns_defaults() -> None:
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE)
    assert experiment_block(cfg, "lensing", _Lensing) == _Lensing()
    assert experiment_block(cfg, "disk", _Disk) == _Disk()


def test_experiment_block_empty_sub_block_returns_defaults() -> None:
    cfg = _cfg_with_parameters("    lensing:\n")
    assert experiment_block(cfg, "lensing", _Lensing) == _Lensing()


def test_experiment_block_parses_scalars_lists_and_strings() -> None:
    cfg = _cfg_with_parameters(
        "    lensing: {impact_min: 4, impact_max: '15.5', n_rays: 8}\n"
        "    convergence:\n"
        "      resolutions: [32, 64.0, '128']\n"
        "      tolerances: [1e-6, 1.0e-8]\n"
        "    benchmark: {backends: [numpy], dtypes: [float64], repeats: 1}\n"
    )
    lensing = experiment_block(cfg, "lensing", _Lensing)
    assert lensing == _Lensing(impact_min=4.0, impact_max=15.5, n_rays=8)
    assert isinstance(lensing.impact_min, float)
    convergence = experiment_block(cfg, "convergence", _Convergence)
    assert convergence.resolutions == [32, 64, 128]
    assert all(isinstance(value, int) for value in convergence.resolutions)
    assert convergence.tolerances == pytest.approx([1e-6, 1e-8])
    assert convergence.step_sizes == _Convergence().step_sizes  # default kept
    benchmark = experiment_block(cfg, "benchmark", _Benchmark)
    assert benchmark.backends == ["numpy"] and benchmark.repeats == 1
    assert benchmark.ray_counts == _Benchmark().ray_counts


def test_experiment_block_optional_float_and_bool() -> None:
    cfg = _cfg_with_parameters("    disk: {enabled: true, r_in: null, prograde: false}\n")
    disk = experiment_block(cfg, "disk", _Disk)
    assert disk.enabled is True and disk.prograde is False and disk.r_in is None
    cfg = _cfg_with_parameters("    disk: {r_in: 5}\n")
    assert experiment_block(cfg, "disk", _Disk).r_in == 5.0


@pytest.mark.parametrize(
    ("yaml_parameters", "block", "block_type", "fragment"),
    [
        ("    lensing: {impact_minimum: 3.0}\n", "lensing", _Lensing, "impact_minimum"),
        ("    lensing: {n_rays: 2.5}\n", "lensing", _Lensing, "lensing.n_rays"),
        ("    lensing: [3, 20]\n", "lensing", _Lensing, "must be a mapping"),
        ("    convergence: {resolutions: 64}\n", "convergence", _Convergence, "list"),
        ("    convergence: {resolutions: [64, x]}\n", "convergence", _Convergence, r"resolutions\[1\]"),
        ("    disk: {enabled: 'yes'}\n", "disk", _Disk, "boolean"),
        ("    disk: {r_in: 'inner'}\n", "disk", _Disk, "disk.r_in"),
        ("    benchmark: {backends: numpy}\n", "benchmark", _Benchmark, "list"),
    ],
)
def test_experiment_block_rejects_bad_values(
    yaml_parameters: str, block: str, block_type: type, fragment: str
) -> None:
    cfg = _cfg_with_parameters(yaml_parameters)
    with pytest.raises(ConfigError, match=fragment):
        experiment_block(cfg, block, block_type)


@dataclasses.dataclass(frozen=True)
class _Required:
    spins: list[float]


def test_experiment_block_missing_required_field_is_config_error() -> None:
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE)
    with pytest.raises(ConfigError, match="experiment.parameters.sweep"):
        experiment_block(cfg, "sweep", _Required)
    cfg = _cfg_with_parameters("    sweep: {}\n")
    with pytest.raises(ConfigError, match="experiment.parameters.sweep"):
        experiment_block(cfg, "sweep", _Required)


# --- validation -----------------------------------------------------------------


@pytest.mark.parametrize("spin", [1.0, -1.0, 1.5, float("nan"), float("inf")])
def test_invalid_spin_rejected(base: dict[str, Any], spin: float) -> None:
    base["black_hole"]["spin"] = spin
    with pytest.raises(ConfigError, match="spin"):
        KerrRayConfig.from_mapping(base)


@pytest.mark.parametrize("mass", [0.0, -1.0, float("nan")])
def test_invalid_mass_rejected(base: dict[str, Any], mass: float) -> None:
    base["black_hole"]["mass"] = mass
    with pytest.raises(ConfigError, match="mass"):
        KerrRayConfig.from_mapping(base)


@pytest.mark.parametrize("inclination", [-1.0, 180.5])
def test_invalid_inclination_rejected(base: dict[str, Any], inclination: float) -> None:
    base["observer"]["inclination_deg"] = inclination
    with pytest.raises(ConfigError, match="inclination_deg"):
        KerrRayConfig.from_mapping(base)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (0.0, INCLINATION_EPSILON_DEG),
        (1e-5, INCLINATION_EPSILON_DEG),
        (180.0, 180.0 - INCLINATION_EPSILON_DEG),
        (179.9999, 180.0 - INCLINATION_EPSILON_DEG),
    ],
)
def test_axis_inclination_is_clamped_with_warning(
    base: dict[str, Any], caplog: pytest.LogCaptureFixture, given: float, expected: float
) -> None:
    """D-008: the ZAMO tetrad is singular on the axis, so the observer is moved off it."""
    base["observer"]["inclination_deg"] = given
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        cfg = KerrRayConfig.from_mapping(base)
    assert cfg.observer.inclination_deg == expected
    assert cfg.observer.inclination_rad == pytest.approx(math.radians(expected))
    assert any("D-008" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("given", [INCLINATION_EPSILON_DEG, 45.0, 90.0, 180.0 - INCLINATION_EPSILON_DEG])
def test_off_axis_inclination_is_not_clamped(
    base: dict[str, Any], caplog: pytest.LogCaptureFixture, given: float
) -> None:
    base["observer"]["inclination_deg"] = given
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        cfg = KerrRayConfig.from_mapping(base)
    assert cfg.observer.inclination_deg == given
    assert cfg.observer.inclination_rad == math.radians(given)
    assert not [r for r in caplog.records if "D-008" in r.getMessage()]


def test_clamped_inclination_round_trips_without_second_warning(
    base: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    base["observer"]["inclination_deg"] = 0.0
    cfg = KerrRayConfig.from_mapping(base)  # warns once (D-008)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        again = KerrRayConfig.from_yaml(cfg.to_yaml())
    assert again == cfg
    assert not [r for r in caplog.records if "D-008" in r.getMessage()]


def test_dop853_is_an_accepted_method(base: dict[str, Any]) -> None:
    """D-005 (updated): dop853 is the SciPy reference integrator for cross-checks."""
    base["integration"]["method"] = "dop853"
    assert KerrRayConfig.from_mapping(base).integration.method == "dop853"
    assert SUPPORTED_INTEGRATION_METHODS == {"rk4", "rk45", "dop853"}


@pytest.mark.parametrize(
    ("block", "key", "value", "fragment"),
    [
        ("observer", "radius", 0.0, "radius"),
        ("raytrace", "resolution", 0, "resolution"),
        ("raytrace", "fov", 0.0, "fov"),
        ("raytrace", "fov", float("inf"), "fov"),
        ("raytrace", "backend", "opencl", "backend"),
        ("raytrace", "dtype", "float16", "dtype"),
        ("integration", "rtol", 0.0, "rtol"),
        ("integration", "atol", -1e-11, "atol"),
        ("integration", "max_steps", 0, "max_steps"),
        ("integration", "method", "euler", "method"),
        ("integration", "step_size", 0.0, "step_size"),
        ("integration", "step_size", -0.01, "step_size"),
        ("integration", "lambda_max", 0.0, "lambda_max"),
        ("integration", "lambda_max", float("nan"), "lambda_max"),
        ("termination", "horizon_epsilon", 0.0, "horizon_epsilon"),
        ("termination", "escape_radius", -5.0, "escape_radius"),
        ("raytrace", "resolution", "many", "resolution"),
        ("black_hole", "mass", True, "mass"),
        ("experiment", "name", "", "name"),
        ("experiment", "parameters", [1, 2], "parameters"),
        ("experiment", "seed", -1, "seed"),
        ("experiment", "seed", 1.5, "seed"),
        ("experiment", "output_dir", "", "output_dir"),
        ("experiment", "report_dir", " ", "report_dir"),
    ],
)
def test_constraint_violations_rejected(
    base: dict[str, Any], block: str, key: str, value: Any, fragment: str
) -> None:
    base[block][key] = value
    with pytest.raises(ConfigError, match=fragment):
        KerrRayConfig.from_mapping(base)


def test_unknown_top_level_key_rejected(base: dict[str, Any]) -> None:
    base["accretion"] = {"emissivity_index": 3}
    with pytest.raises(ConfigError, match="accretion"):
        KerrRayConfig.from_mapping(base)


def test_unknown_block_key_rejected(base: dict[str, Any]) -> None:
    base["observer"]["theta_deg"] = 60.0
    with pytest.raises(ConfigError, match="theta_deg"):
        KerrRayConfig.from_mapping(base)


def test_missing_key_rejected(base: dict[str, Any]) -> None:
    del base["integration"]["atol"]
    with pytest.raises(ConfigError, match="integration.atol"):
        KerrRayConfig.from_mapping(base)


@pytest.mark.parametrize(
    "block", ["black_hole", "observer", "integration", "raytrace", "termination"]
)
def test_missing_physics_block_rejected(base: dict[str, Any], block: str) -> None:
    del base[block]
    with pytest.raises(ConfigError, match=f"missing required block '{block}'"):
        KerrRayConfig.from_mapping(base)


def test_error_message_names_source(base: dict[str, Any]) -> None:
    base["black_hole"]["spin"] = 2.0
    with pytest.raises(ConfigError, match="my-config.yaml"):
        KerrRayConfig.from_mapping(base, source="my-config.yaml")


def test_yaml_scientific_notation_strings_are_coerced() -> None:
    assert "rtol: 1e-9" in SECTION_34_EXAMPLE  # PyYAML reads this spelling as a string
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE)
    assert cfg.integration.rtol == pytest.approx(1e-9)
    assert cfg.integration.atol == pytest.approx(1e-11)


def test_phi_deg_defaults_to_zero(base: dict[str, Any]) -> None:
    del base["observer"]["phi_deg"]
    assert KerrRayConfig.from_mapping(base).observer.phi_deg == 0.0


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_yaml_round_trip(path: Path, tmp_path: Path) -> None:
    cfg = load_config(path)
    written = dump_config(cfg, tmp_path / path.name)
    assert load_config(written) == cfg
    assert KerrRayConfig.from_yaml(cfg.to_yaml()) == cfg
    assert b"\r\n" not in written.read_bytes()


def test_dotted_override_merging(base: dict[str, Any]) -> None:
    overrides = {
        "black_hole.spin": 0.5,
        "observer.inclination_deg": "30",
        "raytrace.resolution": "64",
        "raytrace.backend": "numba",
        "experiment.seed": "7",
    }
    merged = apply_overrides(base, overrides)
    assert merged["black_hole"]["spin"] == 0.5
    assert base["black_hole"]["spin"] != 0.5  # the input mapping is untouched
    cfg = KerrRayConfig.from_mapping(merged)
    assert cfg.black_hole.spin == 0.5
    assert cfg.observer.inclination_deg == 30.0
    assert cfg.raytrace.resolution == 64
    assert cfg.raytrace.backend == "numba"
    assert cfg.experiment.seed == 7


def test_override_with_unknown_key_rejected(base: dict[str, Any]) -> None:
    merged = apply_overrides(base, {"observer.theta_deg": 60.0})
    with pytest.raises(ConfigError, match="theta_deg"):
        KerrRayConfig.from_mapping(merged)


@pytest.mark.parametrize("key", ["", "black_hole..spin", "black_hole.mass.value"])
def test_malformed_override_rejected(base: dict[str, Any], key: str) -> None:
    with pytest.raises(ConfigError):
        apply_overrides(base, {key: 1.0})


def test_with_overrides_returns_new_config() -> None:
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE, default_name="orig")
    updated = cfg.with_overrides({"raytrace.resolution": 64})
    assert updated.raytrace.resolution == 64
    assert cfg.raytrace.resolution == 512
    assert updated.black_hole == cfg.black_hole
    assert updated.experiment.name == "orig"


def test_load_config_applies_overrides() -> None:
    cfg = load_config(CONFIG_DIR / "kerr.yaml", {"black_hole.spin": 0.0})
    assert cfg.black_hole.spin == 0.0


def test_escape_radius_below_observer_radius_warns(
    base: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    base["termination"]["escape_radius"] = 500.0
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        KerrRayConfig.from_mapping(base)
    assert any("escape_radius" in record.getMessage() for record in caplog.records)


def test_equal_escape_and_observer_radius_does_not_warn(
    base: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    assert base["termination"]["escape_radius"] == base["observer"]["radius"]
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        KerrRayConfig.from_mapping(base)
    assert not [r for r in caplog.records if "escape_radius" in r.getMessage()]


def test_missing_file_reports_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="nope.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_non_utf8_file_reports_path(tmp_path: Path) -> None:
    path = tmp_path / "latin1.yaml"
    path.write_bytes("# café\n".encode("cp1252") + SECTION_34_EXAMPLE.encode("ascii"))
    with pytest.raises(ConfigError, match="latin1.yaml"):
        load_config(path)


def test_empty_document_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(empty)


def test_config_is_immutable() -> None:
    cfg = load_config(CONFIG_DIR / "kerr.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.black_hole.mass = 2.0  # type: ignore[misc]


@dataclasses.dataclass(frozen=True)
class _UnsupportedBlock:
    """A block with an annotation the schema has no converter for."""

    limit: "complex" = 0j


def test_parse_block_reports_missing_converter() -> None:
    with pytest.raises(TypeError, match=r"'demo\.limit'.*'complex'.*CONVERTERS"):
        parse_block(_UnsupportedBlock, {"limit": 1.0}, "demo")


def test_validate_block_type_reports_missing_converter() -> None:
    with pytest.raises(TypeError, match=r"_UnsupportedBlock\.limit"):
        validate_block_type(_UnsupportedBlock)
    with pytest.raises(TypeError, match="not a dataclass"):
        validate_block_type(dict)


def test_experiment_block_reports_missing_converter() -> None:
    cfg = KerrRayConfig.from_yaml(SECTION_34_EXAMPLE)
    with pytest.raises(TypeError, match="CONVERTERS"):
        experiment_block(cfg, "demo", _UnsupportedBlock)
