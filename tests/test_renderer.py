"""Tests for kerrray.raytracing.renderer, the report figures and the render
command (PROJECT.md sections 23 and 24; docs/rendering.md sections 4 and 5)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from kerrray.commands import render as render_cmd
from kerrray.geodesics import IntegratorOptions, TerminationOptions, equatorial_photon, integrate
from kerrray.geometry import kerr, schwarzschild
from kerrray.photons import TerminationState
from kerrray.physics.accretion import DiskModel, DiskParams
from kerrray.physics.redshift import keplerian_angular_velocity
from kerrray.physics.accretion import event_options
from kerrray.raytracing.backends import BackendUnavailable
from kerrray.raytracing.backends.numba_backend import NumbaBackend, numba_available
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.raytracing.renderer import (
    DiskImage,
    integrate_rays,
    integrator_options_from_config,
    load_disk_image_arrays,
    render_disk,
    save_disk_image,
    termination_options_from_config,
)
from kerrray.reporting.figures import disk_figure, lensing_figure, shadow_figure, trajectory_figure
from kerrray.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "shadow.yaml"
INTEG = IntegratorOptions(method="rk45", rtol=1e-8, atol=1e-10, lambda_max=4000.0)
TERM = TerminationOptions(horizon_epsilon=1e-6, escape_radius=1000.0)


def _render(spin: float, inclination: float, *, prograde: bool = True, resolution: int = 24) -> DiskImage:
    st = kerr(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=inclination, fov=25.0, resolution=resolution)
    disk = DiskModel.from_params(st, DiskParams(r_out=20.0, emissivity_index=3.0, intensity_law="g4", prograde=prograde))
    return render_disk(st, cam, INTEG, TERM, disk)


@pytest.fixture(scope="module")
def image_60() -> DiskImage:
    return _render(0.0, 60.0)


def test_disk_render_schwarzschild_60_degrees(image_60: DiskImage) -> None:
    img = image_60
    assert img.shape == (24, 24) and img.n_hits > 0
    assert img.counts()["DISK_HIT"] == img.n_hits
    assert np.array_equal(img.hit, img.state == int(TerminationState.DISK_HIT))
    g = img.g[img.hit]
    assert np.all((g > 0.0) & (g < 2.0))
    assert np.all(np.isnan(img.g[~img.hit])) and np.all(np.isnan(img.r_hit[~img.hit]))
    assert np.all(np.isfinite(img.intensity)) and np.all(img.intensity[~img.hit] == 0.0)
    assert np.all(img.intensity[img.hit] > 0.0)
    r_hit = img.r_hit[img.hit]
    assert np.all((r_hit >= img.disk.r_in) & (r_hit <= img.disk.r_out))
    assert np.all(img.hit_states[img.hit][:, 2] == 0.5 * math.pi)
    expected = (r_hit / img.disk.r_in) ** -3.0 * g**4
    assert np.allclose(img.intensity[img.hit], expected, rtol=1e-12)


def test_approaching_side_is_brighter(image_60: DiskImage) -> None:
    """Derivation (docs/raytracing.md section 4, docs/rendering.md section 5).

    alpha > 0 is the +phi side of the sky: a physical photon seen there arrives
    moving towards -phi, so L_z < 0 (checked below through the reversal-invariant
    xi = L_z / E of the stored state). A prograde disk around a hole with a >= 0
    has Omega > 0, so gas on the alpha < 0 limb (phi = phi_o - pi/2) moves
    towards the observer and g = 1 / [u^t (1 - Omega xi)] is larger there:
    the left side (alpha < 0) is the approaching, brighter side.
    """
    img = image_60
    hit = img.hit
    xi = img.hit_states[..., 7] / (-img.hit_states[..., 4])  # L_z / E, invariant under reversal
    assert np.all(np.sign(xi[hit]) == -np.sign(img.alpha[hit]))
    left, right = hit & (img.alpha < 0), hit & (img.alpha > 0)
    assert left.any() and right.any()
    assert np.mean(img.g[left]) > np.mean(img.g[right])
    assert np.mean(img.intensity[left]) > np.mean(img.intensity[right])
    assert np.max(img.g[left]) > 1.0 > np.min(img.g[right])


def test_retrograde_disk_flips_the_bright_side() -> None:
    img = _render(0.9, 60.0, prograde=False, resolution=16)
    assert img.n_hits > 0
    left, right = img.hit & (img.alpha < 0), img.hit & (img.alpha > 0)
    assert np.mean(img.g[right]) > np.mean(img.g[left])
    assert img.disk.r_in > 8.0  # retrograde ISCO of a = 0.9 lies beyond 8 M


def test_face_on_map_is_mirror_symmetric_up_to_the_derived_doppler_term() -> None:
    """i = 0 is clamped to 1e-3 deg (D-008). Exactly face-on the map is symmetric under alpha -> -alpha;
    at the clamp angle xi = L_z / E ~ -alpha sin(theta_o) survives and g(alpha) - g(-alpha) =
    2 Omega xi / [u^t (1 - Omega^2 xi^2)] (measured 1.1e-5, docs/rendering.md section 5). The
    Doppler-free factor g (1 - Omega xi) = 1 / u^t and r_hit are mirror symmetric to 1e-6."""
    img = _render(0.0, 0.0, resolution=24)
    hit = img.hit
    assert hit.any() and np.array_equal(hit, hit[:, ::-1])
    both = hit & hit[:, ::-1]
    r_mirror = np.abs(img.r_hit - img.r_hit[:, ::-1])[both]
    assert np.max(r_mirror) < 1e-6
    omega = keplerian_angular_velocity(img.spacetime, np.where(hit, img.r_hit, 10.0), True)
    xi = img.hit_states[..., 7] / (-img.hit_states[..., 4])
    doppler_free = img.g * (1.0 - omega * xi)
    assert np.max(np.abs(doppler_free - doppler_free[:, ::-1])[both]) < 1e-6
    asym = np.abs(img.g - img.g[:, ::-1])[both]
    bound = (2.0 * img.g * omega * np.abs(xi) / (1.0 - (omega * xi) ** 2))[both]
    assert np.max(asym) <= np.max(bound) + 1e-6
    assert np.max(asym) < 1e-4


def test_npz_round_trip(tmp_path: Path, image_60: DiskImage) -> None:
    path = save_disk_image(tmp_path / "img.npz", image_60)
    data = load_disk_image_arrays(path)
    assert np.array_equal(data["hit"], image_60.hit)
    assert np.allclose(data["intensity"], image_60.intensity)
    assert data["resolution"] == 24 and data["spin"] == 0.0 and data["redshift_power"] == 4
    assert data["r_in"] == image_60.disk.r_in and data["inclination_deg"] == 60.0


def test_backend_resolution() -> None:
    st = schwarzschild()
    y0 = equatorial_photon(st, 100.0, 8.0)[None, :]
    integ = IntegratorOptions(rtol=1e-7, atol=1e-9, lambda_max=500.0)
    res = integrate_rays(st, y0, integ, TerminationOptions(escape_radius=100.0), backend="numpy")
    assert res.state[0] == int(TerminationState.ESCAPED)
    with pytest.raises(RuntimeError):
        integrate_rays(st, y0, integ, TERM, backend="cuda")
    with pytest.raises(BackendUnavailable, match="optional future work"):
        integrate_rays(st, y0, integ, TERM, backend="cuda")


def test_chunked_render_with_progress_equals_single_chunk_and_numba_kernel() -> None:
    """Chunking and the backend do not change the image: every ray carries its own step control.

    The numba kernel runs compiled when Numba is installed and interpreted otherwise
    (a 4 x 4 image keeps the interpreted run short); it must reproduce the numpy
    disk image, hit mask and redshift map.
    """
    st = kerr(1.0, 0.9)
    disk = DiskModel.from_params(st, DiskParams(r_out=20.0))
    cam = Camera(radius=1000.0, inclination_deg=75.0, fov=25.0, resolution=6)
    whole = render_disk(st, cam, INTEG, TERM, disk)
    fractions: list[float] = []
    y0 = initial_states(cam, st)
    chunked = integrate_rays(st, y0, INTEG, TERM, events=event_options(disk), progress=fractions.append, chunk_size=10)
    assert fractions == [10 / 36, 20 / 36, 30 / 36, 1.0]
    assert np.array_equal(chunked.state, whole.state.ravel()) and np.array_equal(chunked.Y, whole.result.Y)
    small = Camera(radius=1000.0, inclination_deg=75.0, fov=25.0, resolution=4)
    ref = render_disk(st, small, INTEG, TERM, disk)
    engine = NumbaBackend(interpreted=not numba_available())
    other = render_disk(st, small, INTEG, TERM, disk, backend=engine)
    assert other.backend == "numba" and ref.n_hits > 0
    assert np.array_equal(other.hit, ref.hit) and np.array_equal(other.state, ref.state)
    assert np.allclose(other.g[ref.hit], ref.g[ref.hit], rtol=1e-9, atol=0.0)
    assert np.allclose(other.intensity, ref.intensity, rtol=1e-9, atol=0.0)


def test_options_from_config() -> None:
    cfg = load_config(CONFIG, {"integration.rtol": 1e-7, "raytrace.dtype": "float32", "termination.escape_radius": 900.0})
    integ = integrator_options_from_config(cfg)
    assert integ.method == "rk45" and integ.rtol == 1e-7 and integ.dtype == "float32"
    assert integ.lambda_max == cfg.integration.lambda_max and integ.max_steps == cfg.integration.max_steps
    term = termination_options_from_config(cfg)
    assert term.escape_radius == 900.0 and term.horizon_epsilon == cfg.termination.horizon_epsilon


def test_figures_are_written(tmp_path: Path, image_60: DiskImage) -> None:
    for quantity, stretch in (("intensity", "linear"), ("intensity", "log"), ("redshift", "linear")):
        path = disk_figure(tmp_path / f"disk_{quantity}_{stretch}.png", image_60, quantity=quantity, stretch=stretch)
        assert path.is_file() and path.stat().st_size > 0
    with pytest.raises(ValueError):
        disk_figure(tmp_path / "bad.png", image_60, stretch="sqrt")
    st = kerr(1.0, 0.9)
    trajs = [
        integrate(st, equatorial_photon(st, 30.0, b, prograde=p), IntegratorOptions(rtol=1e-7, atol=1e-9, lambda_max=200.0),
                  TerminationOptions(escape_radius=30.0))
        for b, p in ((6.0, True), (6.0, False))
    ]
    path = trajectory_figure(tmp_path / "traj.png", trajs, st, labels=["prograde", "retrograde"], title="rays")
    assert path.is_file()
    with pytest.raises(ValueError):
        trajectory_figure(tmp_path / "none.png", [], st)

    class _Shadow:
        alpha, beta, captured = image_60.alpha, image_60.beta, image_60.state == int(TerminationState.CAPTURED)

    assert shadow_figure(tmp_path / "shadow.png", _Shadow(), (np.cos(np.linspace(0, 6.3, 50)) * 5, np.sin(np.linspace(0, 6.3, 50)) * 5)).is_file()
    b = np.linspace(6.0, 20.0, 8)
    assert lensing_figure(tmp_path / "lens.png", b, {"a": 4.0 / b, "b": 5.0 / b}, logx=True).is_file()
    with pytest.raises(ValueError):
        lensing_figure(tmp_path / "bad_lens.png", b, {"a": b[:-1]})


def _group_app() -> typer.Typer:
    """A fresh Typer app in sub-command mode, as in ``kerrray.cli``.

    A Typer app with a single command and no callback runs that command as the
    application itself, so the command name would be an unexpected argument;
    the no-op callback makes the app a command group like the real CLI.
    """
    app = typer.Typer(add_completion=False)

    @app.callback()
    def _root() -> None:
        """Test application root (forces sub-command mode)."""

    return app


def test_render_command_writes_pngs_npz_and_report(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, {
        "experiment.output_dir": str(tmp_path / "runs"), "experiment.report_dir": str(tmp_path / "reports"),
        "raytrace.fov": 25.0, "integration.rtol": 1e-7, "integration.atol": 1e-9,
    })
    cfg_path = tmp_path / "render.yaml"
    cfg_path.write_text(cfg.to_yaml(), encoding="utf-8")
    app = _group_app()
    render_cmd.register(app)
    result = CliRunner().invoke(app, [
        "render", "--config", str(cfg_path), "--spin", "0.0", "--inclination", "60", "--resolution", "8",
        "--r-out", "15", "--emissivity-index", "2", "--intensity-law", "g3",
    ])
    assert result.exit_code == 0, result.output
    assert "Disk pixels" in result.output and "g max" in result.output
    runs = list((tmp_path / "runs").iterdir())
    reports = list((tmp_path / "reports").iterdir())
    assert len(runs) == 1 and len(reports) == 1
    assert (runs[0] / render_cmd.NPZ_FILENAME).is_file() and (runs[0] / "manifest.json").is_file()
    for name in ("disk_linear.png", "disk_log.png", "disk_redshift.png", "report.md", "summary.json"):
        assert (reports[0] / name).is_file()
    data = load_disk_image_arrays(runs[0] / render_cmd.NPZ_FILENAME)
    assert data["resolution"] == 8 and data["r_out"] == 15.0 and data["redshift_power"] == 3
    bad = CliRunner().invoke(app, ["render", "--config", str(cfg_path), "--intensity-law", "g5"])
    assert bad.exit_code == 1
    gpu = CliRunner().invoke(app, ["render", "--config", str(cfg_path), "--backend", "cuda"])
    assert gpu.exit_code == 1 and "optional future work" in gpu.output
    assert len(list((tmp_path / "runs").iterdir())) == 1  # failed invocations write no run


def test_render_command_fov_flag_reaches_the_camera(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, {
        "experiment.output_dir": str(tmp_path / "runs"), "experiment.report_dir": str(tmp_path / "reports"),
        "integration.rtol": 1e-7, "integration.atol": 1e-9,
    })
    cfg_path = tmp_path / "render.yaml"
    cfg_path.write_text(cfg.to_yaml(), encoding="utf-8")
    app = _group_app()
    render_cmd.register(app)
    result = CliRunner().invoke(app, ["render", "--config", str(cfg_path), "--resolution", "4", "--fov", "22"])
    assert result.exit_code == 0, result.output
    run_dir = next((tmp_path / "runs").iterdir())
    assert load_disk_image_arrays(run_dir / render_cmd.NPZ_FILENAME)["fov"] == 22.0


@pytest.mark.parametrize("spin, prograde", [(0.0, True), (0.9, False)])
def test_interpretation_states_the_computed_doppler_comparison(spin: float, prograde: bool) -> None:
    """The report text names the approaching side from (a, prograde) and quotes the computed means."""
    image = _render(spin, 60.0, prograde=prograde, resolution=12)
    summary = render_cmd._image_summary(image)
    text = render_cmd._interpretation(image, summary)
    approaching = "alpha < 0" if prograde else "alpha > 0"  # a >= 0 in both cases
    g_app = summary["g_mean_alpha_negative"] if prograde else summary["g_mean_alpha_positive"]
    g_rec = summary["g_mean_alpha_positive"] if prograde else summary["g_mean_alpha_negative"]
    assert f"({approaching}) is {g_app:.4f}" in text and f"{g_rec:.4f}" in text
    assert g_app > g_rec and "NOT" not in text  # the Doppler boost is resolved even at 12 x 12


def test_redshift_colour_map_is_blue_for_blue_shift() -> None:
    """g > 1 (top of the colour range) must render blue and g < 1 red (physical convention)."""
    from matplotlib import colormaps

    from kerrray.reporting.figures import REDSHIFT_CMAP

    cmap = colormaps[REDSHIFT_CMAP]
    r_hi, _, b_hi, _ = cmap(1.0)
    r_lo, _, b_lo, _ = cmap(0.0)
    assert b_hi > r_hi and r_lo > b_lo
