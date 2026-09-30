"""Tests for the KerrRay Desktop JSON API and server (kerrray.web)."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request

import pytest

from kerrray.web.api import REF_CRITICAL_B, ApiError, dispatch
from kerrray.web.server import bind


def test_blackhole_matches_orbit_module_for_schwarzschild() -> None:
    d = dispatch("blackhole", {"spin": 0.0})
    assert d["r_plus"] == pytest.approx(2.0)
    assert d["b_crit_prograde"] == pytest.approx(REF_CRITICAL_B, rel=1e-12)
    assert d["isco_prograde"] == pytest.approx(6.0, rel=1e-12)


def test_geodesic_payload_is_json_and_conserves() -> None:
    d = dispatch("geodesic", {"spin": 0.9, "b": 3.0, "r0": 30.0})
    json.dumps(d, allow_nan=False)
    assert d["state"] == "ESCAPED"
    assert len(d["x"]) == len(d["y"]) == len(d["lam"])
    assert d["max_null_error"] < 1e-8
    assert d["max_energy_drift"] <= 1e-13


def test_shadow_small_grid() -> None:
    d = dispatch("shadow", {"spin": 0.0, "inclination": 60, "resolution": 12, "fov": 8, "rtol": 1e-7})
    json.dumps(d, allow_nan=False)
    assert len(d["state"]) == 144
    assert sum(d["counts"].values()) == 144
    assert d["counts"].get("CAPTURED", 0) > 0 and d["counts"].get("ESCAPED", 0) > 0
    # the analytic Schwarzschild shadow is a disk of radius 3 sqrt(3) M
    assert d["shadow_area_analytic"] == pytest.approx(math.pi * REF_CRITICAL_B**2, rel=1e-3)


def test_spin_sweep_monotonic_prograde_b() -> None:
    d = dispatch("spin_sweep", {"n": 10})
    assert all(b1 > b2 for b1, b2 in zip(d["b_pro"], d["b_pro"][1:]))


@pytest.mark.parametrize(
    ("name", "params"),
    [("shadow", {"spin": 1.5}), ("geodesic", {"b": -1}), ("nope", {})],
)
def test_bad_requests_raise(name: str, params: dict) -> None:
    with pytest.raises(ApiError):
        dispatch(name, params)


def test_server_roundtrip_and_exclusive_port() -> None:
    srv = bind(18741)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            assert b"<title>KerrRay</title>" in r.read()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/blackhole", data=b'{"spin": 0.5}', method="POST")
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["spin"] == 0.5
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/static/../api.py")
        assert err.value.code == 404
        second = bind(port, attempts=2)  # the port is taken, so the next one is used
        assert second.server_address[1] != port
        second.server_close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_photon_beam_paths_share_a_time_grid() -> None:
    d = dispatch("photon_beam", {"spin": 0.9, "n": 12, "frames": 60})
    json.dumps(d, allow_nan=False)
    n = d["n_photons"]
    assert n == 12 + sum(d["near_critical"])
    assert len(d["x"]) == len(d["y"]) == n and all(len(row) == 60 for row in d["x"])
    assert d["captured"] + d["escaped"] == n
    assert d["t"][0] == 0.0 and d["t"][-1] > 30.0
    assert d["max_null_error_escaped"] < 1e-6


def test_photon_beam_side_asymmetry_follows_spin() -> None:
    """Photons moving with the spin (upper side for a > 0) are captured closer in."""
    d = dispatch("photon_beam", {"spin": 0.9, "n": 40, "frames": 30})
    b = [bi for bi, s in zip(d["b"], d["state"]) if s == "CAPTURED"]
    assert max(b) < d["b_crit_prograde"] * 1.001
    assert -min(b) < d["b_crit_retrograde"] * 1.001
    assert max(b) < -min(b)


def _wait(job: dict, timeout: float = 120.0) -> dict:
    import time

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        r = dispatch("job_poll", job)
        if r["status"] != "running":
            return r
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_shadow_job_reports_progress_and_result() -> None:
    job = dispatch("job_start", {"kind": "shadow", "params": {"spin": 0.0, "resolution": 12, "fov": 8, "rtol": 1e-7}})
    r = _wait(job)
    assert r["status"] == "done"
    assert r["progress"]["done"] == r["progress"]["total"] == 144
    assert 0.99 <= r["progress"]["fraction"] <= 1.0
    assert sum(r["result"]["counts"].values()) == 144


def test_job_cancel_and_validation_progress() -> None:
    job = dispatch("job_start", {"kind": "shadow", "params": {"resolution": 64}})
    dispatch("job_cancel", job)
    assert _wait(job)["status"] == "cancelled"
    v = _wait(dispatch("job_start", {"kind": "validate", "params": {"which": "photon_sphere", "tol": 1e-3}}))
    assert v["status"] == "done" and v["progress"]["check"] == "photon_sphere"
    assert abs(v["result"]["checks"][0]["computed"] - 3.0) < 2e-3
    with pytest.raises(ApiError):
        dispatch("job_start", {"kind": "nope"})
