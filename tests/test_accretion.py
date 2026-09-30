"""Tests for kerrray.physics.accretion (PROJECT.md section 23; docs/rendering.md section 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kerrray.geodesics import EventOptions
from kerrray.geometry import kerr, schwarzschild
from kerrray.photons.orbits import equatorial_photon_orbit_radius, isco_radius_numeric
from kerrray.physics.accretion import (
    INTENSITY_LAWS,
    DiskModel,
    DiskParams,
    disk_params,
    emissivity,
    event_options,
    isco_radius,
    observed_intensity,
)
from kerrray.utils.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "shadow.yaml"


def test_isco_schwarzschild_is_6M_computed() -> None:
    st = schwarzschild()
    assert abs(isco_radius(st) - 6.0 * st.mass) < 1e-12
    assert abs(isco_radius_numeric(st) - 6.0 * st.mass) < 1e-9


def test_disk_params_defaults_and_validation() -> None:
    cfg = load_config(CONFIG)
    params = disk_params(cfg)
    assert params == DiskParams()
    assert params.enabled is False and params.r_in is None and params.intensity_law == "g4"
    for bad in (
        {"r_out": -1.0},
        {"r_in": 25.0, "r_out": 20.0},
        {"intensity_law": "g5"},
        {"emissivity_index": float("inf")},
    ):
        with pytest.raises(ConfigError):
            DiskParams(**bad)
    with pytest.raises(ConfigError):
        disk_params(load_config(CONFIG, {"experiment.parameters.disk.unknown": 1}))
    with pytest.raises(ConfigError):
        disk_params(load_config(CONFIG, {"experiment.parameters.disk.intensity_law": "g2"}))


@pytest.mark.parametrize("st", [schwarzschild(), kerr(1.0, 0.9)], ids=["a=0", "a=0.9"])
@pytest.mark.parametrize("prograde", [True, False])
def test_disk_model_defaults_r_in_to_isco(st, prograde: bool) -> None:
    cfg = load_config(CONFIG, {"experiment.parameters.disk.prograde": prograde})
    disk = DiskModel.from_config(st, cfg)
    assert disk.r_in == isco_radius(st, prograde=prograde)
    assert disk.r_out == 20.0 and disk.redshift_power == 4 and disk.prograde is prograde
    explicit = DiskModel.from_params(st, DiskParams(r_in=7.5, r_out=30.0, intensity_law="g3"))
    assert explicit.r_in == 7.5 and explicit.redshift_power == 3


def test_disk_model_rejects_inner_radius_below_photon_orbit() -> None:
    st = kerr(1.0, 0.5)
    r_ph = equatorial_photon_orbit_radius(st, prograde=True)
    with pytest.raises(ValueError):
        DiskModel.from_params(st, DiskParams(r_in=0.9 * r_ph, r_out=20.0))
    with pytest.raises(ValueError):
        DiskModel(r_in=10.0, r_out=5.0)
    with pytest.raises(ValueError):
        DiskModel(r_in=6.0, r_out=20.0, intensity_law="g2")


def test_emissivity_power_law_normalised_and_zero_outside() -> None:
    disk = DiskModel(r_in=6.0, r_out=20.0, emissivity_index=3.0)
    r = np.array([5.9, 6.0, 12.0, 20.0, 20.1])
    values = emissivity(disk, r)
    assert values[0] == 0.0 and values[-1] == 0.0
    assert values[1] == 1.0
    assert abs(values[2] - 2.0**-3) < 1e-15
    assert np.all(np.diff(values[1:4]) < 0.0)
    flat = DiskModel(r_in=6.0, r_out=20.0, emissivity_index=0.0)
    assert np.all(emissivity(flat, r[1:4]) == 1.0)


def test_observed_intensity_uses_selected_power_of_g() -> None:
    r = np.array([6.0, 12.0])
    g = np.array([0.5, 2.0])
    for law, power in INTENSITY_LAWS.items():
        disk = DiskModel(r_in=6.0, r_out=20.0, emissivity_index=2.0, intensity_law=law)
        expected = emissivity(disk, r) * g**power
        assert np.allclose(observed_intensity(disk, r, g), expected, rtol=1e-15)
    assert INTENSITY_LAWS == {"g3": 3, "g4": 4}


def test_event_options_match_disk() -> None:
    disk = DiskModel(r_in=6.0, r_out=20.0)
    ev = event_options(disk)
    assert ev == EventOptions(disk_plane=True, r_in=6.0, r_out=20.0)
