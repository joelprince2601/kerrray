"""Tests for kerrray.validation.convergence and kerrray.validation.conservation (synthetic data, fast)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kerrray.validation.conservation import DriftSummary, worst_drift
from kerrray.validation.convergence import (
    convergence_study,
    loglog_order,
    richardson_order,
    successive_orders,
)


@pytest.mark.parametrize("order", [1.0, 2.0, 4.0])
def test_loglog_order_recovers_power_law(order: float) -> None:
    h = np.array([0.4, 0.2, 0.1, 0.05])
    assert loglog_order(h, 3.0 * h**order) == pytest.approx(order, rel=1e-12)


def test_loglog_order_ignores_unusable_points_and_returns_nan() -> None:
    h = np.array([0.4, 0.2, 0.1])
    err = np.array([1.6e-3, math.nan, 1.0e-4])
    assert loglog_order(h, err) == pytest.approx(2.0, rel=1e-12)
    assert math.isnan(loglog_order([0.1], [1e-3]))
    assert math.isnan(loglog_order([0.1, 0.1], [1e-3, 2e-3]))
    with pytest.raises(ValueError):
        loglog_order([0.1, 0.2], [1e-3])


def test_successive_orders() -> None:
    h = np.array([1.0, 0.5, 0.25])
    orders = successive_orders(h, h**3)
    assert orders == pytest.approx([3.0, 3.0], rel=1e-12)
    assert math.isnan(successive_orders([1.0, 0.5], [1.0, 0.0])[0])


def test_richardson_order_and_extrapolation() -> None:
    q_star, c, p, k = 2.5, 0.7, 4.0, 2.0
    q = [q_star + c * h**p for h in (0.4, 0.2, 0.1)]
    est = richardson_order(*q, ratio=k)
    assert est.order == pytest.approx(p, rel=1e-9)
    assert est.extrapolated == pytest.approx(q_star, abs=1e-12)


def test_richardson_order_without_convergence_is_nan() -> None:
    assert math.isnan(richardson_order(1.0, 1.0, 1.0, 2.0).order)
    assert math.isnan(richardson_order(1.0, 2.0, 4.0, 2.0).order)  # differences grow
    with pytest.raises(ValueError):
        richardson_order(1.0, 0.5, 0.25, 1.0)


def test_convergence_study_noise_floor_and_richardson() -> None:
    levels = [1e-6, 1e-8, 1e-10, 1e-12]
    reference = 5.0
    values = [reference + 3e-1 * lv for lv in levels[:3]] + [reference + 1e-14]
    study = convergence_study("rtol", levels, values, reference, noise_floor=1e-12)
    assert study.errors[-1] < study.noise_floor
    assert study.fitted_order == pytest.approx(1.0, rel=1e-6)  # last level excluded as unresolved
    assert math.isnan(study.local_orders[-1])
    assert study.richardson is not None and study.richardson.ratio == pytest.approx(100.0)
    as_dict = study.as_dict()
    assert as_dict["parameter"] == "rtol" and len(as_dict["values"]) == 4


def test_convergence_study_without_constant_ratio_has_no_richardson() -> None:
    study = convergence_study("h", [0.4, 0.2, 0.15], [1.0, 0.9, 0.85], 0.8)
    assert study.richardson is None


def test_drift_summary_helpers() -> None:
    a = DriftSummary(energy=0.0, lz=1e-12, carter=1e-9, null=2e-9)
    b = DriftSummary(energy=1e-10, lz=0.0, carter=5e-9, null=1e-9)
    w = worst_drift([a, b])
    assert w.as_dict() == {"energy": 1e-10, "lz": 1e-12, "carter": 5e-9, "null": 2e-9}
    assert w.worst == 5e-9
    assert w.within(1e-8) and not w.within(1e-9)
    assert not DriftSummary(math.nan, 0.0, 0.0, 0.0).within(1.0)
    with pytest.raises(ValueError):
        worst_drift([])
