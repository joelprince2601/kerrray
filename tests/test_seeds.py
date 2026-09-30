"""Seed-control tests (PROJECT.md section 35: the random seed is a recorded input)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from kerrray.utils.seeds import DEFAULT_SEED, set_seed


def test_default_seed_is_zero() -> None:
    assert DEFAULT_SEED == 0
    np.testing.assert_array_equal(set_seed().random(4), set_seed(0).random(4))


def test_same_seed_gives_same_sequence() -> None:
    first = set_seed(42).random(8)
    second = set_seed(42).random(8)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_sequences() -> None:
    assert not np.array_equal(set_seed(0).random(8), set_seed(1).random(8))


def test_returns_numpy_generator() -> None:
    assert isinstance(set_seed(3), np.random.Generator)


def test_legacy_global_generators_are_seeded_too() -> None:
    set_seed(123)
    legacy_numpy, legacy_python = np.random.random(), random.random()
    set_seed(123)
    assert np.random.random() == legacy_numpy
    assert random.random() == legacy_python


def test_numpy_integer_seed_accepted() -> None:
    np.testing.assert_array_equal(set_seed(np.int64(5)).random(3), set_seed(5).random(3))


@pytest.mark.parametrize("bad", [-1])
def test_negative_seed_rejected(bad: int) -> None:
    with pytest.raises(ValueError):
        set_seed(bad)


@pytest.mark.parametrize("bad", [True, 1.5, "0", None])
def test_non_integer_seed_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        set_seed(bad)  # type: ignore[arg-type]
