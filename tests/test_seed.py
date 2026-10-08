"""确定性：全局 seed 入口 + 独立 RNG。作者：晨星"""

from __future__ import annotations

import numpy as np

from core.seed import get_seed, new_rng, set_all


def test_set_all_deterministic():
    set_all(123)
    a = np.random.randn(5)
    set_all(123)
    b = np.random.randn(5)
    assert np.array_equal(a, b), "set_all 未复现 numpy 随机流"


def test_new_rng_independent():
    r1 = new_rng(7)
    r2 = new_rng(7)
    assert np.array_equal(r1.integers(0, 100, 10), r2.integers(0, 100, 10))
    r3 = new_rng(11)
    assert not np.array_equal(r1.integers(0, 100, 10), r3.integers(0, 100, 10))


def test_get_seed_after_set():
    set_all(20240601)
    assert get_seed() == 20240601
