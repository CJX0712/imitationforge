"""全局确定性种子（唯一入口）。

世界观：所有随机源（random / numpy）都由 set_all(seed) 一次性钉死。
demo 两次运行 benchmark.json 核心指标逐位一致，靠的就是它。
作者：晨星
"""

from __future__ import annotations

import random

import numpy as np

_SEED: int = 0
_RNG: np.random.Generator | None = None


def set_all(seed: int = 0) -> None:
    """钉死全局随机源。numpy / random 一次设齐。"""
    global _SEED, _RNG
    _SEED = int(seed)
    random.seed(_SEED)
    np.random.seed(_SEED)
    _RNG = np.random.default_rng(_SEED)


def get_seed() -> int:
    return _SEED


def new_rng(seed: int | None = None) -> np.random.Generator:
    """派生一个独立的 Generator（不打乱全局状态）。"""
    if seed is None:
        seed = _SEED
    return np.random.default_rng(seed)
