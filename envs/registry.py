"""环境注册表。作者：晨星"""

from __future__ import annotations

from .cartpole import CartPole
from .gridworld import GridWorld
from .pendulum import Pendulum

REGISTRY = {
    "gridworld": GridWorld,
    "pendulum": Pendulum,
    "cartpole": CartPole,
}

ENV_NAMES = list(REGISTRY.keys())

# 固定整数偏移，避免用 hash()（跨进程不稳定 → 破坏确定性）。
ENV_SEED_OFFSET = {"gridworld": 11, "pendulum": 22, "cartpole": 33}


def make_env(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown env: {name}; choices={ENV_NAMES}")
    return REGISTRY[name]()
